import torch
from torch.nn import functional as F
import sys

def cll_v1(args, cam_cnn, transformer_embed, label_bg):
    '''
    Semantic Aware Projection
    '''
    # *********Hyperparams********* #
    top_num = args.top_bot_k[0]
    bottom_num = args.top_bot_k[1]
    tau = args.tau

    B, C, N, _ = cam_cnn.shape
    N2 = N * N  # for 196
    
    scores = F.softmax(cam_cnn * label_bg, dim=1)  # [B, 21, 14, 14]; Softmax on class level
    pseudo_score, pseudo_label = torch.max(scores, dim=1)  # [B, 14, 14]; Select best class-score on CNN CAMs by pixel
    cam_cnn = cam_cnn.reshape(B, C, -1) # [B, C, 196]
    pseudo_label = pseudo_label.reshape(B, -1) # [B, 196]

    cam = [cam_cnn[i, pseudo_label[i]] for i in range(B)]  # [B, 196, 196]
    cam = torch.stack(cam, dim=0)

    top_values, top_indices = torch.topk(
        cam, k=top_num, dim=-1, largest=True)  # [B, 196, 20]
    bottom_values, bottom_indices = torch.topk(
        cam, k=bottom_num, dim=-1, largest=False)  # [B, 196, 20]

    transformer_embed = transformer_embed.transpose(1, 2)  # [B, 196, 128]

    pos_init = []
    neg_init = []

    for i in range(B):
        pos_init.append(transformer_embed[i, top_indices[i]])
        neg_init.append(transformer_embed[i, bottom_indices[i]])

    pos = torch.stack(pos_init, dim=0)  # [B, 196, 20, 128]
    neg = torch.stack(neg_init, dim=0)  # [B, 196, 20, 128]
    
    # Computing Loss
    loss = torch.zeros((1)).cuda()
    '''
    basically fomula of loss = X/(X+Y)
    '''
    for i in range(N2):

        main_vector_tf = transformer_embed[:, i].unsqueeze(-1)

        # X where of numerator
        pos_inner = pos[:, i] @ main_vector_tf  # [B, 20, 1]
        X = torch.exp(pos_inner.squeeze(-1) / tau)

        # Y where of denominator
        neg_inner = neg[:, i] @ main_vector_tf  # [B, 20, 1]
        Y = torch.sum((torch.exp(neg_inner.squeeze(-1)) / tau),
                      dim=-1, keepdim=True)

        # X/(X+Y)
        loss += torch.sum(-torch.log(X / (X + Y)))

    return loss / (N2 * (top_num * B))


def cll_v2(args, attn_weights, cnn_embed):
    '''
    Class Aware Projection
    '''
    
    # *********Hyperparams*********
    top_num = args.top_bot_k[2]
    bottom_num = args.top_bot_k[3]
    tau = args.tau
    # *****************************

    attn_weights = attn_weights[:, 1:, 1:] # P2P Attention Score Excepted Background Token
    B, N, N = attn_weights.shape

    top_values, top_indices = torch.topk(
        attn_weights, k=top_num, dim=-1, largest=True)  # [B, 196, 20]
    bottom_values, bottom_indices = torch.topk(
        attn_weights, k=bottom_num, dim=-1, largest=False)  # [B, 196, 20]

    cnn_embed = cnn_embed.transpose(1, 2)  # [B, 196, 128]

    pos_init = []
    neg_init = []

    for i in range(B):
        pos_init.append(cnn_embed[i, top_indices[i]])
        neg_init.append(cnn_embed[i, bottom_indices[i]])

    pos = torch.stack(pos_init, dim=0)  # [B, 196, k, 128]
    neg = torch.stack(neg_init, dim=0)  # [B, 196, k, 128]

    # Computing Loss
    loss = torch.zeros(1).cuda()
    '''
    basically fomula of loss is X/(X+Y)
    '''
    for i in range(N):

        main_vector_tf = cnn_embed[:, i].unsqueeze(-1) # main vector for all batch

        # X where of numerator
        pos_inner = pos[:, i] @ main_vector_tf  # [B, 20, 1], matmul
        X = torch.exp(pos_inner.squeeze(-1) / tau)

        # Y where of denominator
        neg_inner = neg[:, i] @ main_vector_tf  # [B, 20, 1], matmul
        Y = torch.sum(torch.exp(neg_inner.squeeze(-1) / tau),
                      dim=-1, keepdim=True)

        # X/(X+Y)
        loss += torch.sum(-torch.log(X / (X + Y)))

    # loss /= N * B * top_num
    loss /= N * B * top_num
    return loss

def loss_CAM(cam_cnn_224, cam_tf_224, label):
    cam_cnn_224_classes = cam_cnn_224[:, 1:]
    cam_tf_224_classes = cam_tf_224[:, 1:]
    loss_interCAM = 0
    for i in range(len(cam_cnn_224_classes)):
        valid_cat = torch.nonzero(label[i])[:, 0]
        cam_cnn_224_class = cam_cnn_224_classes[i, valid_cat]
        cam_tf_224_class = cam_tf_224_classes[i, valid_cat]
        loss_bg = torch.mean(torch.abs(cam_cnn_224[i][0] - cam_tf_224[i][0]))
        loss_inter = torch.mean(torch.abs(cam_cnn_224_class - cam_tf_224_class))
        # loss = (loss_bg + loss_inter)/2
        loss = loss_inter
        loss_interCAM += loss
    loss_interCAM1 = loss_interCAM / len(cam_cnn_224)
    return loss_interCAM1
