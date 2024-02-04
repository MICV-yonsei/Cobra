
def InfoNCE_v1(cam1, trans_embed, label_bg): # [B, 21, 14, 14], [B, 14, 14], [196, 128]
    B, C, N, _ = cam1.shape

    scores = F.softmax(cam1 * label_bg, dim=1)
    pseudo_score, pseudo_label = torch.max(scores, dim=1) 
    cam1 = cam1.reshape(B, C, -1)
    pseudo_label = pseudo_label.reshape(B, -1)

    cam = [cam1[i, pseudo_label[i]]for i in range(B)]
    cam = torch.stack(cam, dim = 0)#.reshape(B, -1) 

    top_values, top_indices = torch.topk(cam, k=20, dim = -1, largest = True) # [B,196, 20]
    under_values, under_indices = torch.topk(cam, k=20, dim = -1, largest = False)  # [B,196, 20]

    trans_embed = trans_embed.transpose(1,2) # [16, 196, 128]

    pos = []
    neg = []
    for i in range(B):
        pos.append(trans_embed[i, top_indices[i]])
        neg.append(trans_embed[i, under_indices[i]])
    pos1 = torch.stack(pos, dim = 0) # [B, 196, 20, 128]
    neg1 = torch.stack(neg, dim = 0) # [B, 196, 20, 128]     

    for i in range(N):
        loss = 0
        embed = trans_embed[:, i].unsqueeze(-1)
        pos_inner = pos1[:, i] @ embed # [B, 20, 1]
        neg_inner = embed @ neg1[:, i].transpose(1,2) # [B, 20, 1]
        
        A = torch.exp(pos_inner.squeeze(-1))
        B = torch.sum(torch.exp(neg_inner.squeeze(-1)), dim = -1, keepdim = True)
        loss += torch.sum(- torch.log( A / (A + B)))
    return loss /N
