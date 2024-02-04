import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

from .vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model

__all__ = [
    "deit_tscam_tiny_patch16_224",
    "deit_tscam_small_patch16_224",
    "deit_tscam_base_patch16_224",
]

class TSCAM(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Conv2d(384, 256, 1, bias=False)
        self.head.apply(self._init_weights)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0:1]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to return patch embedding outputs
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)

        attn_weights = []
        for blk in self.blocks:
            x, weights = blk(x)
            attn_weights.append(weights)
        x = self.norm(x)
        if self.training:
            embed_ = x[:, 1:].reshape(B, 14, 14, x.shape[2]).permute(0, 3, 1, 2)  # torch.Size([16, 14, 14, 384])
            trans_embed = self.embed(embed_)
        else:
            trans_embed = None
        return x[:, 0], x[:, 1:], attn_weights, trans_embed

    def forward(self, x, return_cam=False):
        B, c, h_orig, w_orig = x.shape
        x_cls, x_patch, attn_weights, trans_embed = self.forward_features(x)
        n, p, c = x_patch.shape

        if h_orig != w_orig:
            h0 = h_orig // self.patch_embed.patch_size[0]
            w0 = w_orig // self.patch_embed.patch_size[0]
            x_patch = torch.reshape(x_patch, [n, h0, w0, c])
        else:
            x_patch = torch.reshape(x_patch, [n, int(p**0.5), int(p**0.5), c])
        x_patch = x_patch.permute([0, 3, 1, 2])
        x_patch = x_patch.contiguous()

        x_patch = self.head(x_patch)
        attn_weights = torch.stack(attn_weights)  # 12 * B * H * N * N
        attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N
       
        attn_weights = attn_weights.permute(1, 0, 2, 3)

        if self.training:
            return x_patch, attn_weights, trans_embed
        else:
            feature_map = x_patch.clone().detach()  
            b, c, h, w = feature_map.shape
            cams = attn_weights.sum(1)[:, 0, 1:].reshape([b, h, w]).unsqueeze(1)  # summation for head
            cams_ = cams * feature_map  # B * C * 14 * 14
            cams_ = F.relu(cams_)
            cams = F.relu(cams)
            if return_cam is True:
                cams_ = cams_[0] + cams_[1].flip(-1)
                cams = cams[0] + cams[1].flip(-1)
            return [cams_, cams]

    def trainable_parameters(self):
        parameters_list = list(self.parameters())
        return (parameters_list[0:-2], parameters_list[-2:])

@register_model
def Net(pretrained=False, **kwargs):
    model = TSCAM(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth", map_location="cpu", check_hash=True)["model"]
        model_dict = model.state_dict()
        for k in ["head.weight", "head.bias", "head_dist.weight", "head_dist.bias"]:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model