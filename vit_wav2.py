"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict
import pywt
import torch
import torch.nn as nn
import numpy as np
from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class WavLayer(nn.Module):
    def __init__(self, level, h, w):
        super(WavLayer, self).__init__()
        self.level = level
        self.high_freq_weight = nn.Parameter(torch.randn(h, w) * 0.02)
        self.low_freq_weight = nn.Parameter(torch.randn(h // (2 ** level), w // (2 ** level)) * 0.02)
        self.conv = nn.Conv2d(2, 16, kernel_size=3, padding=1)  # 输入通道为2

    def forward(self, x):
        x = x[:, 0]  # 取第一个通道
        B, h, w = x.shape
        high_freq_output = torch.zeros_like(x)
        low_freq_output = torch.zeros(B, h // (2 ** self.level), w // (2 ** self.level), device=x.device)
        
        for i in range(B):
            data = x[i].cpu().numpy()
            coeffs = pywt.wavedec2(data, 'db2', mode='periodization', level=self.level)
            coeffs[0] /= np.abs(coeffs[0]).max()
            for detail_level in range(self.level):        
                coeffs[detail_level + 1] = [
                    d / np.abs(d).max() for d in coeffs[detail_level + 1]
                ]
            low_freq = torch.from_numpy(coeffs[0]).to(x.device)  # [28, 28]
            arr, _ = pywt.coeffs_to_array(coeffs)  # [224, 224]
            arr = torch.from_numpy(arr).to(x.device)

            high_freq_output[i] = arr * self.high_freq_weight
            low_freq_output[i] = low_freq * self.low_freq_weight

        # 上采样低频到高频尺寸
        low_freq_output = low_freq_output.unsqueeze(1)  # [B, 1, 28, 28]
        low_freq_output = torch.nn.functional.interpolate(low_freq_output, size=(h, w), mode='bilinear', align_corners=False)  # [B, 1, 224, 224]
        high_freq_output = high_freq_output.unsqueeze(1)  # [B, 1, 224, 224]

        # 拼接高频和低频
        output = torch.cat([high_freq_output, low_freq_output], dim=1)  # [B, 2, 224, 224]
        output = self.conv(output)  # [B, 16, 224, 224]
        return output

class WavCon(nn.Module):
    def __init__(self):
        super(WavCon, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x

class FusionModule(nn.Module):
    def __init__(self, embed_dim=768):
        super(FusionModule, self).__init__()
        self.wav_fc = nn.Linear(embed_dim, embed_dim)
        self.vit_fc = nn.Linear(embed_dim, embed_dim)
        self.attn = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x_vit, x_wav):
        x_vit = self.vit_fc(x_vit)
        x_wav = self.wav_fc(x_wav)
        combined = torch.cat([x_vit, x_wav], dim=-1)
        weights = self.attn(combined)
        out = x_vit * weights[:, 0].unsqueeze(-1) + x_wav * weights[:, 1].unsqueeze(-1)
        return out

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, level=3):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        
        self.wav = WavLayer(level=level, h=img_size, w = img_size)
        self.wav_con = WavCon()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Flatten(start_dim=1),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 768),
        )

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.fusion = FusionModule(embed_dim=embed_dim)

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]
    def forward_wav_features(self, x):
        x = self.wav(x)
        print("WavLayer output shape:", x.shape)
        x = self.wav_con(x)
        print("aaaWavLayer output shape:", x.shape)
        x = self.classifier(x)
        print("cccWavLayer output shape:", x.shape)
        return x

    def forward(self, x):
        print("input shape: ", x.shape)
        y = self.forward_wav_features(x)
        print("Input to wav_con shape:", y.shape)
        print("y.shape: ", y.shape)
        x = self.forward_features(x)
        print("y.shape: ", y.shape)
        print("x.shape: ", x.shape)

        # x = x+y
        x = self.fusion(x,y)

        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        
        print("output shape: ", x.shape)
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        # print("initing ",m)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        # print("initing ",m)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        # print("initing ",m)
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224(num_classes: int = 1000):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1cxBgZJJ6qUWPSBNcE4TdRQ  密码: qqt8
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    return total_params, trainable_params

class ScoreCAMModel(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model
        # 提取特征层，移除head以获取tokens
        self.feature_extractor = nn.Sequential(*list(original_model.children())[:-1])  # 取到norm

    def forward(self, x):
        # 返回最后一层norm后的tokens (B, N+1, D)
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        if self.model.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.model.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)
        for blk in self.model.blocks:
            x = blk(x)
        x = self.model.norm(x)  # (B, 197, 768) for ViT-B/16
        return x

def reshape_transform(tensor, height=14, width=14):
    # 移除cls_token (B, N+1, D) -> (B, N, D)
    result = tensor[:, 1:, :]  # (1, 196, 768) for 14x14 patches
    # Reshape to (B, H, W, D) -> (B, D, H, W)
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def generate_scorecam(model, input_tensor, target_class, save_path='scorecam_heatmap.png'):
    # 目标层 (最后一层norm1或norm)
    target_layers = [model.blocks[-1].norm1]  # 或model.norm

    # 创建ScoreCAM实例
    scorecam = ScoreCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    # 生成热图
    grayscale_cam = scorecam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_class)])[0, :]

    # 叠加到原图像
    rgb_img = input_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()  # (3, 224, 224) -> (224, 224, 3)
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())  # 归一化 [0,1]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # 可视化
    plt.imshow(visualization)
    plt.axis('off')
    plt.title(f'Score-CAM for Class {target_class}')
    plt.savefig(save_path)
    plt.show()

    return visualization

def main():
    model = vit_base_patch16_224(num_classes=5)
    # model.load_state_dict(torch.load('your_model_checkpoint.pth'))
    model.eval()
    model.cuda()

    # 确保WavLayer的参数在GPU上
    model.wav = WavLayer(level=2, h=224, w=224).cuda()
    model.wav_con = model.wav_con.cuda()
    model.classifier = model.classifier.cuda()
    model.fusion = model.fusion.cuda()

    # 动态设置wavelet并确保设备一致
    def set_wavelet(wavelet):
        def forward_with_wavelet(x):
            x = x[:, 0]  # (B, C, H, W) -> (B, H, W)
            B, h, w = x.shape
            high_freq_output = torch.zeros_like(x, device=x.device)  # 在GPU上初始化
            low_freq_output = torch.zeros(B, h // (2 ** model.wav.level), w // (2 ** model.wav.level), device=x.device)
            
            for i in range(B):
                data = x[i].cpu().numpy()  # 必须移到CPU，因为pywt不支持GPU
                coeffs = pywt.wavedec2(data, wavelet, mode='periodization', level=model.wav.level)
                coeffs[0] /= np.abs(coeffs[0]).max()
                for detail_level in range(model.wav.level):
                    coeffs[detail_level + 1] = [d / np.abs(d).max() for d in coeffs[detail_level + 1]]
                low_freq = torch.from_numpy(coeffs[0]).to(x.device)  # 移回GPU
                arr, _ = pywt.coeffs_to_array(coeffs)
                arr = torch.from_numpy(arr).to(x.device)  # 移回GPU

                high_freq_output[i] = arr * model.wav.high_freq_weight.to(x.device)  # 确保weight在GPU
                low_freq_output[i] = low_freq * model.wav.low_freq_weight.to(x.device)

            low_freq_output = low_freq_output.unsqueeze(1)
            low_freq_output = torch.nn.functional.interpolate(low_freq_output, size=(h, w), mode='bilinear', align_corners=False)
            high_freq_output = high_freq_output.unsqueeze(1)
            output = torch.cat([high_freq_output, low_freq_output], dim=1)
            return model.wav.conv(output)
        model.wav.forward = forward_with_wavelet

    set_wavelet('db2')

    img_path = 'tulip.jpg'
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(img).unsqueeze(0).cuda()

    with torch.no_grad():
        outputs = model(input_tensor)
        pred_class = outputs.argmax(dim=1).item()
        print(f'Predicted class: {pred_class}')

    visualization = generate_scorecam(
        model=ScoreCAMModel(model),
        input_tensor=input_tensor,
        target_class=pred_class,
        save_path=f'scorecam_class_{pred_class}_db2_level2.png'
    )

    high_act_ratio = (visualization[..., 0] > 0.5).sum() / visualization[..., 0].size
    print(f'High activation ratio: {high_act_ratio:.2f}')

if __name__ == '__main__':
    model1 = vit_base_patch16_224()
    count_parameters(model1)
    x = torch.randn(3,3,224,224)
    y = model1(x)
    print(y.shape)

    main()
