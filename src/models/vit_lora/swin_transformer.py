#!/usr/bin/env python3
"""
swin transformer with prompt
"""
import math
import torch
import torch.nn as nn
import torchvision as tv

from functools import reduce
from operator import mul
from torch.nn import Conv2d, Dropout

from timm.models.layers import to_2tuple, trunc_normal_

from ..vit_backbones.swin_transformer import (
    BasicLayer, PatchMerging, SwinTransformer, SwinTransformerBlock,
    window_partition, window_reverse, WindowAttention
    )
from ...utils import logging
logger = logging.get_logger("RoLI")


class LoraSwinTransformer(SwinTransformer):
    def __init__(
        self, lora_config, img_size=224, patch_size=4, in_chans=3,
        num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., qkv_bias=True,
        qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
        use_checkpoint=False, **kwargs
    ):
        super(LoraSwinTransformer, self).__init__(
            img_size, patch_size, in_chans, num_classes, embed_dim, depths,
            num_heads, window_size, mlp_ratio, qkv_bias, qk_scale, drop_rate,
            attn_drop_rate, drop_path_rate, norm_layer, ape, patch_norm,
            use_checkpoint, **kwargs
        )
        self.lora_config = lora_config

        # build layers
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    self.patches_resolution[0] // (2 ** i_layer),
                    self.patches_resolution[1] // (2 ** i_layer)
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                block_module=LoraSwinTransformerBlock,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                lora_rank=self.lora_config.RANK,
                lora_droptout=self.lora_config.DROPOUT,
                lora_alpha=self.lora_config.ALPHA,
                lora_zero_init=self.lora_config.ZEROINIT
            )
            self.layers.append(layer)


class LoraSwinTransformerBlock(SwinTransformerBlock):
    def __init__(
        self, dim, input_resolution,
        num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True,
        qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
        norm_layer=nn.LayerNorm, lora_rank=16, lora_droptout=0.1, lora_alpha=16,
        lora_zero_init=False
    ):
        super(LoraSwinTransformerBlock, self).__init__(
            dim, input_resolution, num_heads, window_size,
            shift_size, mlp_ratio, qkv_bias, qk_scale, drop,
            attn_drop, drop_path, act_layer, norm_layer)
        
        self.attn = LoraWindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            lora_rank=lora_rank, lora_droptout=lora_droptout, lora_alpha=lora_alpha, 
            lora_zero_init=lora_zero_init)


class LoraWindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 lora_rank=16, lora_droptout=0.1, lora_alpha=16, lora_zero_init=False):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.lora_rank = lora_rank
        self.lora_droptout = lora_droptout

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if lora_rank > 0:
            self.query_lora_A = nn.Linear(dim, lora_rank, bias=False)
            self.query_lora_B = nn.Linear(lora_rank, dim, bias=False)
            self.value_lora_A = nn.Linear(dim, lora_rank, bias=False)
            self.value_lora_B = nn.Linear(lora_rank, dim, bias=False)

            if lora_zero_init:
                nn.init.zeros_(self.query_lora_A.weight)
                nn.init.zeros_(self.value_lora_A.weight)
            else:
                nn.init.kaiming_uniform_(self.query_lora_A.weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.value_lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.query_lora_B.weight)
            nn.init.zeros_(self.value_lora_B.weight)

            self.lora_scaling = lora_alpha / lora_rank

        if lora_droptout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_droptout)
        else:
            self.lora_dropout = lambda x: x

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        if self.lora_rank > 0:
            q_lora = self.lora_scaling * self.query_lora_B(self.query_lora_A(self.lora_dropout(x)))
            q = q + q_lora.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.lora_rank > 0:

            v_lora = self.lora_scaling * self.value_lora_B(self.value_lora_A(self.lora_dropout(x)))
            v = v + v_lora.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # not supported
        # # calculate flops for 1 window with token length of N
        # flops = 0
        # # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        # # attn = (q @ k.transpose(-2, -1))
        # flops += self.num_heads * N * (self.dim // self.num_heads) * N
        # #  x = (attn @ v)
        # flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # # x = self.proj(x)
        # flops += N * self.dim * self.dim
        return -1

    def train(self, mode=True):
        # set train status for this class: disable all but the lora-related modules
        if mode:
            # training:
            # first set all to eval and set the prompt to train later
            for module in self.children():
                module.train(False)
            self.lora_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)