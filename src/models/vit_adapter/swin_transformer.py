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


class AdapterSwinTransformer(SwinTransformer):
    def __init__(
        self, adapter_config, img_size=224, patch_size=4, in_chans=3,
        num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., qkv_bias=True,
        qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
        use_checkpoint=False, **kwargs
    ):
        super(AdapterSwinTransformer, self).__init__(
            img_size, patch_size, in_chans, num_classes, embed_dim, depths,
            num_heads, window_size, mlp_ratio, qkv_bias, qk_scale, drop_rate,
            attn_drop_rate, drop_path_rate, norm_layer, ape, patch_norm,
            use_checkpoint, **kwargs
        )
        self.adapter_config = adapter_config

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
                block_module=AdapterSwinTransformerBlock,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                adapter_type=self.adapter_config.STYLE, 
                adapter_reduction_factor=self.adapter_config.REDUCATION_FACTOR,
            )
            self.layers.append(layer)


class AdapterSwinTransformerBlock(SwinTransformerBlock):
    def __init__(
        self, dim, input_resolution,
        num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True,
        qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
        norm_layer=nn.LayerNorm, adapter_type="Pfeiffer", adapter_reduction_factor=8
    ):
        super(AdapterSwinTransformerBlock, self).__init__(
            dim, input_resolution, num_heads, window_size,
            shift_size, mlp_ratio, qkv_bias, qk_scale, drop,
            attn_drop, drop_path, act_layer, norm_layer)
        
        self.adapter_type = adapter_type
        self.adapter_reduction_factor = adapter_reduction_factor
        
        if adapter_type == "Pfeiffer":
            self.adapter_downsample = nn.Linear(
                dim,
                dim // adapter_reduction_factor
            )
            self.adapter_upsample = nn.Linear(
                dim // adapter_reduction_factor,
                dim
            )
            self.adapter_act_fn = torch.nn.functional.gelu

            nn.init.zeros_(self.adapter_downsample.weight)
            nn.init.zeros_(self.adapter_downsample.bias)

            nn.init.zeros_(self.adapter_upsample.weight)
            nn.init.zeros_(self.adapter_upsample.bias)
        else:
            raise ValueError("Other adapter styles are not supported.")
        
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        shortcut = x
        x = self.drop_path(self.mlp(self.norm2(x)))
            
        # start to insert adapter layers...
        adpt = self.adapter_downsample(x)
        adpt = self.adapter_act_fn(adpt)
        adpt = self.adapter_upsample(adpt)
        x = adpt + x
        # ...end
        x = shortcut + x
        return x