#!/usr/bin/env python3
import numpy as np
import torch
import os
from .vit_backbones.swin_transformer import SwinTransformer
from .vit_backbones.vit import VisionTransformer

from .vit_prompt.vit import PromptedVisionTransformer
from .vit_prompt.swin_transformer import PromptedSwinTransformer

from .vit_adapter.vit import ADPT_VisionTransformer
from .vit_adapter.swin_transformer import AdapterSwinTransformer

from .vit_lora.vit import LORA_VisionTransformer
from .vit_lora.swin_transformer import LoraSwinTransformer


MODEL_ZOO = {
    "swinb_imagenet_224": "swin_base_patch4_window7_224.pth",
    "ARES_ViT_base_patch16_224_AT": "ARES_ViT_base_patch16_224_AT.pth",
    "ARES_Swin_base_patch4_window7_224_AT": "ARES_Swin_base_patch4_window7_224_AT.pth",
    "pytorch_vitb16_224_imagenet": "vit_b_16-c867db91.pth",
    "CLIP-ViT-B-16-laion2B-s34B-b88K" : "open_clip_pytorch_model.bin"
}


def build_swin_model(model_type, crop_size, prompt_cfg, model_root, adapter_cfg, lora_cfg):
    if prompt_cfg is not None:
        return _build_prompted_swin_model(
            model_type, crop_size, prompt_cfg, model_root)
    elif adapter_cfg is not None:
        return _build_adapter_swin_model(
            model_type, crop_size, adapter_cfg, model_root)
    elif lora_cfg is not None:
        return _build_lora_swin_model(
            model_type, crop_size, lora_cfg, model_root)
    else:
        return _build_swin_model(model_type, crop_size, model_root)


def _build_prompted_swin_model(model_type, crop_size, prompt_cfg, model_root):
    model = PromptedSwinTransformer(
        prompt_cfg,
        img_size=crop_size,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        drop_path_rate=0.5,
        num_classes=-1,
    )
    embed_dim = 128
    num_layers = 4
    feat_dim = int(embed_dim * 2 ** (num_layers - 1))
    # load checkpoint
    model_w = os.path.join(model_root, MODEL_ZOO[model_type])
    checkpoint = torch.load(model_w, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    model.load_state_dict(state_dict, strict=False)

    return model, feat_dim


def _build_swin_model(model_type, crop_size, model_root):
    model = SwinTransformer(
        img_size=crop_size,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        drop_path_rate=0.5,
        num_classes=-1,
    )
    embed_dim = 128
    num_layers = 4
    feat_dim = int(embed_dim * 2 ** (num_layers - 1))
    # load checkpoint
    model_w = os.path.join(model_root, MODEL_ZOO[model_type])
    checkpoint = torch.load(model_w, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    model.load_state_dict(state_dict, strict=False)

    return model, feat_dim


def _build_adapter_swin_model(model_type, crop_size, adapter_cfg, model_root):
    model = AdapterSwinTransformer(
        adapter_cfg,
        img_size=crop_size,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        drop_path_rate=0.5,
        num_classes=-1,
    )
    embed_dim = 128
    num_layers = 4
    feat_dim = int(embed_dim * 2 ** (num_layers - 1))
    # load checkpoint
    model_w = os.path.join(model_root, MODEL_ZOO[model_type])
    checkpoint = torch.load(model_w, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    model.load_state_dict(state_dict, strict=False)

    return model, feat_dim


def _build_lora_swin_model(model_type, crop_size, lora_cfg, model_root):
    model = LoraSwinTransformer(
        lora_cfg,
        img_size=crop_size,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        drop_path_rate=0.5,
        num_classes=-1,
    )
    embed_dim = 128
    num_layers = 4
    feat_dim = int(embed_dim * 2 ** (num_layers - 1))
    # load checkpoint
    model_w = os.path.join(model_root, MODEL_ZOO[model_type])
    checkpoint = torch.load(model_w, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    model.load_state_dict(state_dict, strict=False)

    return model, feat_dim


def build_vit_sup_models(
    model_type, crop_size, prompt_cfg=None, model_root=None, adapter_cfg=None, lora_cfg=None, load_pretrain=True, vis=False
):
    # image size is the size of actual image
    m2featdim = {
        "ARES_ViT_base_patch16_224_AT" : 768,
        "pytorch_vitb16_224_imagenet": 768,
        "CLIP-ViT-B-16-laion2B-s34B-b88K":768
    }
    if prompt_cfg is not None:
        model = PromptedVisionTransformer(
            prompt_cfg, model_type,
            crop_size, num_classes=-1, vis=vis
        )
    elif adapter_cfg is not None:
        model = ADPT_VisionTransformer(model_type, crop_size, num_classes=-1, adapter_cfg=adapter_cfg)
    elif lora_cfg is not None:
        model = LORA_VisionTransformer(model_type, crop_size, num_classes=-1, lora_cfg=lora_cfg)
    else:
        model = VisionTransformer(
            model_type, crop_size, num_classes=-1, vis=vis)
    
    if load_pretrain:
        if model_type == 'ARES_ViT_base_patch16_224_AT':
            model.adv_load_from(torch.load(os.path.join(model_root, MODEL_ZOO[model_type]), map_location='cpu'), m2featdim[model_type])
        elif model_type == 'pytorch_vitb16_224_imagenet':
            model.pytorch_load_from(torch.load(os.path.join(model_root, MODEL_ZOO[model_type]), map_location='cpu'), m2featdim[model_type])
        elif model_type == 'CLIP-ViT-B-16-laion2B-s34B-b88K':
            model.clip_load_from(torch.load(os.path.join(model_root, MODEL_ZOO[model_type]), map_location='cpu'), m2featdim[model_type])
        else:
            model.load_from(np.load(os.path.join(model_root, MODEL_ZOO[model_type])))

    return model, m2featdim[model_type]

