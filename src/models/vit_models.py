#!/usr/bin/env python3

"""
ViT-related models
Note: models return logits instead of prob
"""
import torch
import torch.nn as nn

from .build_vit_backbone import (
    build_vit_sup_models, build_swin_model,
)
from .mlp import MLP
from ..utils import logging
logger = logging.get_logger("RoLI")


class ViT(nn.Module):
    """ViT-related model."""

    def __init__(self, cfg, load_pretrain=True, vis=False):
        super(ViT, self).__init__()

        if "prompt" in cfg.MODEL.TRANSFER_TYPE:
            prompt_cfg = cfg.MODEL.PROMPT
        else:
            prompt_cfg = None

        if cfg.MODEL.TRANSFER_TYPE != "end2end" and "prompt" not in cfg.MODEL.TRANSFER_TYPE:
            # linear, adapter, lora
            self.froze_enc = True
        else:
            # prompt, end2end, bias
            self.froze_enc = False
        
        if cfg.MODEL.TRANSFER_TYPE == "adapter":
            adapter_cfg = cfg.MODEL.ADAPTER
        else:
            adapter_cfg = None

        if cfg.MODEL.TRANSFER_TYPE == "lora":
            lora_cfg = cfg.MODEL.LORA
        else:
            lora_cfg = None

        self.build_backbone(
            prompt_cfg, cfg, adapter_cfg, load_pretrain, lora_cfg, vis=vis)
        self.cfg = cfg
        self.setup_head(cfg)

    def build_backbone(self, prompt_cfg, cfg, adapter_cfg, load_pretrain, lora_cfg, vis):
        transfer_type = cfg.MODEL.TRANSFER_TYPE
        self.enc, self.feat_dim = build_vit_sup_models(
            cfg.DATA.FEATURE, cfg.DATA.CROPSIZE, prompt_cfg, cfg.MODEL.MODEL_ROOT, adapter_cfg, lora_cfg, load_pretrain, vis
        )

        if transfer_type == "linear":
            for k, p in self.enc.named_parameters():
                p.requires_grad = False

        elif transfer_type == "tinytl-bias":
            for k, p in self.enc.named_parameters():
                if 'bias' not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k:
                    p.requires_grad = False

        elif transfer_type == "end2end":
            logger.info("Enable all parameters update during training")

        elif transfer_type == "lora":
            for k, p in self.enc.named_parameters():
                if "lora" not in k:
                    p.requires_grad = False

        elif transfer_type == "adapter":
            for k, p in self.enc.named_parameters():
                if "adapter" not in k:
                    p.requires_grad = False
        else:
            raise ValueError("transfer type {} is not supported".format(
                transfer_type))

    def setup_head(self, cfg):
        self.head = MLP(
            input_dim=self.feat_dim,
            mlp_dims=[self.feat_dim] * self.cfg.MODEL.MLP_NUM + \
                [cfg.DATA.NUMBER_CLASSES], # noqa
            special_bias=True
        )

    def forward(self, x, return_feature=False):
        if self.froze_enc and self.enc.training:
            self.enc.eval()
        x = self.enc(x)  # batch_size x self.feat_dim

        if return_feature:
            return x, x
        x = self.head(x)

        return x
    
    def forward_cls_layerwise(self, x):
        cls_embeds = self.enc.forward_cls_layerwise(x)
        return cls_embeds

    def get_features(self, x):
        """get a (batch_size, self.feat_dim) feature"""
        x = self.enc(x)  # batch_size x self.feat_dim
        return x


class Swin(ViT):
    """Swin-related model."""

    def __init__(self, cfg, vis):
        super(Swin, self).__init__(cfg)

    def build_backbone(self, prompt_cfg, cfg, adapter_cfg, load_pretrain, lora_cfg, vis):
        transfer_type = cfg.MODEL.TRANSFER_TYPE
        self.enc, self.feat_dim = build_swin_model(
            cfg.DATA.FEATURE, cfg.DATA.CROPSIZE,
            prompt_cfg, cfg.MODEL.MODEL_ROOT, adapter_cfg, lora_cfg
        )

        if transfer_type == "linear":
            for k, p in self.enc.named_parameters():
                p.requires_grad = False

        elif transfer_type == "tinytl-bias":
            for k, p in self.enc.named_parameters():
                if 'bias' not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k:
                    p.requires_grad = False

        elif transfer_type == "end2end":
            logger.info("Enable all parameters update during training")

        elif transfer_type == "lora":
            for k, p in self.enc.named_parameters():
                if "lora" not in k:
                    p.requires_grad = False

        elif transfer_type == "adapter":
            for k, p in self.enc.named_parameters():
                if "adapter" not in k:
                    p.requires_grad = False
        else:
            raise ValueError("transfer type {} is not supported".format(
                transfer_type))

