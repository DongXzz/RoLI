#!/usr/bin/env python3
"""
vit with adapter
"""
import copy
import numpy as np
import torch
import torch.nn as nn

from scipy import ndimage

from torch.nn import Linear, LayerNorm
from ..vit_backbones.vit import *
from ...utils import logging
logger = logging.get_logger("RoLI")


class LORA_Attention(nn.Module):
    def __init__(self, config, vis, lora_cfg):
        super(LORA_Attention, self).__init__()
        self.vis = vis
        self.lora_cfg = lora_cfg
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        if lora_cfg.RANK > 0:
            self.query_lora_A = nn.Linear(config.hidden_size, lora_cfg.RANK, bias=False)
            self.query_lora_B = nn.Linear(lora_cfg.RANK, config.hidden_size, bias=False)
            self.value_lora_A = nn.Linear(config.hidden_size, lora_cfg.RANK, bias=False)
            self.value_lora_B = nn.Linear(lora_cfg.RANK, config.hidden_size, bias=False)

            nn.init.kaiming_uniform_(self.query_lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.query_lora_B.weight)
            nn.init.kaiming_uniform_(self.value_lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.value_lora_B.weight)

            self.scaling = lora_cfg.ALPHA / lora_cfg.RANK

        if lora_cfg.DROPOUT > 0.:
            self.lora_dropout = nn.Dropout(p=lora_cfg.DROPOUT)
        else:
            self.lora_dropout = lambda x: x

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states) # B, num_patches, head_size*num_head
        if self.lora_cfg.RANK > 0:
            mixed_query_layer += self.scaling * self.query_lora_B(self.query_lora_A(self.lora_dropout(hidden_states)))
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        if self.lora_cfg.RANK > 0:
            mixed_value_layer += self.scaling * self.value_lora_B(self.value_lora_A(self.lora_dropout(hidden_states)))

        query_layer = self.transpose_for_scores(mixed_query_layer) # B, num_head, num_patches, head_size
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer) # B, num_head, num_patches, head_size

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # B, num_head, num_patches, num_patches
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        

        attention_probs = self.softmax(attention_scores) # B, num_head, num_patches(query), num_patches(key)


        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer) # B, num_head, num_patches, head_size
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class LORA_Block(nn.Module):
    def __init__(self, config, vis, lora_cfg):
        super(LORA_Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = LORA_Attention(config, vis, lora_cfg)

    def forward(self, x):
        # same as reguluar ViT block
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h 
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class LORA_Encoder(nn.Module):
    def __init__(self, config, vis, lora_cfg):
        super(LORA_Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        
        self.num_layers = config.transformer["num_layers"]
        for _ in range(self.num_layers):
            layer = LORA_Block(config, vis, lora_cfg)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class LORA_Transformer(nn.Module):
    def __init__(self, config, img_size, vis, lora_cfg):
        super(LORA_Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = LORA_Encoder(config, vis, lora_cfg)

        if 'pre_ln' in config:
            self.pre_ln = LayerNorm(config.hidden_size, eps=1e-6)
        else:
            self.pre_ln = nn.Identity()

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        embedding_output = self.pre_ln(embedding_output)

        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class LORA_VisionTransformer(nn.Module):
    def __init__(
        self, model_type,
        img_size=224, num_classes=21843, vis=False, lora_cfg=None
    ):
        super(LORA_VisionTransformer, self).__init__()
        config = CONFIGS[model_type]
        self.num_classes = num_classes
        self.classifier = config.classifier

        self.transformer = LORA_Transformer(config, img_size, vis, lora_cfg)
        self.head = Linear(config.hidden_size, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, vis=False):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if not vis:
            return logits
        return logits, attn_weights

    def load_from(self, weights):
        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)

    def adv_load_from(self, weights, feat_dim):
        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings.weight.copy_(weights.pop('patch_embed.proj.weight'))
            self.transformer.embeddings.patch_embeddings.bias.copy_(weights.pop('patch_embed.proj.bias'))
            self.transformer.embeddings.cls_token.copy_(weights.pop('cls_token'))
            self.transformer.encoder.encoder_norm.weight.copy_(weights.pop('norm.weight'))
            self.transformer.encoder.encoder_norm.bias.copy_(weights.pop('norm.bias'))
            self.transformer.embeddings.position_embeddings.copy_(weights.pop('pos_embed'))

            keys = list(weights.keys())
            weights_ = OrderedDict()
            for key in keys:
                if key.startswith('head'):
                    continue
                key_ = key.replace('blocks.', '') \
                          .replace('norm1', 'attention_norm') \
                          .replace('norm2', 'ffn_norm') \
                          .replace('mlp', 'ffn') \
                          .replace('proj', 'out')
                if 'qkv' in key:
                    value = weights.pop(key)
                    q, k, v = torch.split(value, feat_dim, dim=0)
                    weights_[key_.replace('qkv', 'query')] = q
                    weights_[key_.replace('qkv', 'key')] = k
                    weights_[key_.replace('qkv', 'value')] = v
                else:
                    weights_[key_] = weights.pop(key)
            missing_keys, unexpected_keys = self.transformer.encoder.layer.load_state_dict(weights_, strict=False)
            # assert only adapter weights are not loaded
            for missing_key in missing_keys:
                assert 'lora' in missing_key, "not strictly load weights"

    def pytorch_load_from(self, weights, feat_dim):
        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings.weight.copy_(weights.pop('conv_proj.weight'))
            self.transformer.embeddings.patch_embeddings.bias.copy_(weights.pop('conv_proj.bias'))
            self.transformer.embeddings.cls_token.copy_(weights.pop('class_token'))
            self.transformer.encoder.encoder_norm.weight.copy_(weights.pop('encoder.ln.weight'))
            self.transformer.encoder.encoder_norm.bias.copy_(weights.pop('encoder.ln.bias'))
            self.transformer.embeddings.position_embeddings.copy_(weights.pop('encoder.pos_embedding'))

            keys = list(weights.keys())
            weights_ = OrderedDict()
            for key in keys:
                if key.startswith('head'):
                    continue
                key_ = key.replace('encoder.layers.encoder_layer_', '') \
                          .replace('self_attention', 'attn') \
                          .replace('ln_1', 'attention_norm') \
                          .replace('ln_2', 'ffn_norm') \
                          .replace('mlp.linear_1', 'ffn.fc1') \
                          .replace('mlp.linear_2', 'ffn.fc2') \
                          .replace('out_proj', 'out')
                if 'in_proj_weight' in key:
                    value = weights.pop(key)
                    q, k, v = torch.split(value, feat_dim, dim=0)
                    weights_[key_.replace('in_proj_weight', 'query.weight')] = q
                    weights_[key_.replace('in_proj_weight', 'key.weight')] = k
                    weights_[key_.replace('in_proj_weight', 'value.weight')] = v
                elif 'in_proj_bias' in key:
                    value = weights.pop(key)
                    q, k, v = torch.split(value, feat_dim, dim=0)
                    weights_[key_.replace('in_proj_bias', 'query.bias')] = q
                    weights_[key_.replace('in_proj_bias', 'key.bias')] = k
                    weights_[key_.replace('in_proj_bias', 'value.bias')] = v
                else:
                    weights_[key_] = weights.pop(key)
            missing_keys, unexpected_keys = self.transformer.encoder.layer.load_state_dict(weights_, strict=False)
            # assert only adapter weights are not loaded
            for missing_key in missing_keys:
                assert 'lora' in missing_key, "not strictly load weights"


    def clip_load_from(self, weights, feat_dim):
        # remove text weights
        all_keys = list(weights.keys())
        for k in all_keys:
            if 'visual' not in k:
                weights.pop(k)

        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings.weight.copy_(weights.pop('visual.conv1.weight'))
            self.transformer.embeddings.patch_embeddings.bias.copy_(torch.zeros_like(self.transformer.embeddings.patch_embeddings.bias))
            self.transformer.embeddings.cls_token.copy_(weights.pop('visual.class_embedding'))
            self.transformer.pre_ln.weight.copy_(weights.pop('visual.ln_pre.weight'))
            self.transformer.pre_ln.bias.copy_(weights.pop('visual.ln_pre.bias'))
            self.transformer.encoder.encoder_norm.weight.copy_(weights.pop('visual.ln_post.weight'))
            self.transformer.encoder.encoder_norm.bias.copy_(weights.pop('visual.ln_post.bias'))
            self.transformer.embeddings.position_embeddings.copy_(weights.pop('visual.positional_embedding'))

            keys = list(weights.keys())
            weights_ = OrderedDict()
            for key in keys:
                if key.startswith('visual.proj'):
                    continue
                key_ = key.replace('visual.transformer.resblocks.', '') \
                          .replace('ln_1', 'attention_norm') \
                          .replace('ln_2', 'ffn_norm') \
                          .replace('mlp.c_fc', 'ffn.fc1') \
                          .replace('mlp.c_proj', 'ffn.fc2') \
                          .replace('out_proj', 'out')
                if 'in_proj_weight' in key:
                    value = weights.pop(key)
                    q, k, v = torch.split(value, feat_dim, dim=0)
                    weights_[key_.replace('in_proj_weight', 'query.weight')] = q
                    weights_[key_.replace('in_proj_weight', 'key.weight')] = k
                    weights_[key_.replace('in_proj_weight', 'value.weight')] = v
                elif 'in_proj_bias' in key:
                    value = weights.pop(key)
                    q, k, v = torch.split(value, feat_dim, dim=0)
                    weights_[key_.replace('in_proj_bias', 'query.bias')] = q
                    weights_[key_.replace('in_proj_bias', 'key.bias')] = k
                    weights_[key_.replace('in_proj_bias', 'value.bias')] = v
                else:
                    weights_[key_] = weights.pop(key)
            missing_keys, unexpected_keys = self.transformer.encoder.layer.load_state_dict(weights_, strict=False)

            for missing_key in missing_keys:
                assert 'lora' in missing_key, "not strictly load weights"