#!/usr/bin/env python3
"""
Model construction functions.
"""
import torch
import torchvision

from .vit_models import ViT, Swin
from ..utils import logging
from ..adv_func import NormalizeByChannelMeanStd
logger = logging.get_logger("RoLI")
# Supported model types
_MODEL_TYPES = {
    "vit": ViT,
    "swin": Swin,
}


def build_model(cfg):
    """
    build model here
    """
    assert (
        cfg.MODEL.TYPE in _MODEL_TYPES.keys()
    ), "Model type '{}' not supported".format(cfg.MODEL.TYPE)
    assert (
        cfg.NUM_GPUS <= torch.cuda.device_count()
    ), "Cannot use more GPU devices than available"

    # Construct the model
    train_type = cfg.MODEL.TYPE
    model = _MODEL_TYPES[train_type](cfg, vis=cfg.MODEL.VIS)

    log_model_info(model)
    if cfg.DATA.FEATURE == 'CLIP-ViT-B-16-laion2B-s34B-b88K':
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    normalize = NormalizeByChannelMeanStd(
        mean=mean, std=std)
    if cfg.MODEL.RESIZE:
        resize = torchvision.transforms.Resize((224, 224))
        model = torch.nn.Sequential(resize, normalize, model)
    else:
        model = torch.nn.Sequential(normalize, model)

    model, device = load_model_to_device(model, cfg)
    logger.info(f"Device used for model: {device}")

    return model, device


def log_model_info(model, verbose=False):
    """Logs model info"""
    if verbose:
        logger.info(f"Classification Model:\n{model}")
    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total Parameters: {0}\t Gradient Parameters: {1}".format(
        model_total_params, model_grad_params))
    logger.info("tuned percent:%.3f"%(model_grad_params/model_total_params*100))


def get_current_device():
    if torch.cuda.is_available():
        # Determine the GPU used by the current process
        cur_device = torch.cuda.current_device()
    else:
        cur_device = torch.device('cpu')
    return cur_device


def load_model_to_device(model, cfg):
    cur_device = get_current_device()
    if torch.cuda.is_available():
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)
        # Use multi-process data parallel model in the multi-gpu setting
        if cfg.NUM_GPUS > 1:
            # Make model replica operate on the current device
            model = torch.nn.parallel.DistributedDataParallel(
                module=model, device_ids=[cur_device], output_device=cur_device,
                find_unused_parameters=True,
            )
    else:
        model = model.to(cur_device)
    return model, cur_device
