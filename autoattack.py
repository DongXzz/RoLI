import torch
from tqdm import tqdm
from timm.utils import AverageMeter, accuracy
from easyrobust.attacks import AutoAttack

import src.utils.logging as logging
from train import setup, get_loaders
from src.models.build_model import build_model
from launch import default_argument_parser, logging_train_setup


def evaluate_autoattack(model, test_loader, logger, attack_type='Linf', epsilon=4/255, attacks_to_run=[]):

    device = next(model.parameters()).device
    assert attack_type in ['Linf', 'L2'], '{} is not supported!'.format(attack_type)
    if attacks_to_run:
        adversary = AutoAttack(model, norm=attack_type, eps=epsilon, attacks_to_run=attacks_to_run, version='custom')
    else:
        adversary = AutoAttack(model, norm=attack_type, eps=epsilon, version='standard')
            
    top1_m_clean = AverageMeter()
    top1_m = AverageMeter()
    model.eval()
    for data in tqdm(test_loader):
        input = data["image"].float()
        target = data["label"]    
        input = input.to(device)
        target = target.to(device)
        with torch.no_grad():
            labels = model(input.detach())

        acc1_clean, _ = accuracy(labels, target, topk=(1, 5))
        top1_m_clean.update(acc1_clean.item(), labels.size(0))

        x_adv = adversary.run_standard_evaluation(input, target, bs=input.shape[0])
        with torch.no_grad():
            labels_adv = model(x_adv.detach())

        acc1, _ = accuracy(labels_adv, target, topk=(1, 5))
        top1_m.update(acc1.item(), labels_adv.size(0))
        logger.info(f"Top1 Accuracy on the AutoAttack: {top1_m.avg:.1f}%")

    logger.info(f"Top1 Accuracy Clean: {top1_m_clean.avg:.1f}%")
    logger.info(f"Top1 Accuracy on the AutoAttack: {top1_m.avg:.1f}%")


def main(args):
    cfg = setup(args)

    # setup training env including loggers
    logging_train_setup(args, cfg)
    logger = logging.get_logger("RoLI")
    train_loader, val_loader, test_loader = get_loaders(cfg, logger)
    
    logger.info("Constructing models...")
    model, cur_device = build_model(cfg)
    
    checkpoint = torch.load(cfg.MODEL.WEIGHT_PATH, map_location=torch.device("cpu"))['model']
    model.load_state_dict(checkpoint)

    evaluate_autoattack(model, test_loader, logger, epsilon=cfg.ADV.VAL_ATTACK_EPS)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)