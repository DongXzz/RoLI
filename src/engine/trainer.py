#!/usr/bin/env python3
"""
a trainer class
"""
import datetime
import time
import torch
import torch.nn as nn
import os
from collections import OrderedDict
import copy

from fvcore.common.config import CfgNode
from fvcore.common.checkpoint import Checkpointer

from ..engine.evaluator import Evaluator
from ..solver.lr_scheduler import make_scheduler
from ..solver.optimizer import make_optimizer
from ..solver.losses import build_loss
from ..utils import logging
from ..utils.train_utils import AverageMeter, gpu_mem_usage
from ..adv_func import adv_generator

logger = logging.get_logger("RoLI")


class Trainer():
    """
    a trainer with below logics:

    1. Build optimizer, scheduler
    2. Load checkpoints if provided
    3. Train and eval at each epoch
    """
    def __init__(
        self,
        cfg: CfgNode,
        model: nn.Module,
        evaluator: Evaluator,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.real_model = model[-1]
        self.device = device

        # solver related
        logger.info("\tSetting up the optimizer...")
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        self.cls_criterion = build_loss(self.cfg)

        self.checkpointer = Checkpointer(
            self.real_model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # load checkpoints
            checkpoint = torch.load(cfg.MODEL.WEIGHT_PATH, map_location=torch.device("cpu"))['model']
            model.load_state_dict(checkpoint, strict=cfg.MODEL.WEIGHT_STRICT)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")

    def forward_one_batch(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """

        # forward
        with torch.set_grad_enabled(is_train):
            outputs = self.model(inputs)  # (batchsize, num_cls)
            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss, outputs

    def get_input(self, data):
        if not isinstance(data["image"], torch.Tensor):
            for k, v in data.items():
                data[k] = torch.from_numpy(v)

        inputs = data["image"].float()
        labels = data["label"]
        return inputs, labels

    def train_classifier(self, train_loader, val_loader, test_loader):
        """
        Train a classifier using epoch
        """
        # save the model prompt if required before training
        self.model.eval()

        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = len(train_loader)
        best_epoch = -1
        best_metric = 0
        best_model = self.model
        log_interval = self.cfg.SOLVER.LOG_EVERY_N

        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training

        best_weights = None
        for epoch in range(total_epoch):
            # reset averagemeters to measure per-epoch results
            losses.reset()
            batch_time.reset()
            data_time.reset()

            lr = self.scheduler.get_lr()[0]
            logger.info(
                "Training {} / {} epoch, with learning rate {}".format(
                    epoch + 1, total_epoch, lr
                )
            )

            # Enable training mode
            self.model.train()

            end = time.time()

            for idx, input_data in enumerate(train_loader):
                X, targets = self.get_input(input_data)
                # move data to device
                X = X.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                if self.cfg.TRAIN_MODE == 'adv':
                    train_attack_eps = self.cfg.ADV.TRAIN_ATTACK_EPS
                    train_attack_steps = self.cfg.ADV.TRAIN_ATTACK_STEPS
                    random_start = self.cfg.ADV.TRAIN_RANDOMSTART
                    train_attack_lr = 2/255
                    if self.cfg.ADV.HALF_ADV:
                        bs = X.shape[0]
                        X_adv = adv_generator(X[:bs//2], targets[:bs//2], self.model, train_attack_eps, train_attack_steps, 
                                              train_attack_lr, random_start=random_start, attack_criterion=self.cls_criterion,
                                              cls_weights=self.cls_weights)
                        X = torch.cat([X_adv, X[bs//2:]], dim=0)
                    else:
                        X = adv_generator(X, targets, self.model, train_attack_eps, train_attack_steps, train_attack_lr, 
                                          random_start=random_start, attack_criterion=self.cls_criterion, cls_weights=self.cls_weights)

                # measure data loading time
                data_time.update(time.time() - end)

                train_loss, _ = self.forward_one_batch(X, targets, True)

                if train_loss == -1:
                    # continue
                    return None

                losses.update(train_loss.item(), X.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    eta = datetime.timedelta(seconds=int(
                        seconds_per_batch * (total_data - idx - 1) + seconds_per_batch*total_data*(total_epoch-epoch-1)))
                    logger.info(
                        "\tTraining {}/{}. train loss: {:.4f},".format(
                            idx + 1,
                            total_data,
                            train_loss
                        )
                        + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                            seconds_per_batch,
                            data_time.val,
                            str(eta),
                        )
                        + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                    )
            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)
                + "average train loss: {:.4f}".format(losses.avg))
             # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
            self.scheduler.step()

            # Enable eval mode
            self.model.eval()

            if epoch % self.cfg.VAL_FREQ == self.cfg.VAL_FREQ - 1:
                # eval at each epoch for single gpu training
                self.evaluator.update_iteration(epoch)
                self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)
                # check the patience
                t_name = "val_" + val_loader.dataset.name
                t_name += '-adv' if self.cfg.TRAIN_MODE == 'adv' and self.cfg.ADV.TRAIN_ATTACK_STEPS>0 else ''
                try:
                    curr_acc = self.evaluator.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
                except KeyError:
                    return

                # for clean
                if self.cfg.TRAIN_MODE == 'clean':
                    self.eval_classifier(test_loader, "test")

                if curr_acc > best_metric:
                    best_metric = curr_acc
                    best_epoch = epoch + 1
                    best_weights = copy.deepcopy(self.model.state_dict())
                    logger.info(
                        f'Best epoch {best_epoch}: best metric: {best_metric:.3f}')
                    patience = 0
                else:
                    patience += 1
                if patience >= self.cfg.SOLVER.PATIENCE:
                    logger.info("No improvement. Breaking out of loop.")
                    break

        # save the last checkpoints
        if self.cfg.MODEL.SAVE_CHECKPOINT:
            Checkpointer(
                self.model,
                save_dir=self.cfg.OUTPUT_DIR,
                save_to_disk=True
            ).save("last_model")

        if best_weights:
            logger.info(
                "Best Epoch {} / {}: ".format(best_epoch, total_epoch))
            # test the best checkpoints
            if test_loader is not None:
                self.model.load_state_dict(best_weights)
                self.eval_classifier(test_loader, "test")
            
            # save the best checkpoints
            if self.cfg.MODEL.SAVE_CHECKPOINT:
                Checkpointer(
                    self.model,
                    save_dir=self.cfg.OUTPUT_DIR,
                    save_to_disk=True
                ).save("best_model")  
        else:
            # test the last checkpoints
            if test_loader is not None:
                self.eval_classifier(test_loader, "test")

    # @torch.no_grad()
    def eval_classifier(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        losses_adv = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []
        if self.cfg.TRAIN_MODE == 'adv':
            total_logits_adv = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            X, targets = self.get_input(input_data)
            X = X.to(self.device, non_blocking=True)    # (batchsize, 2048)
            targets = targets.to(self.device, non_blocking=True)  # (batchsize, )
            # measure data loading time
            data_time.update(time.time() - end)

            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if self.cfg.TRAIN_MODE == 'adv':
                eval_attack_eps = self.cfg.ADV.VAL_ATTACK_EPS
                eval_attack_steps = self.cfg.ADV.VAL_ATTACK_STEPS
                eval_attack_lr = 2/255
                random_start = self.cfg.ADV.VAL_RANDOMSTART
                X = adv_generator(X, targets, self.model, eval_attack_eps, eval_attack_steps, eval_attack_lr, 
                                    random_start=random_start, attack_criterion=self.cls_criterion, cls_weights=self.cls_weights)
                loss_adv, outputs_adv = self.forward_one_batch(X, targets, False)
                losses_adv.update(loss_adv, X.shape[0])
            
            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTest {}/{}. loss: {:.3f}, loss_adv: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        losses_adv.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.cpu().numpy()))
            total_logits.append(outputs)
            if self.cfg.TRAIN_MODE == 'adv':
                total_logits_adv.append(outputs_adv)
        logger.info(
            f"Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}, ".format(losses.avg)
            + "average loss_adv: {:.4f}".format(losses_adv.avg))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )
        if self.cfg.TRAIN_MODE == 'adv':
            joint_logits_adv = torch.cat(total_logits_adv, dim=0).cpu().numpy()
            self.evaluator.classify(
                joint_logits_adv, total_targets,
                test_name+'-adv', self.cfg.DATA.MULTILABEL,
            )
