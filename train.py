import os
import sys
sys.path.append('pytorch-image-models')
sys.path.append('/kaggle/input/pytorch-image-models/pytorch-image-models-master')

import warnings
import argparse
import random
import gc

import pandas as pd
from glob import glob
from tqdm import tqdm
import numpy as np
import torchvision
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import logging
import matplotlib
import matplotlib.pyplot as plt
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.nn.modules.loss import _WeightedLoss
import torchmetrics
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append('models')
sys.path.append('/kaggle/input/petfinder/models')
from models import get_model

from utils import AverageMeter
from datasets import PetDataModule
warnings.simplefilter('ignore')


class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight = None, reduction = 'mean', smoothing = 0.0, pos_weight = None):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
        self.pos_weight = pos_weight

    @staticmethod
    def _smooth(targets, n_labels, smoothing = 0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad(): targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1), self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight, pos_weight = self.pos_weight)
        if  self.reduction == 'sum': loss = loss.sum()
        elif  self.reduction == 'mean': loss = loss.mean()
        return loss


def set_seed(seed=int):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    seed_everything(seed)
    return random_state


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    assert alpha > 0, "alpha should be larger than 0"
    assert x.size(0) > 1, "Mixup cannot be applied to a single instance."
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0])
    mixed_x = lam * x + (1 - lam) * x[rand_index, :]
    target_a, target_b = y, y[rand_index]
    return mixed_x, target_a, target_b, lam

class RMSELoss(torchmetrics.Metric):

    def __init__(self):
        super().__init__()
        self.add_state("sum_squared_errors", torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_observations", torch.tensor(0), dist_reduce_fx="sum")
        

    def update(self, preds, target):
        self.sum_squared_errors += torch.sum((preds - target) ** 2)
        self.n_observations += preds.numel()

    def compute(self):
        return torch.sqrt(self.sum_squared_errors / self.n_observations)


class PetClassifier(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fold = config.fold
        self.model = get_model(config)

        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        criterion = RMSELoss()
        self.train_criterion = criterion.clone()
        self.valid_criterion = criterion.clone()
        self.valid_pred = {}

    def training_step(self, batch, batch_idx):
        imgs, features, labels = batch
        loss, preds, labels = self._get_preds_loss(batch, "train")
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.train_criterion(preds.long(), labels.long())
        return {"loss": loss, "preds": preds, "targets": labels, "grad_norm": grad_norm}

    def training_step_end(self, outs):
        self.log(f"fold{self.fold}/train_loss_step", outs["loss"])
        self.log(f"fold{self.fold}/train_criterion_step", self.train_criterion)
        self.log(f"fold{self.fold}/train_grad_norm", outs["grad_norm"])        

    def training_epoch_end(self, outs):
        # additional log mean accuracy at the end of the epoch
        self.log(f"fold{self.fold}/train_criterion_epoch", self.train_criterion.compute())

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._get_preds_loss(batch, "valid")
        self.valid_criterion(preds.long(), labels.long())
        return {"loss": loss, "preds": preds, "targets": labels}

    def validation_step_end(self, outs):
        self.log(f"fold{self.fold}/valid_loss_step", outs["loss"])
        self.log(f"fold{self.fold}/valid_criterion_step", self.valid_criterion)

    def validation_epoch_end(self, outs):
        predictions = []
        for out in outs:
            predictions.append(out["preds"].flatten())
        predictions = torch.cat(predictions).detach().cpu()
        self.oof_preds = predictions.squeeze(-1).numpy()
        self.log(f"fold{self.fold}/valid_criterion_epoch", self.valid_criterion.compute())

    def test_step(self, batch, batch_idx):
        preds = self._get_preds_loss(batch, "test")
        return preds

    def test_epoch_end(self, outs):
        predictions = []
        for out in outs:
            predictions.append(out.flatten())
        predictions = torch.cat(predictions).detach().cpu()
        self.final_preds = predictions.squeeze(-1).numpy()
        
    def _get_preds_loss(self, batch, run_type):
        if run_type == "test":
            imgs, features = batch
            labels = None
        else:
            imgs, features, labels = batch
        if torch.rand(1)[0] < 0.5 and self.config.use_mixup and run_type == "train":
            mix_images, target_a, target_b, lam = mixup(imgs, labels, alpha=0.5)
            logits = self.model(mix_images, features, labels).squeeze(1)
            loss = self.loss_fn(logits, target_a) * lam + (1 - lam) * self.loss_fn(logits, target_b)
        elif run_type == "test":
            logits = self.model(imgs, features, labels).squeeze(1)
            preds = logits.sigmoid() * 100.
            return preds
        else:
            logits = self.model(imgs, features, labels).squeeze(1)
            loss = self.loss_fn(logits, labels)           
        preds = logits.sigmoid() * 100.
        labels = labels.detach() * 100.
        return loss, preds, labels

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.shd_para.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.config.shd_para.T_0, eta_min=self.config.shd_para.eta_min)
        return [optimizer], [scheduler]


@ hydra.main(config_name="config")
def main(config):
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        id=config.default.output_folder,
    )
    wandb_logger = WandbLogger()

    for fold in range(config.n_fold):
        config.fold = fold
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename=f"fold{fold}-best-checkpoint",
            save_top_k=1,
            verbose=True,
            monitor=f"fold{fold}/valid_criterion_epoch",
            mode="min"
        )
        earystopping = EarlyStopping(monitor=f"fold{fold}/valid_criterion_epoch", mode="min")
        lr_monitor = callbacks.LearningRateMonitor()

        data = PetDataModule(config)

        model = PetClassifier(config)

        trainer = Trainer(
            callbacks=[checkpoint_callback, earystopping, lr_monitor],
            max_epochs=config.model.num_epochs,
            gpus=1,
            precision=16,
            progress_bar_refresh_rate=5,
            logger=wandb_logger,
            accumulate_grad_batches=config.accumulate_grad_batches
        )

        trainer.fit(model, data)
        trainer.test(ckpt_path='best')

        if fold == 0:
            train = pd.read_csv(config.default.train_path)
            submission = pd.read_csv(config.default.test_path)
            oof_pred_arr = np.zeros(len(train))
            ensenmble_preds = model.final_preds / config.n_fold
        else:
            ensenmble_preds += model.final_preds / config.n_fold
        oof_pred_arr[data.val_index] = model.oof_preds
        submission[config.target_col] = model.final_preds
        submission.to_csv(f'sub_{fold}.csv', index=False)

        del data, model, trainer
        gc.collect()

    train['pred'] = oof_pred_arr
    train.to_csv('oof_pred.csv', index=False)

    submission = pd.read_csv(config.default.test_path)
    submission[config.target_col] = ensenmble_preds
    submission.to_csv('sub.csv', index=False)


if __name__ == '__main__':
    random_state = set_seed(1024)
    main()
