import os
import sys
sys.path.append('pytorch-image-models')
sys.path.append('/kaggle/input/timm-pytorch-image-models/pytorch-image-models-master')


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
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torchmetrics
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append('models')
sys.path.append('/kaggle/input/petfinder/models')
from models import get_model

from utils import AverageMeter
from datasets import PetDataModule
warnings.simplefilter('ignore')


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
        logits = self.model(imgs, features, labels).squeeze(1)
        loss = self.loss_fn(logits, labels)
        preds = logits.sigmoid() * 100.
        labels = labels.detach() * 100.
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.train_criterion(preds.long(), labels.long())
        return {"loss": loss, "preds": preds, "targets": labels, "grad_norm": grad_norm}

    def training_step_end(self, outs):
        self.log("train/loss_step", outs["loss"])
        self.log("train/criterion_step", self.train_criterion)
        self.log("train/grad_norm", outs["grad_norm"])        

    def training_epoch_end(self, outs):
        # additional log mean accuracy at the end of the epoch
        self.log("train/criterion_epoch", self.train_criterion.compute())

    def validation_step(self, batch, batch_idx):
        imgs, features, labels = batch
        logits = self.model(imgs, features, labels).squeeze(1)
        loss = self.loss_fn(logits, labels)
        preds = logits.sigmoid() * 100.
        labels = labels.detach() * 100.
        self.valid_criterion(preds.long(), labels.long())
        return {"loss": loss, "preds": preds, "targets": labels}

    def validation_step_end(self, outs):
        self.log("valid/loss_step", outs["loss"])
        self.log("valid/criterion_step", self.valid_criterion)

    def validation_epoch_end(self, outs):
        if self.current_epoch == self.config.model.num_epochs - 1:
            predictions = []
            for out in outs:
                predictions.append(out.flatten().sigmoid())
            predictions = torch.cat(predictions).detach().cpu()
            self.oof_preds = predictions.squeeze(-1).numpy()
        self.log("valid/criterion_epoch", self.valid_criterion.compute())

    def test_step(self, batch, batch_idx):
        imgs, features = batch
        logits = self.model(imgs, features, None).squeeze(1)
        preds = logits.sigmoid() * 100.
        return preds

    def test_epoch_end(self, outs):
        predictions = []
        for out in outs:
            predictions.append(out.flatten())
        predictions = torch.cat(predictions).detach().cpu()
        self.final_preds = predictions.squeeze(-1).numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.shd_para.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.config.shd_para.T_0, eta_min=self.config.shd_para.eta_min)
        return [optimizer], [scheduler]


@hydra.main(config_name="config")
def main(config):

    ckp_paths = glob(os.path.join(config.default.ckp_dir, "*"))
    for fold, ckp_path in enumerate(ckp_paths):
        config.fold = fold

        data = PetDataModule(config)
        data.prepare_data()
        data.setup(stage="test")

        trainer = Trainer(gpus=1,)

        model = PetClassifier.load_from_checkpoint(ckp_path, config=config)

        trainer.test(model=model, datamodule=data)

        if fold == 0:
            submission = pd.read_csv(config.default.test_path)
            ensenmble_preds = model.final_preds / config.n_fold
        else:
            ensenmble_preds += model.final_preds / config.n_fold
        submission[config.target_col] = model.final_preds
        submission.to_csv(f'submission_{fold}.csv', index=False)

        del data, model, trainer
        gc.collect()

    submission = pd.read_csv(config.default.test_path)
    submission[config.target_col] = ensenmble_preds
    if config.in_kaggle:
        submission[['Id', 'Pawpularity']].to_csv(
            '/kaggle/working/submission.csv', index=False)
    else:
        submission[['Id', 'Pawpularity']].to_csv('submission.csv', index=False)        


if __name__ == '__main__':
    main()
