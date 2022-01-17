import sys
sys.path.append('pytorch-image-models')
sys.path.append('/kaggle/input/pytorch-image-models/pytorch-image-models-master')

import warnings
warnings.simplefilter('ignore')
import gc

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
import hydra

sys.path.append('models')
sys.path.append('/kaggle/input/petfinder/models')
from utils import set_seed
from losses import RMSELoss, SmoothBCEwLogits
from datasets import PetDataModule
from models import SimpleCNN


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    assert alpha > 0, "alpha should be larger than 0"
    assert x.size(0) > 1, "Mixup cannot be applied to a single instance."
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0])
    mixed_x = lam * x + (1 - lam) * x[rand_index, :]
    target_a, target_b = y, y[rand_index]
    return mixed_x, target_a, target_b, lam


class PetClassifier(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fold = config.fold
        self.__build_model()
        self.__get_loss()
        criterion = RMSELoss()
        self.train_criterion = criterion.clone()
        self.valid_criterion = criterion.clone()

    def forward(self, x, features, targets):
        out = self.model(x, features, targets)
        return out

    def __build_model(self):
        self.model = SimpleCNN(self.config)
    
    def __get_loss(self):
        loss_dict = {
            "sbce": SmoothBCEwLogits,
            "bce": torch.nn.BCEWithLogitsLoss()
        }
        self.loss_fn = loss_dict[self.config.mode.loss]

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.__share_step(batch, 'train')
        self.train_criterion(preds.long(), labels.long())
        return {"loss": loss, "preds": preds, "labels": labels}

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.__share_step(batch, "valid")
        self.valid_criterion(preds.long(), labels.long())
        return {"loss": loss, "preds": preds, "targets": labels}

    def test_step(self, batch, batch_idx):
        preds = self.__share_step(batch, "test")
        return preds

    def __share_step(self, batch, mode):
        images, labels = batch
        labels = labels.float() / 100.0
        images = self.transform[mode](images)
        
        if torch.rand(1)[0] < 0.5 and mode == 'train':
            mix_images, target_a, target_b, lam = mixup(images, labels, alpha=0.5)
            logits = self.forward(mix_images).squeeze(1)
            loss = self._criterion(logits, target_a) * lam + \
                (1 - lam) * self._criterion(logits, target_b)
        else:
            logits = self.forward(images).squeeze(1)
            loss = self._criterion(logits, labels)
        
        pred = logits.sigmoid().detach().cpu() * 100.
        labels = labels.detach().cpu() * 100.
        return loss, pred, labels

    def training_step_end(self, outputs):
        self.__share_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, 'test')

    def __share_epoch_end(self, outputs, mode):
        preds = []
        labels = []
        for out in outputs:
            pred, label = out['pred']
            preds.append(pred)
            labels.append(label)

        if mode == "test":
            self.oof_preds = preds
        else:
            labels = torch.cat(labels)

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.log(f"fold{self.fold}/{mode}_loss", self.train_criterion.compute())
        self.log(f"fold{self.fold}/{mode}_grad_norm", grad_norm)

    def check_gradcam(self, dataloader, target_layer, target_category, reshape_transform=None):
        cam = GradCAMPlusPlus(
            model=self,
            target_layer=target_layer,
            use_cuda=self.cfg.trainer.gpus,
            reshape_transform=reshape_transform)
        
        org_images, labels = iter(dataloader).next()
        cam.batch_size = len(org_images)
        images = self.transform['val'](org_images)
        images = images.to(self.device)
        logits = self.forward(images).squeeze(1)
        pred = logits.sigmoid().detach().cpu().numpy() * 100
        labels = labels.cpu().numpy()
        
        grayscale_cam = cam(input_tensor=images, target_category=target_category, eigen_smooth=True)
        org_images = org_images.detach().cpu().numpy().transpose(0, 2, 3, 1) / 255.
        return org_images, grayscale_cam, pred, labels

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
        id=config.output_folder,
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
            oof_pred_arr = np.zeros(len(train))
        oof_pred_arr[data.val_index] = model.oof_preds

        del data, model, trainer
        gc.collect()

    train['pred'] = oof_pred_arr
    train.to_csv('oof_pred.csv', index=False)


if __name__ == '__main__':
    random_state = set_seed(1024)
    main()
