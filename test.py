import os
import sys
from glob import glob
import warnings
import gc
import pandas as pd
import numpy as np

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import hydra

sys.path.append('models')
sys.path.append('/kaggle/input/petfinder/models')
from models import get_model

from train import PetClassifier
from utils import AverageMeter
from datasets import PetDataModule
warnings.simplefilter('ignore')


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
        submission[['Id', 'Pawpularity']].to_csv('/kaggle/working/submission.csv', index=False)
    else:
        submission[['Id', 'Pawpularity']].to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
