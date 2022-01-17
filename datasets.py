from PIL import Image
import pandas as pd
import numpy as np

import cv2

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, random_split, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import pytorch_lightning as pl

from split_cv import get_cv


class PetDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

    def prepare_data(self):

        train = pd.read_csv(self.config.data.train_path)
        self.train_size = len(train)
        test = pd.read_csv(self.config.data.test_path)

        def get_train_file_path(image_id):
            return "{}/{}.jpg".format(self.config.data.train_dir, image_id)

        def get_test_file_path(image_id):
            return "{}/{}.jpg".format(self.config.data.test_dir, image_id)

        train['file_path'] = train['Id'].apply(get_train_file_path)
        test['file_path'] = test['Id'].apply(get_test_file_path)

        train = self.__data_split(train, self.config)

        self.train_df = train[train['fold'] != self.config.fold]
        self.val_df = train[train['fold'] == self.config.fold]
        self.test_df = test
        self.val_index = self.val_df.index
        # reset_index
        self.train_df = self.train_df.reset_index(drop=True)
        self.val_df = self.val_df.reset_index(drop=True)

    def __data_split(self, df, config):
        df['fold'] = 0
        fold_type = config.fold_type
        if fold_type == 'kfold':
            kf = KFold(n_splits=config.n_fold, shuffle=True, random_state=config.seed)
            for i, (tr_idx, val_index) in enumerate(kf.split(df)):
                df.loc[val_index, 'fold'] = int(i)
        elif fold_type == 'skfold':
            # num_bins = int(np.floor(1 + np.log2(len(df))))
            # bins = pd.cut(df[cfg.target_col], bins=num_bins, labels=False)
            Fold = StratifiedKFold(n_splits=config.n_fold, shuffle=True, random_state=config.seed)
            for n, (train_index, val_index) in enumerate(Fold.split(df, df[config.data.target_col])):
                df.loc[val_index, 'fold'] = int(n)
        return df

    def setup(self, stage=None):

        if stage == 'fit' or stage is None:
            print('Data size:', self.train_df.shape, self.val_df.shape)
            self.train_dataset = PetDataset(
                self.config, self.train_df, self.train_transforms(), 'train')
            self.val_dataset = PetDataset(
                self.config, self.val_df, self.valid_transforms(), 'val')
        if stage == 'test' or stage is None:
            self.test_dataset = PetDataset(
                self.config, self.test_df, self.valid_transforms(), 'test')

    def train_transforms(self):
        image_size = self.config.default.img_size
        train_transform = A.Compose(
            [
                A.RandomResizedCrop(image_size, image_size, scale=(0.85, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                # A.OneOf([
                #     A.GaussNoise(p=1.0),
                #     A.IAAAdditiveGaussianNoise(p=1.0),
                #     A.MotionBlur(p=1.0)
                # ], p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                ToTensorV2()
            ]
        )
        return train_transform

    def valid_transforms(self):
        image_size = self.config.default.img_size
        valid_transform = A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                ToTensorV2(),
            ]
        )
        return valid_transform

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.model.train_batch_size,
            shuffle=True,
            num_workers=self.config.model.num_workers,
            pin_memory=False,
            drop_last=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.model.val_batch_size,
            shuffle=False,
            num_workers=self.config.model.num_workers,
            pin_memory=False,
            drop_last=False)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.model.val_batch_size,
            shuffle=False,
            num_workers=self.config.model.num_workers,
            pin_memory=False,
            drop_last=False)


class PetDataset(Dataset):

    def __init__(self, config, df, transforms, type_):
        self.config = config
        self.type = type_
        self.df = df
        dense_features = [
            'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
            'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
        ]
        self.feature = df[dense_features]
        self.transforms = transforms
        self.file_names = df['file_path'].values
        if self.type != 'test':
            self.labels = df[config.target_col].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        path = self.file_names[index]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image'].float()
        features = self.feature.loc[index].values
        features = torch.tensor(features, dtype=torch.float)

        if self.type == 'test':
            return image, features
        label = torch.tensor(self.labels[index]).float() / 100
        return image, features, label
