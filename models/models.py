import sys
sys.path.append('/kaggle/input/pytorch-image-models/pytorch-image-models-master')

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import timm

from metric_model import ArcMarginProduct, AddMarginProduct, SphereProduct


class SimpleCNN(nn.Module):
    def __init__(self, config):
        super(SimpleCNN, self).__init__()
        self.head_type = config.model.head

        self.backbone = timm.create_model(
            config.model.backbone, pretrained=config.model.pretrained, in_chans=config.input_channel)

        num_features = self.backbone.num_features
        embedding_size = config.default.embedding_size
        num_classes = config.default.n_classes
        self.metric_loss = False
        if self.model.head_2 == "None":
            embedding_size = num_classes
        elif self.model.head_2 in ["add_margin", "arc_margin", "sphere"]:
            self.metric_loss = True

        head_1_dict = {
            "linear": nn.Linear(num_features, embedding_size),
            "MLP": MLP(num_features, embedding_size),
            "MLP_with_BN": MLP(num_features, embedding_size, True)
        }

        head_2_dict = {
            "None": nn.Identity(),
            "linear": nn.Linear(embedding_size, num_classes),
            "linear_head_1_mix": nn.Linear(embedding_size + num_features, num_classes),
            "MLP": MLP(embedding_size, num_classes),
            "MLP_with_BN": MLP(embedding_size, num_classes, True),
            "add_margin": AddMarginProduct(embedding_size, num_classes, s=30, m=0.35),
            "arc_margin": ArcMarginProduct(embedding_size, num_classes, s=30, m=0.5, easy_margin=False),
            "sphere": SphereProduct(embedding_size, num_classes, m=4),
        }

        self.head_1 = head_1_dict[config.model.head_1]
        self.head_2 = head_2_dict[config.model.head_2]

    def forward(self, x, features, label):
        x = self.backbone(x)
        embeddings = self.head_1(x)
        if self.metric_loss:
            outputs = self.head_2(embeddings, label)
        else:
            if self.config.model.head_2 == "linear_head_1_mix":
                x = torch.cat([x, embeddings], dim=1)
            outputs = self.head_2(x)
        return embeddings, outputs


class MLP(nn.Module):
    def __init__(self, input_size, output_size, with_bn: bool = False, geru: bool = False):
        super(MLP, self).__init__()
        self.with_bn = with_bn
        self.fc1 = nn.Linear(input_size, input_size // 2)
        self.dropout1 = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm1d(input_size // 2)
        self.fc2 = nn.Linear(input_size // 2, input_size // 4)
        self.dropout2 = nn.Dropout(0.5)
        self.batch_norm2 = nn.BatchNorm1d(input_size // 4)
        self.fc3 = nn.Linear(input_size // 4, output_size)
        self.relu = nn.ReLU()
        if geru:
            self.relu = nn.GELU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        if self.with_bn:
            x = self.batch_norm1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        if self.with_bn:
            x = self.batch_norm2(x)
        x = self.fc3(x)
        return x
