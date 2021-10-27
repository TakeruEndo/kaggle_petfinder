import sys
sys.path.append('/kaggle/input/pytorch-image-models/pytorch-image-models-master')
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter
import math

from metric_model import ArcMarginProduct, AddMarginProduct, SphereProduct


def get_model(config):
    if config.model.mode == "mlp":
        return MLP(config)
    elif config.model.mode == "falanx":
        return FalanxModel(config)
    else:
        return Custom2DCNN(config)


class FalanxModel(nn.Module):
    """
    Reference: https://www.kaggle.com/phalanx/train-swin-t-pytorch-lightning
    """

    def __init__(self, config):
        self.backbone = timm.create_model(
            config.model.backborn, pretrained=False, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(num_features, 1)
        )

    def forward(self, x):
        f = self.backbone(x)
        out = self.fc(f)
        return out


def get_2d_backborn_model(config, hidden_size, pretrained=False):
    model_name = config.model.backborn
    model = timm.create_model(
        model_name, pretrained=pretrained, in_chans=config.input_channel)

    if 'efficientnet' in model_name:
        n_features = model.classifier.in_features
        model.classifier = nn.Identity()
        # model.global_pool = nn.Identity()
    elif 'vit' in model_name or 'swin' in model_name or 'mixer' in model_name:
        n_features = model.head.in_features
        model.head = nn.Identity()
    elif 'bitm' in model_name:
        n_features = 21843
    else:
        n_features = list(model.children())[-1].in_features
        layers = list(model.children())[:-1]
        return torch.nn.Sequential(*layers), n_features
    return model, n_features


class Custom2DCNN(nn.Module):
    def __init__(self, config):
        super(Custom2DCNN, self).__init__()
        emb_size = config.default.embedding_size
        num_classes = config.default.n_classes
        self.model, n_features = get_2d_backborn_model(
            config, emb_size, pretrained=config.pretrained)

        self.metric_name = config.model.metric
        if self.metric_name == 'add_margin':
            self.linear = nn.Linear(n_features, emb_size)
            self.metric_fc = AddMarginProduct(
                emb_size, num_classes, s=30, m=0.35)
        elif self.metric_name == 'arc_margin':
            self.linear = nn.Linear(n_features, emb_size)
            self.metric_fc = ArcMarginProduct(
                emb_size, num_classes, s=30, m=0.5, easy_margin=False)
        elif self.metric_name == 'sphere':
            self.linear = nn.Linear(n_features, emb_size)
            self.metric_fc = SphereProduct(emb_size, num_classes, m=4)
        elif self.metric_name == 'mlp':
            self.classifier = MLP(config, n_features)
        elif self.metric_name == 'ver1':
            self.fc1 = nn.Linear(n_features, 256)
            self.dropout = nn.Dropout(0.1)
            self.relu = nn.LeakyReLU()
            self.fc2 = nn.Linear(256, 64)
            self.fc3 = nn.Linear(64 + 12, 1)
        elif self.metric_name == 'ver2':
            self.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(n_features, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )

    def forward(self, x, features, label):
        embedder = self.model(x)
        if self.metric_name == 'ver1':
            out = self.relu(self.fc1(embedder))
            out = self.dropout(out)
            out = self.relu(self.fc2(out))
            out = torch.cat([out, features], dim=1)
            out = self.fc3(out)
            return out
        elif self.metric_name == 'ver2':
            out = self.fc(embedder)
            return out
        elif self.metric_name == 'mlp':
            out = self.classifier(embedder)
            return out
        else:
            embedder = self.linear(embedder)
            out = self.metric_fc(embedder, label)
        return embedder, out


class MLP(nn.Module):
    def __init__(self, config, input_size):
        super(MLP, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(input_size)
        self.dense1 = nn.Linear(input_size, input_size // 4)

        self.batch_norm2 = nn.BatchNorm1d(input_size // 4)
        self.dropout2 = nn.Dropout(0.35)
        self.dense2 = nn.Linear(input_size // 4, input_size // 8)

        self.batch_norm3 = nn.BatchNorm1d(input_size // 8)
        self.dropout3 = nn.Dropout(0.3)
        self.dense3 = nn.Linear(input_size // 8, config.default.n_classes)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = self.relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)
        return x
