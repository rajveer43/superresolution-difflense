"""ResNet-18 tagger (qg: quark vs gluon) / regressor (calo: incident energy).

torchvision ResNet-18 ([2, 2, 2, 2] basic blocks, widths 64-512), adapted to calorimeter images:
* the 7x7/2 stem and the max-pool are replaced by a 3x3/1 stem: single-cell deposits survive and the
  same network runs on 32x32 LR, 128x128 HR and 16x9 shower grids (adaptive average pooling at the end);
* input channels = detector planes (3 jet channels, or 45 shower layers);
* the per-plane log total energy joins the pooled features before the head.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18Tagger(nn.Module):
    def __init__(self, in_ch, n_global, out_dim=1, dropout=0.2):
        super().__init__()
        net = resnet18(weights=None)
        net.conv1 = nn.Conv2d(in_ch, 64, 3, 1, 1, bias=False)
        net.maxpool = nn.Identity()
        net.fc = nn.Identity()
        nn.init.kaiming_normal_(net.conv1.weight, mode="fan_out", nonlinearity="relu")
        for m in net.modules():  # zero-init the last BN of each block: every block starts as the identity
            if hasattr(m, "bn2"):
                nn.init.zeros_(m.bn2.weight)
        self.body = net
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(512 + n_global, 256), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout), nn.Linear(256, out_dim))

    def forward(self, x, g):
        return self.head(torch.cat([self.body(x), g], 1)).squeeze(-1)
