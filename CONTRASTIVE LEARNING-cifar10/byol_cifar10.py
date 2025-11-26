import math
import random
import copy
import csv
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ==========================
# CONFIG
# ==========================

@dataclass
class Config:
    batch_size: int = 256
    num_workers: int = 4
    epochs_pretrain: int = 50
    lr_pretrain: float = 3e-4
    weight_decay: float = 1e-6
    byol_dim: int = 256
    m_ema: float = 0.996
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_path: str = "byol_pretrain_log.csv"

CFG = Config()

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

class BYOLTransform:
    def __init__(self, image_size=32):
        self.base_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

    def __call__(self, x):
        v1 = self.base_transform(x)
        v2 = self.base_transform(x)
        return v1, v2


def get_byol_dataloader():
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True,
        transform=BYOLTransform(image_size=32)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        drop_last=True
    )
    return train_loader

# ==========================
# BYOL MODULE
# ==========================

def get_backbone():
    backbone = models.resnet18(weights=None)
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, feat_dim


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class BYOL(nn.Module):
    def __init__(self, base_encoder=models.resnet18, dim=256, m_ema=0.996):
        super().__init__()
        self.m_ema = m_ema

        # online network
        self.encoder_online, feat_dim = get_backbone()
        self.projector_online = MLP(feat_dim, feat_dim, dim)
        self.predictor = MLP(dim, feat_dim, dim)

        # target network
        self.encoder_target, _ = get_backbone()
        self.projector_target = MLP(feat_dim, feat_dim, dim)

        # init target = online
        self._update_target_network(1.0)

        # target params ne uce se direktno
        for p in self.encoder_target.parameters():
            p.requires_grad = False
        for p in self.projector_target.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _update_target_network(self, m=None):
        if m is None:
            m = self.m_ema
        for param_o, param_t in zip(self.encoder_online.parameters(), self.encoder_target.parameters()):
            param_t.data = param_t.data * m + param_o.data * (1.0 - m)
        for param_o, param_t in zip(self.projector_online.parameters(), self.projector_target.parameters()):
            param_t.data = param_t.data * m + param_o.data * (1.0 - m)

    def forward(self, v1, v2):
        """
        v1, v2: [B, C, H, W]
        """
        # online branch
        y1 = self.encoder_online(v1)
        y1 = self.projector_online(y1)
        p1 = self.predictor(y1)

        y2 = self.encoder_online(v2)
        y2 = self.projector_online(y2)
        p2 = self.predictor(y2)

        with torch.no_grad():
            self._update_target_network()

            z1 = self.encoder_target(v1)
            z1 = self.projector_target(z1)
            z2 = self.encoder_target(v2)
            z2 = self.projector_target(z2)

        loss = byol_loss_fn(p1, z2) + byol_loss_fn(p2, z1)
        return loss.mean()

    def get_encoder(self):
        enc = copy.deepcopy(self.encoder_online)
        enc.fc = nn.Identity()
        return enc


def byol_loss_fn(p, z):
    """
    L2 loss između normaliziranih predikcija i target embeddinga.
    """
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return 2 - 2 * (p * z).sum(dim=1)


# ==========================
# PRETRAIN + LOG
# ==========================

def pretrain_byol():
    train_loader = get_byol_dataloader()
    model = BYOL(dim=CFG.byol_dim, m_ema=CFG.m_ema).to(CFG.device)

    optimizer = torch.optim.Adam(
        list(model.encoder_online.parameters()) +
        list(model.projector_online.parameters()) +
        list(model.predictor.parameters()),
        lr=CFG.lr_pretrain,
        weight_decay=CFG.weight_decay
    )

    # CSV log
    with open(CFG.log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])

    model.train()
    for epoch in range(1, CFG.epochs_pretrain + 1):
        total_loss = 0.0
        for (v1, v2), _ in train_loader:
            v1 = v1.to(CFG.device)
            v2 = v2.to(CFG.device)

            loss = model(v1, v2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[BYOL Pretrain] Epoch [{epoch}/{CFG.epochs_pretrain}] Loss: {avg_loss:.4f}")

        with open(CFG.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss])

    encoder = model.get_encoder()
    torch.save(encoder.state_dict(), "byol_encoder.pth")
    print(">> Saved BYOL encoder to byol_encoder.pth")


def main():
    print("Using device:", CFG.device)
    pretrain_byol()


if __name__ == "__main__":
    main()
