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
    lr_pretrain: float = 0.03
    momentum: float = 0.9
    weight_decay: float = 1e-4
    moco_dim: int = 128
    moco_k: int = 4096
    moco_m: float = 0.999
    temperature: float = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_path: str = "stl10_moco_pretrain_log.csv"

CFG = Config()

STL10_MEAN = (0.4467, 0.4398, 0.4066)
STL10_STD = (0.2603, 0.2566, 0.2713)

# ==========================
# AUGMENTACIJE
# ==========================

class SimCLRTransform:
    def __init__(self, image_size=96):
        self.base_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(STL10_MEAN, STL10_STD),
        ])

    def __call__(self, x):
        xi = self.base_transform(x)
        xj = self.base_transform(x)
        return xi, xj


def get_moco_dataloader():
    dataset = datasets.STL10(
        root="./data_stl10",
        split="unlabeled",
        download=True,
        transform=SimCLRTransform(image_size=96),
    )
    loader = DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        drop_last=True,
    )
    return loader

# ==========================
# MOCO MODULE
# ==========================

class MoCo(nn.Module):
    def __init__(self, base_encoder=models.resnet18, dim=128, K=4096, m=0.999, T=0.2):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = base_encoder(weights=None)
        dim_mlp = self.encoder_q.fc.in_features
        self.encoder_q.fc = nn.Identity()
        self.projector_q = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(dim_mlp, dim),
        )

        self.encoder_k = base_encoder(weights=None)
        self.encoder_k.fc = nn.Identity()
        self.projector_k = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(dim_mlp, dim),
        )

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        K = self.K

        ptr = int(self.queue_ptr)
        if ptr + batch_size <= K:
            self.queue[:, ptr:ptr+batch_size] = keys.T
            ptr = (ptr + batch_size) % K
        else:
            first = K - ptr
            self.queue[:, ptr:] = keys[:first].T
            rest = batch_size - first
            if rest > 0:
                self.queue[:, :rest] = keys[first:].T
            ptr = rest

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        q = self.encoder_q(im_q)
        q = self.projector_q(q)
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = self.projector_k(k)
            k = F.normalize(k, dim=1)

        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)   # [B, 1]
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])  # [B, K]

        logits = torch.cat([l_pos, l_neg], dim=1)  # [B, 1+K]
        logits /= self.T

        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        self._dequeue_and_enqueue(k)

        loss = F.cross_entropy(logits, labels)
        return loss

    def get_encoder(self):
        enc = copy.deepcopy(self.encoder_q)
        enc.fc = nn.Identity()
        return enc

# ==========================
# PRETRAIN
# ==========================

def pretrain_moco():
    loader = get_moco_dataloader()
    model = MoCo(dim=CFG.moco_dim, K=CFG.moco_k, m=CFG.moco_m, T=CFG.temperature).to(CFG.device)

    optimizer = torch.optim.SGD(
        list(model.encoder_q.parameters()) + list(model.projector_q.parameters()),
        lr=CFG.lr_pretrain,
        momentum=CFG.momentum,
        weight_decay=CFG.weight_decay,
    )

    with open(CFG.log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])

    model.train()
    for epoch in range(1, CFG.epochs_pretrain + 1):
        total_loss = 0.0
        for (im_q, im_k), _ in loader:
            im_q = im_q.to(CFG.device)
            im_k = im_k.to(CFG.device)

            loss = model(im_q, im_k)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[STL10 MoCo] Epoch [{epoch}/{CFG.epochs_pretrain}] Loss: {avg_loss:.4f}")

        with open(CFG.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss])

    encoder = model.get_encoder()
    torch.save(encoder.state_dict(), "stl10_moco_encoder.pth")
    print(">> Saved STL10 MoCo encoder to stl10_moco_encoder.pth")

def main():
    print("Using device:", CFG.device)
    pretrain_moco()

if __name__ == "__main__":
    main()
