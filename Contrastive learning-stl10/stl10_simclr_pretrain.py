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
    epochs_pretrain: int = 50      # povecaj za bolje rezultate
    lr_pretrain: float = 3e-4
    temperature: float = 0.5
    projection_dim: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_path: str = "stl10_simclr_pretrain_log.csv"

CFG = Config()

STL10_MEAN = (0.4467, 0.4398, 0.4066)
STL10_STD = (0.2603, 0.2566, 0.2713)

# ==========================
# DATA AUGMENTACIJE
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


def get_simclr_dataloader():
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
# MODEL
# ==========================

class ResNetSimCLR(nn.Module):
    def __init__(self, base_model="resnet18", out_dim=128):
        super().__init__()
        if base_model == "resnet18":
            self.encoder = models.resnet18(weights=None)
            feat_dim = 512
        else:
            raise NotImplementedError

        self.encoder.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, out_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z

# ==========================
# NT-Xent LOSS
# ==========================

def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = torch.matmul(z, z.T) / temperature  # [2B, 2B]

    diag = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(diag, -9e15)

    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z.device),
        torch.arange(0, batch_size, device=z.device),
    ], dim=0)

    loss = F.cross_entropy(sim, labels)
    return loss

# ==========================
# PRETRAIN
# ==========================

def pretrain_simclr():
    loader = get_simclr_dataloader()
    model = ResNetSimCLR(out_dim=CFG.projection_dim).to(CFG.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr_pretrain)

    with open(CFG.log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])

    model.train()
    for epoch in range(1, CFG.epochs_pretrain + 1):
        total_loss = 0.0
        for (x_i, x_j), _ in loader:
            x_i = x_i.to(CFG.device)
            x_j = x_j.to(CFG.device)

            _, z_i = model(x_i)
            _, z_j = model(x_j)

            loss = nt_xent_loss(z_i, z_j, temperature=CFG.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[STL10 SimCLR] Epoch [{epoch}/{CFG.epochs_pretrain}] Loss: {avg_loss:.4f}")

        with open(CFG.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss])

    torch.save(model.encoder.state_dict(), "stl10_simclr_encoder.pth")
    print(">> Saved STL10 SimCLR encoder to stl10_simclr_encoder.pth")

def main():
    print("Using device:", CFG.device)
    pretrain_simclr()

if __name__ == "__main__":
    main()
