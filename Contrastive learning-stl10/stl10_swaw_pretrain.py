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
    proj_dim: int = 256       # dimenzija projekcije
    num_prototypes: int = 300 # broj prototipova (klastera)
    temperature: float = 0.1  # temperatura za softmax
    sinkhorn_iters: int = 3
    sinkhorn_epsilon: float = 0.05
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_path: str = "stl10_swav_pretrain_log.csv"

CFG = Config()

STL10_MEAN = (0.4467, 0.4398, 0.4066)
STL10_STD = (0.2603, 0.2566, 0.2713)

# ==========================
# TRANSFORM & DATALOADER
# ==========================

class SwAVTransform:
    """
    Vrlo pojednostavljena verzija: dvije augmentacije (dvije "view"-a).
    U originalnom SWaV-u se koristi multi-crop, ali ovdje radi jasnoće
    ostajemo na 2 view-a (slično kao BYOL).
    """
    def __init__(self, image_size=96):
        self.base_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(STL10_MEAN, STL10_STD),
        ])

    def __call__(self, x):
        v1 = self.base_transform(x)
        v2 = self.base_transform(x)
        return v1, v2


def get_swav_dataloader():
    dataset = datasets.STL10(
        root="./data_stl10",
        split="unlabeled",
        download=True,
        transform=SwAVTransform(image_size=96),
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
# BACKBONE & MLP
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
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ==========================
# SINKHORN-KNOPP (za SWaV)
# ==========================

@torch.no_grad()
def sinkhorn_knopp(logits, epsilon=0.05, n_iters=3):
    """
    Pojednostavljena verzija Sinkhorn-Knopp normalizacije
    iz SWaV paper-a.

    logits: [B, K] (B = batch size, K = broj prototipova)
    vraća Q: [B, K] - "meki" assignmenti koji se koriste kao targeti.
    """
    Q = torch.exp(logits / epsilon).T  # [K, B]
    B = Q.shape[1]
    K = Q.shape[0]

    Q /= torch.sum(Q)
    for _ in range(n_iters):
        # normalizacija po redovima
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K

        # normalizacija po stupcima
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # svaki stupac treba sumirati na 1
    return Q.T  # [B, K]


# ==========================
# SWaV MODULE
# ==========================

class SwAV(nn.Module):
    """
    Pojednostavljena SWaV implementacija:
      - encoder (ResNet18)
      - projector MLP
      - prototipovi (linearni sloj)
      - gubitak: swapped prediction između 2 view-a
    """

    def __init__(self, dim=256, num_prototypes=300, temperature=0.1,
                 sinkhorn_iters=3, sinkhorn_epsilon=0.05):
        super().__init__()

        self.temperature = temperature
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_epsilon = sinkhorn_epsilon

        # encoder + projector
        self.encoder, feat_dim = get_backbone()
        self.projector = MLP(feat_dim, feat_dim, dim)

        # prototipovi (klasteri)
        self.prototypes = nn.Linear(dim, num_prototypes, bias=False)

    def forward(self, v1, v2):
        """
        v1, v2: [B, C, H, W] (dva augmentirana pogleda iste slike)
        """
        # 1) Izračun embeddinga i projekcija
        z1 = self.encoder(v1)            # [B, feat_dim]
        z1 = self.projector(z1)          # [B, dim]
        z1 = F.normalize(z1, dim=1)      # normalizacija

        z2 = self.encoder(v2)
        z2 = self.projector(z2)
        z2 = F.normalize(z2, dim=1)

        # 2) Logiti prema prototipovima
        logits1 = self.prototypes(z1)    # [B, K]
        logits2 = self.prototypes(z2)    # [B, K]

        # 3) Sinkhorn-Knopp assignmenti (targeti q1, q2)
        with torch.no_grad():
            q1 = sinkhorn_knopp(
                logits1.detach(),
                epsilon=self.sinkhorn_epsilon,
                n_iters=self.sinkhorn_iters
            )  # [B, K]
            q2 = sinkhorn_knopp(
                logits2.detach(),
                epsilon=self.sinkhorn_epsilon,
                n_iters=self.sinkhorn_iters
            )

        # 4) Predikcije p1, p2 (softmax s temperaturom)
        p1 = F.log_softmax(logits1 / self.temperature, dim=1)  # [B, K]
        p2 = F.log_softmax(logits2 / self.temperature, dim=1)

        # 5) Swapped loss:
        #    q1 služi kao target za p2, q2 za p1
        loss_12 = - torch.mean(torch.sum(q1 * p2, dim=1))
        loss_21 = - torch.mean(torch.sum(q2 * p1, dim=1))

        loss = 0.5 * (loss_12 + loss_21)
        return loss

    def get_encoder(self):
        enc = copy.deepcopy(self.encoder)
        enc.fc = nn.Identity()
        return enc


# ==========================
# PRETRAIN
# ==========================

def pretrain_swav():
    loader = get_swav_dataloader()
    model = SwAV(
        dim=CFG.proj_dim,
        num_prototypes=CFG.num_prototypes,
        temperature=CFG.temperature,
        sinkhorn_iters=CFG.sinkhorn_iters,
        sinkhorn_epsilon=CFG.sinkhorn_epsilon,
    ).to(CFG.device)

    optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) +
        list(model.projector.parameters()) +
        list(model.prototypes.parameters()),
        lr=CFG.lr_pretrain,
        weight_decay=CFG.weight_decay,
    )

    with open(CFG.log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])

    model.train()
    for epoch in range(1, CFG.epochs_pretrain + 1):
        total_loss = 0.0
        for (v1, v2), _ in loader:
            v1 = v1.to(CFG.device)
            v2 = v2.to(CFG.device)

            loss = model(v1, v2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[STL10 SWaV] Epoch [{epoch}/{CFG.epochs_pretrain}] Loss: {avg_loss:.4f}")

        with open(CFG.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss])

    encoder = model.get_encoder()
    torch.save(encoder.state_dict(), "stl10_swav_encoder.pth")
    print(">> Saved STL10 SWaV encoder to stl10_swav_encoder.pth")


def main():
    print("Using device:", CFG.device)
    pretrain_swav()


if __name__ == "__main__":
    main()
