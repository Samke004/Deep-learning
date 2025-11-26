import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# ==========================
# CONFIG
# ==========================

@dataclass
class Config:
    batch_size: int = 256
    num_workers: int = 4
    epochs_pretrain: int = 20      # povecaj npr. na 100 za ozbiljnije rezultate
    epochs_linear: int = 20
    lr_pretrain: float = 3e-4
    lr_linear: float = 1e-3
    temperature: float = 0.5
    projection_dim: int = 128
    small_labels_per_class: int = 100  # za semi-supervised eksperiment
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

CFG = Config()

# CIFAR-10 normalizacija
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# ==========================
# DATA AUGMENTACIJE
# ==========================

class SimCLRTransform:
    """
    Za svaku sliku vrati dvije jake augmentacije (view1, view2),
    kao u SimCLR-u.
    """
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
        xi = self.base_transform(x)
        xj = self.base_transform(x)
        return xi, xj


def get_dataloaders():
    # self-supervised train loader (bez labela, par augmentacija)
    train_dataset_simclr = datasets.CIFAR10(
        root="./data", train=True, download=True,
        transform=SimCLRTransform(image_size=32)
    )

    train_loader_simclr = DataLoader(
        train_dataset_simclr, batch_size=CFG.batch_size,
        shuffle=True, num_workers=CFG.num_workers, drop_last=True
    )

    # supervised train/test loader (standardne augmentacije)
    train_transform_supervised = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_dataset_sup = datasets.CIFAR10(
        root="./data", train=True, download=False,
        transform=train_transform_supervised
    )

    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=False,
        transform=test_transform
    )

    train_loader_sup = DataLoader(
        train_dataset_sup, batch_size=CFG.batch_size,
        shuffle=True, num_workers=CFG.num_workers
    )

    test_loader = DataLoader(
        test_dataset, batch_size=CFG.batch_size,
        shuffle=False, num_workers=CFG.num_workers
    )

    return train_loader_simclr, train_loader_sup, test_loader, train_dataset_sup


# ==========================
# MODEL: ENCODER + PROJECTION HEAD
# ==========================

class ResNetSimCLR(nn.Module):
    def __init__(self, base_model="resnet18", out_dim=128):
        super().__init__()
        if base_model == "resnet18":
            self.encoder = models.resnet18(weights=None)
            feat_dim = 512
        else:
            raise NotImplementedError

        # makni originalni classifier
        self.encoder.fc = nn.Identity()

        # projection head (MLP)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, out_dim)
        )

    def forward(self, x):
        h = self.encoder(x)         # encoder features
        z = self.projector(h)       # projected features
        return h, z


# ==========================
# NT-Xent LOSS
# ==========================

def nt_xent_loss(z1, z2, temperature=0.5):
    """
    z1, z2: [batch, dim]
    """
    batch_size = z1.size(0)

    # normalizacija
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    z = torch.cat([z1, z2], dim=0)   # [2B, dim]
    sim = torch.matmul(z, z.T) / temperature  # [2B, 2B]

    # mask self-similarity
    diag = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(diag, -9e15)

    # pozitive su (i, i+B) i (i+B, i)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z.device),
        torch.arange(0, batch_size, device=z.device)
    ], dim=0)

    loss = F.cross_entropy(sim, labels)
    return loss


# ==========================
# SELF-SUPERVISED PRETRAIN
# ==========================

def pretrain_simclr(train_loader_simclr):
    model = ResNetSimCLR(out_dim=CFG.projection_dim).to(CFG.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr_pretrain)

    model.train()
    for epoch in range(1, CFG.epochs_pretrain + 1):
        total_loss = 0.0
        for (x_i, x_j), _ in train_loader_simclr:
            x_i = x_i.to(CFG.device)
            x_j = x_j.to(CFG.device)

            _, z_i = model(x_i)
            _, z_j = model(x_j)

            loss = nt_xent_loss(z_i, z_j, temperature=CFG.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader_simclr)
        print(f"[Pretrain] Epoch [{epoch}/{CFG.epochs_pretrain}] Loss: {avg_loss:.4f}")

    # spremi encoder (bez projection heada)
    torch.save(model.encoder.state_dict(), "simclr_encoder.pth")
    print(">> Saved encoder to simclr_encoder.pth")
    return model.encoder


# ==========================
# LINEAR EVALUATION
# ==========================

class LinearClassifier(nn.Module):
    def __init__(self, encoder, num_classes=10):
        super().__init__()
        self.encoder = encoder
        self.encoder.fc = nn.Identity()  # just in case
        # zamrzni encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        with torch.no_grad():
            feats = self.encoder(x)
        logits = self.classifier(feats)
        return logits


def train_linear(encoder, train_loader, test_loader, title=""):
    model = LinearClassifier(encoder).to(CFG.device)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=CFG.lr_linear)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, CFG.epochs_linear + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x = x.to(CFG.device)
            y = y.to(CFG.device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = correct / total * 100.0
        avg_loss = total_loss / len(train_loader)
        print(f"[{title} Linear] Epoch [{epoch}/{CFG.epochs_linear}] "
              f"Loss: {avg_loss:.4f}  Train acc: {train_acc:.2f}%")

    # evaluacija na testu
    test_acc = eval_accuracy(model, test_loader)
    print(f">> {title} Linear Test Accuracy: {test_acc:.2f}%")
    return test_acc


def eval_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(CFG.device)
            y = y.to(CFG.device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total * 100.0


# ==========================
# SMALL-LABEL (SEMI-SUPERVISED) SUBSET
# ==========================

def make_small_labeled_subset(dataset, n_per_class):
    targets = dataset.targets  # lista labela
    n_classes = len(set(targets))

    indices_per_class = {c: [] for c in range(n_classes)}
    for idx, y in enumerate(targets):
        if len(indices_per_class[y]) < n_per_class:
            indices_per_class[y].append(idx)
        if all(len(v) >= n_per_class for v in indices_per_class.values()):
            break

    subset_indices = []
    for c in range(n_classes):
        subset_indices.extend(indices_per_class[c])

    random.shuffle(subset_indices)
    print(f">> Semi-supervised subset: {len(subset_indices)} samples "
          f"({n_per_class} per class)")
    return subset_indices


def main():
    print("Using device:", CFG.device)

    # 1) DATA
    train_loader_simclr, train_loader_sup, test_loader, train_dataset_sup = get_dataloaders()

    # 2) SELF-SUPERVISED PRETRAIN (SimCLR)
    encoder_pretrained = pretrain_simclr(train_loader_simclr)

    # 3) LINEAR EVAL NA SVIM LABELAMA
    print("\n========== LINEAR EVAL (sve labele) ==========")
    full_train_loader = train_loader_sup
    acc_full_simclr = train_linear(encoder_pretrained, full_train_loader, test_loader,
                                   title="SimCLR encoder")

    # 4) SEMI-SUPERVISED: MALO LABELA
    print("\n========== SEMI-SUPERVISED (malo labela) ==========")
    subset_indices = make_small_labeled_subset(
        train_dataset_sup, CFG.small_labels_per_class
    )
    small_subset = Subset(train_dataset_sup, subset_indices)
    small_loader = DataLoader(
        small_subset, batch_size=CFG.batch_size,
        shuffle=True, num_workers=CFG.num_workers
    )

    # 4a) Linear head na SimCLR encoderu s malo labela
    print("\n-- SimCLR encoder + linear head (malo labela) --")
    acc_small_simclr = train_linear(encoder_pretrained, small_loader, test_loader,
                                    title="SimCLR encoder (small labels)")

    # 4b) Random encoder + linear head (malo labela) – baseline
    print("\n-- Random encoder + linear head (malo labela) --")
    random_encoder = models.resnet18(weights=None)
    random_encoder.fc = nn.Identity()
    random_encoder = random_encoder.to(CFG.device)

    acc_small_random = train_linear(random_encoder, small_loader, test_loader,
                                    title="Random encoder (small labels)")

    print("\n========== SUMMARY ==========")
    print(f"Linear eval (full labels) - SimCLR encoder: {acc_full_simclr:.2f}%")
    print(f"Semi-supervised (small labels={CFG.small_labels_per_class}/class):")
    print(f"  SimCLR encoder: {acc_small_simclr:.2f}%")
    print(f"  Random encoder: {acc_small_random:.2f}%")


if __name__ == "__main__":
    main()
