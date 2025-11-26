import csv
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.nn.functional as F

# ==========================
# CONFIG
# ==========================

@dataclass
class Config:
    batch_size: int = 256
    num_workers: int = 4
    epochs_linear: int = 20
    lr_linear: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

CFG = Config()

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

def get_supervised_dataloaders():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True,
        transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=CFG.batch_size,
        shuffle=True, num_workers=CFG.num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CFG.batch_size,
        shuffle=False, num_workers=CFG.num_workers
    )

    return train_loader, test_loader, train_dataset, test_dataset

# ==========================
# ENCODER + LINEAR CLASSIFIER
# ==========================

def build_backbone():
    encoder = models.resnet18(weights=None)
    encoder.fc = nn.Identity()
    return encoder

class LinearClassifier(nn.Module):
    def __init__(self, encoder, num_classes=10):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        with torch.no_grad():
            feats = self.encoder(x)
        logits = self.classifier(feats)
        return logits

def eval_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    device = CFG.device
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total * 100.0

def train_linear(encoder, train_loader, test_loader, title, log_path):
    device = CFG.device
    encoder = encoder.to(device)
    model = LinearClassifier(encoder).to(device)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=CFG.lr_linear)
    criterion = nn.CrossEntropyLoss()

    # CSV log
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "test_acc"])

    best_test_acc = 0.0

    for epoch in range(1, CFG.epochs_linear + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

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
        test_acc = eval_accuracy(model, test_loader)
        best_test_acc = max(best_test_acc, test_acc)

        print(f"[{title}] Epoch [{epoch}/{CFG.epochs_linear}] "
              f"Loss: {avg_loss:.4f}  Train acc: {train_acc:.2f}%  Test acc: {test_acc:.2f}%")

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss, train_acc, test_acc])

    return best_test_acc

# ==========================
# k-NN EVAL U EMBEDDING PROSTORU
# ==========================

def extract_embeddings(encoder, dataset, batch_size=256):
    device = CFG.device
    encoder = encoder.to(device)
    encoder.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=CFG.num_workers)
    all_feats = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            feats = encoder(x)
            feats = F.normalize(feats, dim=1)
            all_feats.append(feats.cpu())
            all_labels.append(y.clone())

    all_feats = torch.cat(all_feats, dim=0)   # [N, D]
    all_labels = torch.cat(all_labels, dim=0) # [N]
    return all_feats, all_labels

def knn_classifier(train_feats, train_labels, test_feats, test_labels, k=200):
    """
    k-NN evaluacija u embedding prostoru, cosine similarity.
    """
    train_feats = train_feats.t()  # [D, Ntrain]
    num_test = test_feats.size(0)
    correct = 0

    with torch.no_grad():
        # similarity: [Ntest, Ntrain]
        sim = torch.mm(test_feats, train_feats)
        topk = sim.topk(k=k, dim=1).indices  # [Ntest, k]
        neighbor_labels = train_labels[topk]  # [Ntest, k]
        # glasanje
        for i in range(num_test):
            vals, counts = neighbor_labels[i].unique(return_counts=True)
            pred = vals[counts.argmax()]
            if pred.item() == test_labels[i].item():
                correct += 1

    return correct / num_test * 100.0

def knn_eval(encoder, train_dataset, test_dataset, k=200, max_train=None, max_test=None):
    """
    max_train / max_test: ako želiš ubrzati, možeš uzeti podskup.
    """
    encoder = encoder.to(CFG.device)
    encoder.eval()

    if max_train is not None:
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, list(range(max_train)))
    if max_test is not None:
        from torch.utils.data import Subset
        test_dataset = Subset(test_dataset, list(range(max_test)))

    train_feats, train_labels = extract_embeddings(encoder, train_dataset)
    test_feats, test_labels = extract_embeddings(encoder, test_dataset)

    acc = knn_classifier(train_feats, train_labels, test_feats, test_labels, k=k)
    return acc

# ==========================
# MAIN: UCITAJ ENCODERE I USPOREDI
# ==========================

def load_encoder_from_ckpt(path):
    encoder = build_backbone()
    state = torch.load(path, map_location="cpu")
    encoder.load_state_dict(state)
    return encoder

def main():
    print("Using device:", CFG.device)
    train_loader, test_loader, train_dataset, test_dataset = get_supervised_dataloaders()

    encoders: Dict[str, nn.Module] = {}

    # Random encoder
    encoders["Random"] = build_backbone()

    # SimCLR encoder
    try:
        encoders["SimCLR"] = load_encoder_from_ckpt("simclr_encoder.pth")
    except FileNotFoundError:
        print("WARNING: simclr_encoder.pth not found, skipping SimCLR")

    # MoCo encoder
    try:
        encoders["MoCo"] = load_encoder_from_ckpt("moco_encoder.pth")
    except FileNotFoundError:
        print("WARNING: moco_encoder.pth not found, skipping MoCo")

    # BYOL encoder
    try:
        encoders["BYOL"] = load_encoder_from_ckpt("byol_encoder.pth")
    except FileNotFoundError:
        print("WARNING: byol_encoder.pth not found, skipping BYOL")

    results = []

    for name, enc in encoders.items():
        print(f"\n========== {name} : LINEAR EVAL ==========")
        log_path = f"linear_{name.lower()}_log.csv"
        best_test = train_linear(enc, train_loader, test_loader, title=f"{name} Linear Eval", log_path=log_path)

        print(f"\n========== {name} : k-NN EVAL ==========")
        knn_acc = knn_eval(enc, train_dataset, test_dataset, k=200, max_train=None, max_test=None)
        print(f"[{name}] k-NN accuracy: {knn_acc:.2f}%")

        results.append((name, best_test, knn_acc))

    print("\n========== SUMMARY ==========")
    print(f"{'Encoder':<10} | {'Best Linear Acc':>15} | {'k-NN Acc':>10}")
    print("-" * 42)
    for name, lin_acc, knn_acc in results:
        print(f"{name:<10} | {lin_acc:15.2f}% | {knn_acc:10.2f}%")


if __name__ == "__main__":
    main()
