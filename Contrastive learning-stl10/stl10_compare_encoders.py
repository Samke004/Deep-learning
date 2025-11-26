from dataclasses import dataclass
from typing import Dict

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
    epochs_linear: int = 20
    lr_linear: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

CFG = Config()

STL10_MEAN = (0.4467, 0.4398, 0.4066)
STL10_STD = (0.2603, 0.2566, 0.2713)

def get_supervised_dataloaders():
    train_transform = transforms.Compose([
        transforms.RandomCrop(96, padding=12),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(STL10_MEAN, STL10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(STL10_MEAN, STL10_STD),
    ])

    train_dataset = datasets.STL10(
        root="./data_stl10",
        split="train",
        download=True,
        transform=train_transform,
    )
    test_dataset = datasets.STL10(
        root="./data_stl10",
        split="test",
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=CFG.batch_size,
        shuffle=True, num_workers=CFG.num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CFG.batch_size,
        shuffle=False, num_workers=CFG.num_workers,
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
    device = CFG.device
    model.eval()
    correct = 0
    total = 0
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

    import csv
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
# k-NN EVAL
# ==========================

def extract_embeddings(encoder, dataset, batch_size=256):
    device = CFG.device
    encoder = encoder.to(device)
    encoder.eval()

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=CFG.num_workers)

    all_feats = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            feats = encoder(x)
            feats = F.normalize(feats, dim=1)
            all_feats.append(feats.cpu())
            all_labels.append(y.clone())

    all_feats = torch.cat(all_feats, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_feats, all_labels

def knn_classifier(train_feats, train_labels, test_feats, test_labels, k=200):
    train_feats = train_feats.t()  # [D, Ntrain]
    num_test = test_feats.size(0)
    correct = 0

    with torch.no_grad():
        sim = torch.mm(test_feats, train_feats)          # [Ntest, Ntrain]
        topk = sim.topk(k=k, dim=1).indices              # [Ntest, k]
        neighbor_labels = train_labels[topk]             # [Ntest, k]

        for i in range(num_test):
            vals, counts = neighbor_labels[i].unique(return_counts=True)
            pred = vals[counts.argmax()]
            if pred.item() == test_labels[i].item():
                correct += 1

    return correct / num_test * 100.0

def knn_eval(encoder, train_dataset, test_dataset, k=200):
    encoder = encoder.to(CFG.device)
    encoder.eval()

    train_feats, train_labels = extract_embeddings(encoder, train_dataset)
    test_feats, test_labels = extract_embeddings(encoder, test_dataset)

    acc = knn_classifier(train_feats, train_labels, test_feats, test_labels, k=k)
    return acc

# ==========================
# MAIN
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
    encoders["Random"] = build_backbone()

    try:
        encoders["SimCLR"] = load_encoder_from_ckpt("stl10_simclr_encoder.pth")
    except FileNotFoundError:
        print("WARNING: stl10_simclr_encoder.pth not found, skipping SimCLR")

    try:
        encoders["MoCo"] = load_encoder_from_ckpt("stl10_moco_encoder.pth")
    except FileNotFoundError:
        print("WARNING: stl10_moco_encoder.pth not found, skipping MoCo")

    try:
        encoders["BYOL"] = load_encoder_from_ckpt("stl10_byol_encoder.pth")
    except FileNotFoundError:
        print("WARNING: stl10_byol_encoder.pth not found, skipping BYOL")

    results = []

    for name, enc in encoders.items():
        print(f"\n========== {name} : STL10 LINEAR EVAL ==========")
        log_path = f"stl10_linear_{name.lower()}_log.csv"
        best_test = train_linear(enc, train_loader, test_loader,
                                 title=f"STL10 {name} Linear Eval", log_path=log_path)

        print(f"\n========== {name} : STL10 k-NN EVAL ==========")
        knn_acc = knn_eval(enc, train_dataset, test_dataset, k=200)
        print(f"[STL10 {name}] k-NN accuracy: {knn_acc:.2f}%")

        results.append((name, best_test, knn_acc))

    print("\n========== STL10 SUMMARY ==========")
    print(f"{'Encoder':<10} | {'Best Linear Acc':>15} | {'k-NN Acc':>10}")
    print("-" * 42)
    for name, lin_acc, knn_acc in results:
        print(f"{name:<10} | {lin_acc:15.2f}% | {knn_acc:10.2f}%")

if __name__ == "__main__":
    main()
