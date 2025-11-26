import os
from dataclasses import dataclass
from typing import Dict
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[WARNING] 'umap-learn' not installed. UMAP vizualizacija će biti preskočena.")

sns.set(style="whitegrid")

# ============================================
# CONFIG
# ============================================

@dataclass
class Config:
    batch_size: int = 256
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_embed_points: int = 2000   # koliko točaka za t-SNE/UMAP
    knn_k: int = 200
    plots_dir: str = "plots"

CFG = Config()
os.makedirs(CFG.plots_dir, exist_ok=True)

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ============================================
# DATA
# ============================================

def get_test_dataset():
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True,
        transform=test_transform
    )
    return test_dataset

def get_train_test_datasets():
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
    return train_dataset, test_dataset

# ============================================
# ENCODERI
# ============================================

def build_backbone():
    encoder = models.resnet18(weights=None)
    encoder.fc = nn.Identity()
    return encoder

def load_encoder_from_ckpt(path):
    encoder = build_backbone()
    state = torch.load(path, map_location="cpu")
    encoder.load_state_dict(state)
    return encoder

# ============================================
# EMBEDDING EXTRACTOR
# ============================================

def extract_embeddings(encoder, dataset, max_points=None):
    """
    Vrati (embeddings [N, D], labels [N]) za zadani encoder i dataset.
    Ako je max_points zadano, uzima se podskup.
    """
    device = CFG.device
    encoder = encoder.to(device)
    encoder.eval()

    if max_points is not None and max_points < len(dataset):
        indices = list(range(max_points))
        dataset = Subset(dataset, indices)

    loader = DataLoader(dataset, batch_size=CFG.batch_size,
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

    all_feats = torch.cat(all_feats, dim=0)   # [N, D]
    all_labels = torch.cat(all_labels, dim=0) # [N]
    return all_feats.numpy(), all_labels.numpy()

# ============================================
# k-NN EVAL
# ============================================

def knn_classifier(train_feats, train_labels, test_feats, test_labels, k=200):
    train_feats = torch.from_numpy(train_feats)  # [Ntrain, D]
    test_feats = torch.from_numpy(test_feats)    # [Ntest, D]
    train_labels = torch.from_numpy(train_labels)
    test_labels = torch.from_numpy(test_labels)

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

def knn_eval_encoder(encoder, train_dataset, test_dataset, k=200, max_train=None, max_test=None):
    if max_train is not None and max_train < len(train_dataset):
        train_dataset = Subset(train_dataset, list(range(max_train)))
    if max_test is not None and max_test < len(test_dataset):
        test_dataset = Subset(test_dataset, list(range(max_test)))

    train_feats, train_labels = extract_embeddings(encoder, train_dataset, max_points=None)
    test_feats, test_labels = extract_embeddings(encoder, test_dataset, max_points=None)
    acc = knn_classifier(train_feats, train_labels, test_feats, test_labels, k=k)
    return acc

# ============================================
# t-SNE / UMAP PLOT
# ============================================

def plot_embedding_2d(X_2d, labels, title, filename):
    plt.figure(figsize=(8, 6))
    num_classes = len(np.unique(labels))
    palette = sns.color_palette("tab10", num_classes)

    for class_idx in range(num_classes):
        idxs = labels == class_idx
        plt.scatter(
            X_2d[idxs, 0],
            X_2d[idxs, 1],
            s=8,
            color=palette[class_idx],
            label=CIFAR10_CLASSES[class_idx],
            alpha=0.7
        )

    plt.title(title)
    plt.legend(markerscale=2, fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    out_path = os.path.join(CFG.plots_dir, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def compute_tsne(X, perplexity=30, n_iter=1000):
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, init="pca", learning_rate="auto")
    X_2d = tsne.fit_transform(X)
    return X_2d

def compute_umap(X, n_neighbors=15, min_dist=0.1):
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist)
    X_2d = reducer.fit_transform(X)
    return X_2d

# ============================================
# MAIN
# ============================================

def main():
    print("Using device:", CFG.device)

    # 1) Test set za embedding vizualizacije
    test_dataset = get_test_dataset()

    # Uzmemo podskup za vizualizaciju (npr. prvih 2000)
    subset_indices = list(range(min(CFG.max_embed_points, len(test_dataset))))
    subset = Subset(test_dataset, subset_indices)

    # --- Random encoder ---
    enc_random = build_backbone()
    feats_random, labels_random = extract_embeddings(enc_random, subset, max_points=None)
    print("Random embeddings shape:", feats_random.shape)

    # --- SimCLR encoder ---
    enc_simclr = load_encoder_from_ckpt("simclr_encoder.pth")
    feats_simclr, labels_simclr = extract_embeddings(enc_simclr, subset, max_points=None)
    print("SimCLR embeddings shape:", feats_simclr.shape)

    # ========== t-SNE ==========
    print("\nComputing t-SNE for Random encoder...")
    X2d_random_tsne = compute_tsne(feats_random)
    plot_embedding_2d(
        X2d_random_tsne,
        labels_random,
        title="t-SNE – Random encoder (CIFAR-10)",
        filename="tsne_random.png"
    )

    print("Computing t-SNE for SimCLR encoder...")
    X2d_simclr_tsne = compute_tsne(feats_simclr)
    plot_embedding_2d(
        X2d_simclr_tsne,
        labels_simclr,
        title="t-SNE – SimCLR encoder (CIFAR-10)",
        filename="tsne_simclr.png"
    )

    # ========== UMAP ==========
    if HAS_UMAP:
        print("\nComputing UMAP for Random encoder...")
        X2d_random_umap = compute_umap(feats_random)
        plot_embedding_2d(
            X2d_random_umap,
            labels_random,
            title="UMAP – Random encoder (CIFAR-10)",
            filename="umap_random.png"
        )

        print("Computing UMAP for SimCLR encoder...")
        X2d_simclr_umap = compute_umap(feats_simclr)
        plot_embedding_2d(
            X2d_simclr_umap,
            labels_simclr,
            title="UMAP – SimCLR encoder (CIFAR-10)",
            filename="umap_simclr.png"
        )
    else:
        print("\n[INFO] UMAP nije instaliran, preskačem UMAP vizualizacije.")

    # ========== k-NN BAR PLOT ==========
    print("\nComputing k-NN accuracies for all encoders...")
    train_dataset, test_dataset_full = get_train_test_datasets()

    encoders: Dict[str, nn.Module] = {
        "Random": build_backbone(),
        "SimCLR": enc_simclr
    }

    # MoCo
    if os.path.exists("moco_encoder.pth"):
        encoders["MoCo"] = load_encoder_from_ckpt("moco_encoder.pth")
    else:
        print("[WARNING] moco_encoder.pth not found, skipping MoCo.")

    # BYOL
    if os.path.exists("byol_encoder.pth"):
        encoders["BYOL"] = load_encoder_from_ckpt("byol_encoder.pth")
    else:
        print("[WARNING] byol_encoder.pth not found, skipping BYOL.")

    knn_results = {"Encoder": [], "kNN_Acc": []}

    for name, enc in encoders.items():
        print(f"  -> k-NN for {name} (k={CFG.knn_k}) ...")
        acc = knn_eval_encoder(
            enc, train_dataset, test_dataset_full,
            k=CFG.knn_k,
            max_train=None,
            max_test=None
        )
        print(f"     {name} k-NN accuracy: {acc:.2f}%")
        knn_results["Encoder"].append(name)
        knn_results["kNN_Acc"].append(acc)

    # Plot k-NN rezultata
    import pandas as pd
    df_knn = pd.DataFrame(knn_results)
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_knn, x="Encoder", y="kNN_Acc")
    plt.title(f"k-NN Accuracy (k={CFG.knn_k}) na CIFAR-10 embeddingu")
    plt.ylabel("Accuracy (%)")
    plt.tight_layout()
    out_bar = os.path.join(CFG.plots_dir, "knn_accuracies.png")
    plt.savefig(out_bar, dpi=300)
    plt.close()
    print(f"Saved: {out_bar}")

    print("\nGotovo! Svi grafovi su u folderu:", CFG.plots_dir)


if __name__ == "__main__":
    main()
