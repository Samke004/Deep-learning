import math
import random
import copy
import csv
from dataclasses import dataclass
from typing import Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    data_root: str = "./data"
    num_classes: int = 10
    batch_size: int = 128
    num_workers: int = 2
    num_labeled_per_class: int = 400  # 400 * 10 = 4000 labelanih primjera
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 5e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Pseudo-labeling
    pl_threshold: float = 0.95

    # Pi-model
    pi_lambda: float = 1.0

    # MixMatch
    mm_T: float = 0.5  # temperature za "sharpen"
    mm_alpha: float = 0.75  # Beta distribucija za mixup
    mm_lambda_u: float = 75.0  # težina unsup loss-a

    # Co-training
    cotrain_threshold: float = 0.95

CFG = Config()

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


# ============================================================
# UTIL: SEED, TRANSFORMS, DATASETS
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms():
    # Slabi augment za labeled + evaluation
    weak_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    # Jednostavan transform za test
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return weak_train, test_transform


def split_cifar10_l_u(
    num_labeled_per_class: int
) -> Tuple[Subset, Subset, datasets.CIFAR10]:
    """
    Vrati:
      - labeled_subset
      - unlabeled_subset
      - test_dataset
    """
    weak_train, test_transform = get_transforms()

    full_train = datasets.CIFAR10(
        root=CFG.data_root, train=True, download=True, transform=weak_train
    )
    test_dataset = datasets.CIFAR10(
        root=CFG.data_root, train=False, download=True, transform=test_transform
    )

    labels = np.array(full_train.targets)
    labeled_indices = []
    unlabeled_indices = []

    for c in range(CFG.num_classes):
        idxs = np.where(labels == c)[0]
        np.random.shuffle(idxs)
        labeled_indices.extend(idxs[:num_labeled_per_class])
        unlabeled_indices.extend(idxs[num_labeled_per_class:])

    labeled_subset = Subset(full_train, labeled_indices)
    unlabeled_subset = Subset(full_train, unlabeled_indices)

    return labeled_subset, unlabeled_subset, test_dataset


def get_dataloaders():
    labeled_ds, unlabeled_ds, test_ds = split_cifar10_l_u(CFG.num_labeled_per_class)

    labeled_loader = DataLoader(
        labeled_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        drop_last=True,
    )
    unlabeled_loader = DataLoader(
        unlabeled_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
    )

    return labeled_loader, unlabeled_loader, test_loader


# ============================================================
# MODEL
# ============================================================

def get_backbone_resnet18(num_classes=10):
    """
    ResNet18 prilagođen za CIFAR-10 (manja slika).
    """
    model = models.resnet18(weights=None)
    # prilagodimo početni conv za 32x32 slike
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ============================================================
# METRIKE & EVAL
# ============================================================

def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    total_correct = 0
    total_examples = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_examples += x.size(0)
    return total_loss / total_examples, total_correct / total_examples


def log_csv(filename, header, rows):
    """
    rows: list of lists (jedan redak po epohi)
    """
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)


# ============================================================
# 0) SUPERVISED BASELINE
# ============================================================

def train_supervised(labeled_loader, test_loader, log_name="supervised_log.csv"):
    device = CFG.device
    model = get_backbone_resnet18(num_classes=CFG.num_classes).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    log_rows = []

    for epoch in range(1, CFG.epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_batches = 0

        for x, y in labeled_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = accuracy(logits, y)
            total_loss += loss.item()
            total_acc += acc
            total_batches += 1

        train_loss = total_loss / total_batches
        train_acc = total_acc / total_batches

        test_loss, test_acc = evaluate(model, test_loader, device)
        print(
            f"[SUP] Epoch [{epoch}/{CFG.epochs}] "
            f"TrainLoss: {train_loss:.4f} | TrainAcc: {train_acc*100:.2f}% | "
            f"TestAcc: {test_acc*100:.2f}%"
        )

        log_rows.append([epoch, train_loss, train_acc, test_loss, test_acc])

    log_csv(
        log_name,
        header=["epoch", "train_loss", "train_acc", "test_loss", "test_acc"],
        rows=log_rows,
    )
    return model


# ============================================================
# 1) PSEUDO-LABELING
# ============================================================

def train_pseudo_labeling(
    labeled_loader, unlabeled_loader, test_loader, log_name="pseudo_label_log.csv"
):
    """
    Osnovni self-training:
      1) treniramo na labeled batchu (supervised)
      2) na unlabeled batchu radimo predikcije
         -> ako je sigurnost >= threshold, tretiramo kao labelane
    """
    device = CFG.device
    model = get_backbone_resnet18(num_classes=CFG.num_classes).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    log_rows = []

    for epoch in range(1, CFG.epochs + 1):
        model.train()
        total_loss = 0.0
        total_sup_loss = 0.0
        total_unsup_loss = 0.0
        total_batches = 0

        unlabeled_iter = iter(unlabeled_loader)

        for x_l, y_l in labeled_loader:
            try:
                x_u, _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                x_u, _ = next(unlabeled_iter)

            x_l, y_l = x_l.to(device), y_l.to(device)
            x_u = x_u.to(device)

            # supervised dio
            logits_l = model(x_l)
            L_sup = criterion(logits_l, y_l)

            # pseudo-label dio
            with torch.no_grad():
                logits_u = model(x_u)
                probs_u = F.softmax(logits_u, dim=1)
                conf, pseudo_labels = torch.max(probs_u, dim=1)
                mask = conf.ge(CFG.pl_threshold).float()

            if mask.sum() > 0:
                # uzmi samo primjere s visokom sigurnošću
                selected_idx = mask.nonzero(as_tuple=False).squeeze(1)
                x_u_sel = x_u[selected_idx]
                y_u_sel = pseudo_labels[selected_idx]
                logits_u_sel = model(x_u_sel)
                L_unsup = criterion(logits_u_sel, y_u_sel)
            else:
                L_unsup = torch.tensor(0.0, device=device)

            loss = L_sup + L_unsup

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_sup_loss += L_sup.item()
            total_unsup_loss += L_unsup.item()
            total_batches += 1

        train_loss = total_loss / total_batches
        sup_loss = total_sup_loss / total_batches
        unsup_loss = total_unsup_loss / total_batches
        test_loss, test_acc = evaluate(model, test_loader, device)
        print(
            f"[PL] Epoch [{epoch}/{CFG.epochs}] "
            f"TrainLoss: {train_loss:.4f} | Sup: {sup_loss:.4f} | Unsup: {unsup_loss:.4f} | "
            f"TestAcc: {test_acc*100:.2f}%"
        )

        log_rows.append(
            [epoch, train_loss, sup_loss, unsup_loss, test_loss, test_acc]
        )

    log_csv(
        log_name,
        header=[
            "epoch",
            "train_loss",
            "sup_loss",
            "unsup_loss",
            "test_loss",
            "test_acc",
        ],
        rows=log_rows,
    )
    return model


# ============================================================
# 2) CO-TRAINING
# ============================================================

def train_cotraining(
    labeled_loader, unlabeled_loader, test_loader, log_name="cotraining_log.csv"
):
    """
    Vrlo pojednostavljen co-training:
      - imamo dva modela (m1, m2)
      - treniraju se na labeled batchu
      - zatim m1 labela unlabeled primjere za m2 i obrnuto
        (samo oni iznad confid. threshold)
    """
    device = CFG.device
    model1 = get_backbone_resnet18(num_classes=CFG.num_classes).to(device)
    model2 = get_backbone_resnet18(num_classes=CFG.num_classes).to(device)

    optim1 = torch.optim.Adam(
        model1.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    optim2 = torch.optim.Adam(
        model2.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    log_rows = []

    for epoch in range(1, CFG.epochs + 1):
        model1.train()
        model2.train()
        total_loss = 0.0
        total_batches = 0

        unlabeled_iter = iter(unlabeled_loader)

        for (x_l, y_l) in labeled_loader:
            try:
                x_u, _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                x_u, _ = next(unlabeled_iter)

            x_l, y_l = x_l.to(device), y_l.to(device)
            x_u = x_u.to(device)

            # --- supervised dio za oba modela ---
            logits1_l = model1(x_l)
            logits2_l = model2(x_l)

            L_sup1 = criterion(logits1_l, y_l)
            L_sup2 = criterion(logits2_l, y_l)

            # --- co-training dio na unlabeled ---
            with torch.no_grad():
                # model1 labela za model2
                logits1_u = model1(x_u)
                probs1 = F.softmax(logits1_u, dim=1)
                conf1, pseudo1 = probs1.max(dim=1)
                mask1 = conf1.ge(CFG.cotrain_threshold)
                # model2 labela za model1
                logits2_u = model2(x_u)
                probs2 = F.softmax(logits2_u, dim=1)
                conf2, pseudo2 = probs2.max(dim=1)
                mask2 = conf2.ge(CFG.cotrain_threshold)

            # gubitak za model1 na pseudo labelima od model2
            if mask2.sum() > 0:
                x_u2 = x_u[mask2]
                y_u2 = pseudo2[mask2]
                logits1_u2 = model1(x_u2)
                L_cotrain1 = criterion(logits1_u2, y_u2)
            else:
                L_cotrain1 = torch.tensor(0.0, device=device)

            # gubitak za model2 na pseudo labelima od model1
            if mask1.sum() > 0:
                x_u1 = x_u[mask1]
                y_u1 = pseudo1[mask1]
                logits2_u1 = model2(x_u1)
                L_cotrain2 = criterion(logits2_u1, y_u1)
            else:
                L_cotrain2 = torch.tensor(0.0, device=device)

            loss1 = L_sup1 + L_cotrain1
            loss2 = L_sup2 + L_cotrain2

            # update oba modela
            optim1.zero_grad()
            loss1.backward()
            optim1.step()

            optim2.zero_grad()
            loss2.backward()
            optim2.step()

            total_loss += (loss1.item() + loss2.item()) / 2.0
            total_batches += 1

        train_loss = total_loss / total_batches
        # evaluacija koristimo model1 (ili prosjek, ali za demo je dosta jedan)
        test_loss, test_acc = evaluate(model1, test_loader, device)
        print(
            f"[CoTrain] Epoch [{epoch}/{CFG.epochs}] "
            f"TrainLoss(avg): {train_loss:.4f} | TestAcc(model1): {test_acc*100:.2f}%"
        )

        log_rows.append([epoch, train_loss, test_loss, test_acc])

    log_csv(
        log_name,
        header=["epoch", "train_loss", "test_loss", "test_acc"],
        rows=log_rows,
    )
    return model1, model2


# ============================================================
# 3) Π-MODEL (CONSISTENCY REGULARIZATION)
# ============================================================

class SimpleAugment(nn.Module):
    """
    Jednostavna augmentacija koja radi nad Tensorima.
    Koristi se za consistency dio (dvije različite verzije istog x).
    """

    def __init__(self):
        super().__init__()
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
        ])

    def forward(self, x):
        # torchvision transformi rade i nad Tensorima (novije verzije),
        # ako treba, možemo pretvoriti u PIL pa natrag, ali za demo je ok.
        return self.aug(x)


def train_pi_model(
    labeled_loader, unlabeled_loader, test_loader, log_name="pimodel_log.csv"
):
    device = CFG.device
    model = get_backbone_resnet18(num_classes=CFG.num_classes).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    aug = SimpleAugment()

    log_rows = []

    for epoch in range(1, CFG.epochs + 1):
        model.train()
        total_loss = 0.0
        total_sup_loss = 0.0
        total_unsup_loss = 0.0
        total_batches = 0

        unlabeled_iter = iter(unlabeled_loader)

        for x_l, y_l in labeled_loader:
            try:
                x_u, _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                x_u, _ = next(unlabeled_iter)

            x_l, y_l = x_l.to(device), y_l.to(device)
            x_u = x_u.to(device)

            # supervised dio
            logits_l = model(x_l)
            L_sup = criterion(logits_l, y_l)

            # consistency dio na unlabeled: dvije augmentacije
            x_u1 = aug(x_u)
            x_u2 = aug(x_u)
            logits_u1 = model(x_u1)
            logits_u2 = model(x_u2)

            # Pi-model obično koristi MSE nad logitima ili probama
            p1 = F.softmax(logits_u1, dim=1)
            p2 = F.softmax(logits_u2, dim=1)
            L_cons = F.mse_loss(p1, p2)

            loss = L_sup + CFG.pi_lambda * L_cons

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_sup_loss += L_sup.item()
            total_unsup_loss += L_cons.item()
            total_batches += 1

        train_loss = total_loss / total_batches
        sup_loss = total_sup_loss / total_batches
        cons_loss = total_unsup_loss / total_batches
        test_loss, test_acc = evaluate(model, test_loader, device)

        print(
            f"[Pi] Epoch [{epoch}/{CFG.epochs}] "
            f"TrainLoss: {train_loss:.4f} | Sup: {sup_loss:.4f} | Cons: {cons_loss:.4f} | "
            f"TestAcc: {test_acc*100:.2f}%"
        )

        log_rows.append(
            [epoch, train_loss, sup_loss, cons_loss, test_loss, test_acc]
        )

    log_csv(
        log_name,
        header=[
            "epoch",
            "train_loss",
            "sup_loss",
            "cons_loss",
            "test_loss",
            "test_acc",
        ],
        rows=log_rows,
    )
    return model


# ============================================================
# 4) MIXMATCH
# ============================================================

def one_hot(labels, num_classes):
    return F.one_hot(labels, num_classes=num_classes).float()


def sharpen(p, T):
    """
    "Sharpen" distribucije (MixMatch paper).
    """
    p_power = p ** (1.0 / T)
    return p_power / p_power.sum(dim=1, keepdim=True)


def mixup(x, y, alpha):
    """
    Mixup nad ulazima i labelama (meke labele).
    """
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # simetrično
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


def train_mixmatch(
    labeled_loader, unlabeled_loader, test_loader, log_name="mixmatch_log.csv"
):
    """
    Pojednostavljena verzija MixMatch-a:
      1) za unlabeled radimo 2 augmentacije -> prosjek predikcija -> sharpen
      2) spajamo labeled i unlabeled primjere (s mekim labelama)
      3) radimo mixup nad inputima i labelama
      4) supervised loss na labeled dijelu, unsup loss na unlabeled dijelu
    """
    device = CFG.device
    model = get_backbone_resnet18(num_classes=CFG.num_classes).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )

    aug = SimpleAugment()
    log_rows = []

    for epoch in range(1, CFG.epochs + 1):
        model.train()
        total_loss = 0.0
        total_sup_loss = 0.0
        total_unsup_loss = 0.0
        total_batches = 0

        unlabeled_iter = iter(unlabeled_loader)

        for x_l, y_l in labeled_loader:
            try:
                x_u, _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                x_u, _ = next(unlabeled_iter)

            x_l, y_l = x_l.to(device), y_l.to(device)
            x_u = x_u.to(device)

            batch_size = x_l.size(0)

            # --- labeled kao one-hot ---
            y_l_onehot = one_hot(y_l, CFG.num_classes)

            # --- unlabeled pseudo-labele (2 augmentacije + sharpen) ---
            with torch.no_grad():
                x_u1 = aug(x_u)
                x_u2 = aug(x_u)
                logits_u1 = model(x_u1)
                logits_u2 = model(x_u2)
                p1 = F.softmax(logits_u1, dim=1)
                p2 = F.softmax(logits_u2, dim=1)
                pu = (p1 + p2) / 2.0
                q_u = sharpen(pu, T=CFG.mm_T)

            # --- spajamo labeled i unlabeled ---
            X = torch.cat([x_l, x_u], dim=0)
            Y = torch.cat([y_l_onehot, q_u], dim=0)

            # --- mixup ---
            X_mix, Y_mix = mixup(X, Y, alpha=CFG.mm_alpha)

            # podijelimo natrag na labeled + unlabeled
            X_l_mix = X_mix[:batch_size]
            Y_l_mix = Y_mix[:batch_size]
            X_u_mix = X_mix[batch_size:]
            Y_u_mix = Y_mix[batch_size:]

            # forward
            logits_l = model(X_l_mix)
            logits_u = model(X_u_mix)

            # supervised loss (cross-entropy s mekim labelama)
            log_probs_l = F.log_softmax(logits_l, dim=1)
            L_sup = -torch.mean(torch.sum(Y_l_mix * log_probs_l, dim=1))

            # unsup loss (MSE između probâ i mekih labela)
            probs_u = F.softmax(logits_u, dim=1)
            L_unsup = F.mse_loss(probs_u, Y_u_mix)

            loss = L_sup + CFG.mm_lambda_u * L_unsup

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_sup_loss += L_sup.item()
            total_unsup_loss += L_unsup.item()
            total_batches += 1

        train_loss = total_loss / total_batches
        sup_loss = total_sup_loss / total_batches
        unsup_loss = total_unsup_loss / total_batches
        test_loss, test_acc = evaluate(model, test_loader, device)

        print(
            f"[MixMatch] Epoch [{epoch}/{CFG.epochs}] "
            f"TrainLoss: {train_loss:.4f} | Sup: {sup_loss:.4f} | Unsup: {unsup_loss:.4f} | "
            f"TestAcc: {test_acc*100:.2f}%"
        )

        log_rows.append(
            [epoch, train_loss, sup_loss, unsup_loss, test_loss, test_acc]
        )

    log_csv(
        log_name,
        header=[
            "epoch",
            "train_loss",
            "sup_loss",
            "unsup_loss",
            "test_loss",
            "test_acc",
        ],
        rows=log_rows,
    )
    return model


# ============================================================
# SIMPLE "COMPARISON" PRINT
# ============================================================

def compare_from_logs():
    """
    Primjer kako možeš usporediti metode iz CSV logova:
    ovdje samo pročita zadnji redak svakog log fajla i ispiše test acc.
    U prezentaciji onda naglasiš:
      - koliko je test_acc skočio u odnosu na supervised
      - zašto je praćenje metrika po epohama ključno (overfitting, stabilnost, itd.)
    """
    import os

    def read_last_acc(path):
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            rows = list(csv.reader(f))
        if len(rows) <= 1:
            return None
        last = rows[-1]
        # pretpostavka: test_acc je zadnja kolona
        return float(last[-1])

    logs = {
        "Supervised": "supervised_log.csv",
        "Pseudo-Label": "pseudo_label_log.csv",
        "Co-Training": "cotraining_log.csv",
        "Pi-Model": "pimodel_log.csv",
        "MixMatch": "mixmatch_log.csv",
    }

    print("\n=== FINAL COMPARISON (Test Accuracy) ===")
    print("{:<15} | {:>8}".format("Method", "TestAcc"))
    print("-" * 28)
    for name, path in logs.items():
        acc = read_last_acc(path)
        if acc is None:
            print(f"{name:<15} |    N/A")
        else:
            print(f"{name:<15} | {acc*100:6.2f}%")


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(CFG.seed)
    print("Using device:", CFG.device)

    labeled_loader, unlabeled_loader, test_loader = get_dataloaders()

    # ================================
    # 1) Supervised baseline
    # ================================
    print("\n>>> Training Supervised Baseline")
    train_supervised(labeled_loader, test_loader, log_name="supervised_log.csv")

    # ================================
    # 2) Pseudo-labeling
    # ================================
    print("\n>>> Training Pseudo-Labeling")
    train_pseudo_labeling(
        labeled_loader, unlabeled_loader, test_loader, log_name="pseudo_label_log.csv"
    )

    # ================================
    # 3) Co-Training
    # ================================
    print("\n>>> Training Co-Training")
    train_cotraining(
        labeled_loader, unlabeled_loader, test_loader, log_name="cotraining_log.csv"
    )

    # ================================
    # 4) Pi-Model
    # ================================
    print("\n>>> Training Pi-Model")
    train_pi_model(
        labeled_loader, unlabeled_loader, test_loader, log_name="pimodel_log.csv"
    )

    # ================================
    # 5) MixMatch
    # ================================
    print("\n>>> Training MixMatch")
    train_mixmatch(
        labeled_loader, unlabeled_loader, test_loader, log_name="mixmatch_log.csv"
    )

    # ================================
    # Završna usporedba iz logova
    # ================================
    compare_from_logs()


if __name__ == "__main__":
    main()
