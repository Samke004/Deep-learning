import csv
import os
import matplotlib.pyplot as plt

# ============================================
# Helper: čitanje kolona iz CSV-a
# ============================================

def load_column_from_csv(path, col_name):
    epochs = []
    values = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            values.append(float(row[col_name]))
    return epochs, values


# ============================================
# 1) Test accuracy svih metoda kroz epohe
# ============================================

def plot_test_accuracy_all():
    methods = {
        "Supervised": "supervised_log.csv",
        "Pseudo-Label": "pseudo_label_log.csv",
        "Co-Training": "cotraining_log.csv",
        "Pi-Model": "pimodel_log.csv",
        "MixMatch": "mixmatch_log.csv",
    }

    plt.figure()
    for name, path in methods.items():
        if not os.path.exists(path):
            print(f"[WARN] File {path} not found, skipping {name}.")
            continue
        epochs, test_acc = load_column_from_csv(path, "test_acc")
        test_acc = [a * 100 for a in test_acc]  # iz [0,1] u %
        plt.plot(epochs, test_acc, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy [%]")
    plt.title("Test Accuracy kroz epohe - usporedba metoda")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================
# 2) Pseudo-Label: supervised vs unsupervised loss
# ============================================

def plot_pseudo_label_losses():
    path = "pseudo_label_log.csv"
    if not os.path.exists(path):
        print(f"[WARN] {path} not found, skipping Pseudo-Label losses plot.")
        return

    epochs, sup_loss = load_column_from_csv(path, "sup_loss")
    _, unsup_loss = load_column_from_csv(path, "unsup_loss")

    plt.figure()
    plt.plot(epochs, sup_loss, label="Supervised loss")
    plt.plot(epochs, unsup_loss, label="Unsup (pseudo-label) loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Pseudo-Label: supervised vs unsupervised loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================
# 3) Pi-Model: supervised vs consistency loss
# ============================================

def plot_pi_model_losses():
    path = "pimodel_log.csv"
    if not os.path.exists(path):
        print(f"[WARN] {path} not found, skipping Pi-Model losses plot.")
        return

    epochs, sup_loss = load_column_from_csv(path, "sup_loss")
    _, cons_loss = load_column_from_csv(path, "cons_loss")

    plt.figure()
    plt.plot(epochs, sup_loss, label="Supervised loss")
    plt.plot(epochs, cons_loss, label="Consistency loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Pi-Model: supervised vs consistency loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================
# 4) MixMatch: supervised vs unsupervised loss
# ============================================

def plot_mixmatch_losses():
    path = "mixmatch_log.csv"
    if not os.path.exists(path):
        print(f"[WARN] {path} not found, skipping MixMatch losses plot.")
        return

    epochs, sup_loss = load_column_from_csv(path, "sup_loss")
    _, unsup_loss = load_column_from_csv(path, "unsup_loss")

    plt.figure()
    plt.plot(epochs, sup_loss, label="Supervised loss")
    plt.plot(epochs, unsup_loss, label="Unsup (MixMatch) loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MixMatch: supervised vs unsupervised loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================
# MAIN
# ============================================

def main():
    # 1) Usporedba test accuracy za sve metode
    plot_test_accuracy_all()

    # 2) Pojedinačni grafovi loss-eva
    plot_pseudo_label_losses()
    plot_pi_model_losses()
    plot_mixmatch_losses()


if __name__ == "__main__":
    main()
