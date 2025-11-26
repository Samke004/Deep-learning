import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ============================================
# Helper: load csv safely
# ============================================
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        print(f"[WARNING] File not found: {path}")
        return None

# ============================================
# LOAD ALL LOG FILES
# ============================================
logs = {
    "SimCLR": {
        "linear": load_csv("linear_simclr_log.csv")
    },
    "MoCo": {
        "pretrain": load_csv("moco_pretrain_log.csv"),
        "linear": load_csv("linear_moco_log.csv")
    },
    "BYOL": {
        "pretrain": load_csv("byol_pretrain_log.csv"),
        "linear": load_csv("linear_byol_log.csv")
    },
    "Random": {
        "linear": load_csv("linear_random_log.csv")
    }
}

# ============================================
# 1) PRETRAINING LOSS CURVES
# ============================================
plt.figure(figsize=(10, 6))
for name in ["SimCLR", "MoCo", "BYOL"]:
    if logs[name].get("pretrain") is not None:
        df = logs[name]["pretrain"]
        plt.plot(df["epoch"], df["loss"], label=name)

plt.title("Pretraining Loss Curves")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/pretraining_loss.png")
plt.close()

print("Saved: pretraining_loss.png")

# ============================================
# 2) LINEAR EVAL – TEST ACCURACY CURVES
# ============================================
plt.figure(figsize=(10, 6))
for name in logs:
    df = logs[name].get("linear")
    if df is not None:
        plt.plot(df["epoch"], df["test_acc"], label=name)

plt.title("Linear Evaluation – Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/linear_eval_test_acc.png")
plt.close()

print("Saved: linear_eval_test_acc.png")

# ============================================
# 3) LINEAR EVAL – TRAIN LOSS CURVES
# ============================================
plt.figure(figsize=(10, 6))
for name in logs:
    df = logs[name].get("linear")
    if df is not None:
        plt.plot(df["epoch"], df["train_loss"], label=name)

plt.title("Linear Evaluation – Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/linear_eval_train_loss.png")
plt.close()

print("Saved: linear_eval_train_loss.png")

# ============================================
# 4) FINAL ACCURACY BAR PLOTS
# ============================================
summary = {
    "Encoder": [],
    "LinearAcc": [],
}

# Read linear final accuracy
for name, sets in logs.items():
    df = sets.get("linear")
    if df is not None:
        best_acc = df["test_acc"].max()
        summary["Encoder"].append(name)
        summary["LinearAcc"].append(best_acc)

summary_df = pd.DataFrame(summary)

plt.figure(figsize=(8, 6))
sns.barplot(data=summary_df, x="Encoder", y="LinearAcc")
plt.title("Best Linear Evaluation Accuracy")
plt.ylabel("Accuracy (%)")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/final_linear_accuracy.png")
plt.close()

print("Saved: final_linear_accuracy.png")

# ============================================
# (Optional) 5) SHOW SOME RESULTS IN TERMINAL
# ============================================
print("\n===== SUMMARY (Best Linear Accuracy) =====")
print(summary_df)

print("\nAll plots saved in 'plots/' folder.")
