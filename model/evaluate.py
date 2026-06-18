"""
模型评估 — 混淆矩阵、ROC曲线、PR曲线、分类报告
"""
import os, sys, argparse, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import create_model
from dataset import create_dataloaders, CLASS_NAMES

# 设置中文字体（macOS 系统字体）
plt.rcParams["font.sans-serif"] = ["PingFang HK"] + plt.rcParams["font.sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./evaluation")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--input_size", type=int, default=224)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(labels, preds, names, en_names, path):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=en_names, yticklabels=en_names, ax=axes[0])
    axes[0].set_title("混淆矩阵（计数）")
    axes[0].set_xlabel("预测类别")
    axes[0].set_ylabel("真实类别")
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=en_names, yticklabels=en_names, ax=axes[1])
    axes[1].set_title("混淆矩阵（归一化）")
    axes[1].set_xlabel("预测类别")
    axes[1].set_ylabel("真实类别")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"混淆矩阵: {path}")


def plot_roc(labels, probs, names, en_names, path):
    n = len(names)
    labels_oh = np.eye(n)[labels]
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
    for i in range(n):
        fpr, tpr, _ = roc_curve(labels_oh[:, i], probs[:, i])
        ax.plot(fpr, tpr, lw=2, color=colors[i % len(colors)],
                label=f"{en_names[i]} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.7)
    ax.set_xlabel("假阳性率 (FPR)")
    ax.set_ylabel("真阳性率 (TPR)")
    ax.set_title("ROC 曲线")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC曲线: {path}")


def plot_pr(labels, probs, names, en_names, path):
    n = len(names)
    labels_oh = np.eye(n)[labels]
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
    for i in range(n):
        p, r, _ = precision_recall_curve(labels_oh[:, i], probs[:, i])
        ap = average_precision_score(labels_oh[:, i], probs[:, i])
        ax.plot(r, p, lw=2, color=colors[i % len(colors)],
                label=f"{en_names[i]} (AP={ap:.3f})")
    ax.set_xlabel("召回率 (Recall)")
    ax.set_ylabel("精确率 (Precision)")
    ax.set_title("PR 曲线")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PR曲线: {path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"设备: {device}")

    _, _, test_loader, names = create_dataloaders(args.data_dir, args.batch_size, args.input_size)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = create_model(len(names), pretrained=False, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("模型加载成功")

    labels, preds, probs = evaluate(model, test_loader, device)
    accuracy = np.mean(labels == preds)
    print(f"\n📊 测试集准确率: {accuracy*100:.2f}%")
    print("\n分类报告:\n", classification_report(labels, preds, target_names=names, digits=3))

    plot_confusion_matrix(labels, preds, names, CLASS_NAMES, os.path.join(args.output_dir, "confusion_matrix.png"))
    plot_roc(labels, probs, names, CLASS_NAMES, os.path.join(args.output_dir, "roc_curves.png"))
    plot_pr(labels, probs, names, CLASS_NAMES, os.path.join(args.output_dir, "pr_curves.png"))

    with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
        f.write(f"Test Accuracy: {accuracy*100:.3f}%\n\n")
        f.write(classification_report(labels, preds, target_names=names, digits=3))

    print(f"{'🎉' if accuracy>=0.92 else '⚠️'} 准确率 {accuracy*100:.2f}% {'≥' if accuracy>=0.92 else '<'} 92%")


if __name__ == "__main__":
    main()
