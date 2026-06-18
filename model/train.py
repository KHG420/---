"""
ResNet50 + CBAM 训练脚本
"""
import os, sys, time, json, argparse, numpy as np
from tqdm import tqdm
import torch, torch.nn as nn, torch.optim as optim

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import create_model
from dataset import create_dataloaders, CLASS_NAMES


def parse_args():
    p = argparse.ArgumentParser(description="ResNet50+CBAM 训练")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./checkpoints")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--freeze_epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    return p.parse_args()


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    running_loss = correct = total = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.4f}"})
    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, epoch=None):
    model.eval()
    running_loss = correct = total = 0
    desc = f"Epoch {epoch} [Val]" if epoch else "Test"
    pbar = tqdm(loader, desc=desc, leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.4f}"})
    return running_loss / total, correct / total


def main():
    args = parse_args()
    set_seed(args.seed)

    # 设备检测
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"设备: {device}")
    print(f"PyTorch: {torch.__version__}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print("\n加载数据集...")
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        args.data_dir, args.batch_size, args.input_size, args.num_workers)
    print(f"类别: {class_names}")

    print("\n创建模型: ResNet50 + CBAM")
    model = create_model(len(class_names), pretrained=True, device=device)
    print(f"参数总量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-7)
    # 混合精度 — MPS/CPU 不需要 GradScaler
    scaler = torch.amp.GradScaler(device_type=device.type) if device.type == "cuda" else None

    start_epoch = 1
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    # 阶段1: 冻结backbone训练
    if args.freeze_epochs > 0:
        print(f"\n阶段1: 冻结backbone ({args.freeze_epochs} epochs)")
        for name, param in model.named_parameters():
            if "classifier" not in name and "cbam" not in name:
                param.requires_grad = False

        freeze_optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr * 2, weight_decay=args.weight_decay)

        for epoch in range(1, args.freeze_epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, freeze_optimizer, scaler, device, epoch)
            val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)
            print(f"Freeze Epoch {epoch:2d}/{args.freeze_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            history["train_loss"].append(train_loss); history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc; best_epoch = epoch

        for param in model.parameters():
            param.requires_grad = True
        print(f"冻结阶段完成, 最佳val_acc: {best_val_acc:.4f}")

    # 阶段2: 全参数微调
    print("\n阶段2: 全参数微调")
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss); history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc); history["lr"].append(current_lr)

        print(f"Epoch {epoch:2d}/{args.epochs} | LR: {current_lr:.2e} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc; best_epoch = epoch; patience_counter = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc, "best_epoch": best_epoch,
                "class_names": class_names, "args": vars(args),
                "history": history,
            }, os.path.join(args.output_dir, "best_model.pth"))
            print(f"  ✅ 保存最佳模型 (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\n⏹️ 早停: {args.patience} epochs 未提升")
            break

    # 测试
    print("\n测试集评估...")
    checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"\n📊 测试集 Accuracy: {test_acc * 100:.2f}%")
    if test_acc >= 0.92:
        print(f"🎉 准确率 ≥ 92%，满足要求！")
    else:
        print(f"⚠️ 准确率 < 92%，需调整超参数")

    summary = {"best_val_acc": float(best_val_acc), "best_epoch": best_epoch,
               "test_acc": float(test_acc), "test_loss": float(test_loss),
               "meets_requirement": test_acc >= 0.92}
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n🎯 训练完成！最佳验证准确率: {best_val_acc*100:.2f}% | 测试准确率: {test_acc*100:.2f}%")
    return test_acc


if __name__ == "__main__":
    main()
