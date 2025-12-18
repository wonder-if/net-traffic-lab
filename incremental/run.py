# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from contextlib import contextmanager
import random
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# === 导入您的模型 ===
from models_adapted import OpenEmbed


# ---------------------- 全局稳定性（可复现实验） ----------------------
def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


set_global_seed(42)

# 可选增强：若 data_utils 缺失则兜底为恒等
try:
    from utils import signal_awgn, freq_perturb, time_scaling
except Exception:
    def signal_awgn(x):
        return x


    def freq_perturb(x):
        return x


    def time_scaling(x):
        return x


# ======================= 配置 =======================
class Cfg:
    # 路径
    data_root = "../data/ustc_all"
    teacher_model_path = "ustc_model/15_ustcALL_0.pt"  # 预训练的15类模型
    new_class_data_paths = (
        "../data/ustc_all/15.csv",
        "../data/ustc_all/16.csv",
        "../data/ustc_all/17.csv",
        "../data/ustc_all/18.csv",
        "../data/ustc_all/19.csv",
    )
    out_dir = "./checkpoints_ustc"
    final_model = os.path.join(out_dir, "gem_final.pth")

    # 可视化保存路径
    vis_dir = "./visualizations"
    loss_plot_path = os.path.join(vis_dir, "training_loss.png")
    accuracy_plot_path = os.path.join(vis_dir, "accuracy_evolution.png")
    confusion_matrix_path = os.path.join(vis_dir, "confusion_matrix.png")

    # 类别
    known_classes = tuple(range(15))  # 0-14为已知类
    new_class_labels = (15, 16, 17, 18, 19)  # 新增5个类别
    _unknown_for_loader = (15, 19)  # 用于数据加载器接口兼容

    # 训练
    epochs = 10
    batch_size = 1024
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 0
    pin_memory = True
    print_every = 20
    do_eval_during_train = True
    eval_every = 1

    # 优化器
    optimizer_type = "adamw"  # 'sgd' or 'adamw'
    lr = 1e-3
    weight_decay = 0.0  # 建议 0：避免破坏 GEM 几何约束
    momentum = 0.9
    nesterov = True
    label_smoothing = 0.1

    # GEM 参数
    gem_margin = 0.5  # 可试 5e-4 ~ 1e-3 增强不可遗忘硬度
    gem_num_ref_batches = 5  # K=1 走闭式解；>1 会走 QP/PGD

    # 记忆池（从旧类全量训练集均匀采样）
    memory_per_class = 100

    # 行为开关
    freeze_old_rows_with_hook = False  # True: 只训新增类别行
    show_teacher_upper_bound = True  # 打印教师模型旧类上限


# ======================= 可视化工具 =======================
class Visualization:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.training_losses = []
        self.eval_metrics = []  # 保存每个epoch的评估结果
        self.epochs_list = []

        # 创建可视化目录
        os.makedirs(cfg.vis_dir, exist_ok=True)

    def add_training_loss(self, epoch, loss):
        """记录训练损失"""
        self.training_losses.append((epoch, loss))

    def add_eval_metrics(self, epoch, metrics):
        """记录评估指标"""
        self.eval_metrics.append((epoch, metrics))
        self.epochs_list.append(epoch)

    def plot_training_loss(self):
        """绘制训练损失曲线"""
        if not self.training_losses:
            print("没有训练损失数据可绘制")
            return

        epochs = [e for e, _ in self.training_losses]
        losses = [l for _, l in self.training_losses]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(self.cfg.loss_plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"训练损失曲线已保存到: {self.cfg.loss_plot_path}")

    def plot_accuracy_evolution(self):
        """绘制准确率演化曲线"""
        if not self.eval_metrics:
            print("没有评估指标数据可绘制")
            return

        epochs = self.epochs_list
        old_acc = [metrics['old'] for _, metrics in self.eval_metrics]
        new_acc = [metrics['new'] for _, metrics in self.eval_metrics]
        overall_acc = [metrics['overall'] for _, metrics in self.eval_metrics]

        plt.figure(figsize=(12, 7))
        plt.plot(epochs, old_acc, 'r-o', linewidth=2, markersize=8, label='Old Classes Accuracy')
        plt.plot(epochs, new_acc, 'g-s', linewidth=2, markersize=8, label='New Classes Accuracy')
        plt.plot(epochs, overall_acc, 'b-^', linewidth=2, markersize=8, label='Overall Accuracy')

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Accuracy Evolution During Training', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, loc='best')
        plt.ylim([0, 1.05])

        # 添加准确率数值标注
        for i, (old, new, overall) in enumerate(zip(old_acc, new_acc, overall_acc)):
            if i % 2 == 0:  # 每隔一个epoch标注一次
                plt.annotate(f'{old:.3f}', (epochs[i], old_acc[i]), textcoords="offset points",
                             xytext=(0, 10), ha='center', fontsize=9, color='red')
                plt.annotate(f'{new:.3f}', (epochs[i], new_acc[i]), textcoords="offset points",
                             xytext=(0, -15), ha='center', fontsize=9, color='green')

        plt.tight_layout()
        plt.savefig(self.cfg.accuracy_plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"准确率演化曲线已保存到: {self.cfg.accuracy_plot_path}")

    def plot_confusion_matrix(self, model, test_old_loader, new_test_loader, device):
        """绘制混淆矩阵"""
        model.eval()
        all_preds = []
        all_labels = []

        # 收集旧类预测
        with torch.no_grad():
            for x, y in test_old_loader:
                x, y = x.to(device), y.to(device)
                _, logits = model(x)
                preds = logits.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        # 收集新类预测
        with torch.no_grad():
            for x, y in new_test_loader:
                x, y = x.to(device), y.to(device)
                _, logits = model(x)
                preds = logits.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        # 创建混淆矩阵
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)

        # 绘制混淆矩阵
        plt.figure(figsize=(14, 12))

        # 创建类别标签
        num_old = len(self.cfg.known_classes)
        num_new = len(self.cfg.new_class_labels)
        total_classes = num_old + num_new
        class_names = [f'Old-{i}' for i in range(num_old)] + [f'New-{i}' for i in self.cfg.new_class_labels]

        # 使用seaborn绘制热力图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'})

        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')

        # 添加分隔线区分新旧类别
        plt.axhline(y=num_old, color='red', linewidth=2, linestyle='--', alpha=0.7)
        plt.axvline(x=num_old, color='red', linewidth=2, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(self.cfg.confusion_matrix_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"混淆矩阵已保存到: {self.cfg.confusion_matrix_path}")

        return cm

    def print_final_report(self, final_metrics, confusion_mat):
        """打印最终训练报告"""
        print("\n" + "=" * 80)
        print("训练完成 - 最终报告")
        print("=" * 80)

        # 基本信息
        print(f"\n📊 模型性能指标:")
        print(f"   旧类别准确率: {final_metrics['old']:.4f} ({final_metrics['old'] * 100:.2f}%)")
        print(f"   新类别准确率: {final_metrics['new']:.4f} ({final_metrics['new'] * 100:.2f}%)")
        print(f"   总体准确率:   {final_metrics['overall']:.4f} ({final_metrics['overall'] * 100:.2f}%)")

        # 混淆矩阵统计
        print(f"\n📈 混淆矩阵统计:")
        print(f"   总样本数: {confusion_mat.sum()}")
        print(f"   正确分类数: {np.trace(confusion_mat)}")
        print(f"   整体准确率: {np.trace(confusion_mat) / confusion_mat.sum():.4f}")

        # 新旧类别分离统计
        num_old = len(self.cfg.known_classes)
        old_old = confusion_mat[:num_old, :num_old].sum()
        old_new = confusion_mat[:num_old, num_old:].sum()
        new_old = confusion_mat[num_old:, :num_old].sum()
        new_new = confusion_mat[num_old:, num_old:].sum()

        print(f"\n🔍 新旧类别交叉分析:")
        print(f"   旧类被正确识别为旧类: {old_old}")
        print(f"   旧类被误识别为新类:   {old_new}")
        print(f"   新类被误识别为旧类:   {new_old}")
        print(f"   新类被正确识别为新类: {new_new}")

        # 保存报告到文件
        report_path = os.path.join(self.cfg.vis_dir, "training_report.txt")
        with open(report_path, 'w') as f:
            f.write("训练完成报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"训练轮次: {self.cfg.epochs}\n")
            f.write(f"已知类别数: {len(self.cfg.known_classes)}\n")
            f.write(f"新增类别数: {len(self.cfg.new_class_labels)}\n")
            f.write(f"记忆样本/类: {self.cfg.memory_per_class}\n\n")

            f.write("性能指标:\n")
            f.write(f"  旧类别准确率: {final_metrics['old']:.4f}\n")
            f.write(f"  新类别准确率: {final_metrics['new']:.4f}\n")
            f.write(f"  总体准确率:   {final_metrics['overall']:.4f}\n\n")

            f.write("可视化文件:\n")
            f.write(f"  损失曲线: {self.cfg.loss_plot_path}\n")
            f.write(f"  准确率曲线: {self.cfg.accuracy_plot_path}\n")
            f.write(f"  混淆矩阵: {self.cfg.confusion_matrix_path}\n")

        print(f"\n📄 详细报告已保存到: {report_path}")


# ======================= 数据加载和处理 =======================
def load_csv_data(file_path):
    """加载CSV文件数据，处理混合类型"""
    try:
        # 方法1: 尝试自动转换类型
        df = pd.read_csv(file_path, header=None, low_memory=False)

        # 清理数据：将非数值数据转换为NaN，然后填充为0
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 填充NaN值为0
        df = df.fillna(0)

        data = df.values.astype(np.float32)

        print(f"Loaded {file_path}: shape {data.shape}")
        return data

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        # 方法2: 如果自动转换失败，使用更稳健的方法
        try:
            data_list = []
            with open(file_path, 'r') as f:
                for line in f:
                    # 分割每行的数据
                    parts = line.strip().split(',')
                    row_data = []
                    for part in parts:
                        try:
                            # 尝试转换为float
                            val = float(part)
                            row_data.append(val)
                        except ValueError:
                            # 如果转换失败，使用0
                            row_data.append(0.0)
                    data_list.append(row_data)

            # 确保所有行长度一致
            max_len = max(len(row) for row in data_list)
            for i in range(len(data_list)):
                while len(data_list[i]) < max_len:
                    data_list[i].append(0.0)
                data_list[i] = data_list[i][:max_len]  # 截断到最大长度

            data = np.array(data_list, dtype=np.float32)
            print(f"Loaded {file_path} with fallback method: shape {data.shape}")
            return data

        except Exception as e2:
            print(f"Fallback method also failed for {file_path}: {e2}")
            return np.empty((0, 784), dtype=np.float32)  # 返回空数组


def load_all_data(data_root, classes):
    """加载指定类别的所有数据"""
    X_list, y_list = [], []
    for cls in classes:
        file_path = os.path.join(data_root, f"{cls}.csv")
        if os.path.exists(file_path):
            data = load_csv_data(file_path)
            if len(data) > 0:
                X_list.append(data)
                y_list.append(np.full((data.shape[0],), cls, dtype=np.int64))
                print(f"Class {cls}: loaded {data.shape[0]} samples")
            else:
                print(f"Warning: No data loaded from {file_path}")
        else:
            print(f"Warning: {file_path} not found")

    if len(X_list) == 0:
        print("Error: No data loaded for any class!")
        return np.empty((0, 784)), np.empty((0,), dtype=np.int64)

    X = np.vstack(X_list)
    y = np.hstack(y_list)

    print(f"Total loaded: {X.shape[0]} samples, {y.shape[0]} labels")
    return X, y


# ======================= 实用：安全增强 + Dataset =======================
def _to_tensor(x):
    return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)


def safe_augment(x: torch.Tensor) -> torch.Tensor:
    x = _to_tensor(x).to(torch.float32)
    x = _to_tensor(signal_awgn(x)).to(torch.float32)
    x = _to_tensor(freq_perturb(x)).to(torch.float32)
    x = _to_tensor(time_scaling(x)).to(torch.float32)
    return x


class SignalDataset(Dataset):
    def __init__(self, X, y, transform=None):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long()

        self.X = X.detach().cpu().float()
        self.y = y.detach().cpu().long()
        self.tf = transform

        # 确保数据形状正确 (batch_size, 784)
        if self.X.dim() == 1:
            self.X = self.X.unsqueeze(0)

        # 如果数据长度不是784，进行调整
        if self.X.shape[1] > 784:
            self.X = self.X[:, :784]  # 截断
        elif self.X.shape[1] < 784:
            # 填充到784
            padding = torch.zeros(self.X.shape[0], 784 - self.X.shape[1])
            self.X = torch.cat([self.X, padding], dim=1)

        print(f"Dataset: X shape {self.X.shape}, y shape {self.y.shape}")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        x, y = self.X[i], self.y[i]
        if self.tf is not None:
            try:
                x = self.tf(x)
            except Exception:
                pass
        return x, y


# ======================= Data 构建（新数据 + 旧类记忆 + 测试） =======================
def build_loaders_and_memory(cfg: Cfg):
    # === 新类整池：把新簇标为 [num_old, num_old+1, ...] ===
    X_new, y_new = [], []
    num_old = len(cfg.known_classes)

    print("Loading new class data...")
    for i, p in enumerate(cfg.new_class_data_paths):
        if os.path.exists(p):
            arr = load_csv_data(p)
            if len(arr) > 0:
                X_new.append(torch.from_numpy(arr).float())
                y_new.append(torch.full((arr.shape[0],), num_old + i, dtype=torch.long))
                print(f"New class {num_old + i}: {arr.shape[0]} samples")
        else:
            print(f"Warning: {p} not found")

    if len(X_new) == 0:
        raise ValueError("No new class data found!")

    X_new = torch.cat(X_new, 0)
    y_new = torch.cat(y_new, 0)

    new_train_loader = DataLoader(
        SignalDataset(X_new, y_new, transform=safe_augment),
        batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=False
    )

    # === 旧类训练全集 & 测试集 ===
    print("Loading old class data...")
    X_old_train, y_old_train = load_all_data(cfg.data_root, cfg.known_classes)

    if len(X_old_train) == 0:
        raise ValueError("No old class data found!")

    # 分割训练测试集 (8:2)
    from sklearn.model_selection import train_test_split
    X_old_train, X_old_test, y_old_train, y_old_test = train_test_split(
        X_old_train, y_old_train, test_size=0.2, random_state=42, stratify=y_old_train
    )

    old_train_loader = DataLoader(
        SignalDataset(X_old_train, y_old_train),
        batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )

    old_test_loader = DataLoader(
        SignalDataset(X_old_test, y_old_test),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )

    # === 新类测试集 ===
    print("Creating new class test set...")
    X_new_test, y_new_test = [], []
    for i, p in enumerate(cfg.new_class_data_paths):
        if os.path.exists(p):
            arr = load_csv_data(p)
            if len(arr) > 0:
                # 取20%作为测试
                if len(arr) > 10:  # 确保有足够样本
                    arr_train, arr_test = train_test_split(arr, test_size=0.2, random_state=42)
                    X_new_test.append(torch.from_numpy(arr_test).float())
                    y_new_test.append(torch.full((arr_test.shape[0],), num_old + i, dtype=torch.long))

    if len(X_new_test) > 0:
        X_new_test = torch.cat(X_new_test, 0)
        y_new_test = torch.cat(y_new_test, 0)
        new_test_loader = DataLoader(
            SignalDataset(X_new_test, y_new_test),
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
        )
        print(f"New test set: {X_new_test.shape[0]} samples")
    else:
        print("Warning: Using old test loader as fallback for new test set")
        new_test_loader = old_test_loader

    # === 记忆池（每旧类随机采样 memory_per_class；常驻 CPU） ===
    print("Building memory pool...")
    by_cls = {}
    for xb, yb in old_train_loader:
        for i in range(len(yb)):
            c = int(yb[i].item())
            by_cls.setdefault(c, []).append(xb[i].detach().cpu())

    mem_X_list, mem_y_list = [], []
    for c, xs in by_cls.items():
        n = len(xs)
        if n == 0:
            continue
        k = min(cfg.memory_per_class, n)
        idx = np.random.choice(n, k, replace=False)
        mem_X_list += [xs[j] for j in idx]
        mem_y_list += [c] * k

    if len(mem_X_list) > 0:
        mem_X = torch.stack(mem_X_list).contiguous()  # CPU 常驻
        mem_y = torch.tensor(mem_y_list, dtype=torch.long)  # CPU
        # 安全重映射（若标签不是 0..num_old-1）
        if mem_y.numel() > 0 and (int(mem_y.max()) >= num_old or int(mem_y.min()) < 0):
            cls2idx = {c: i for i, c in enumerate(cfg.known_classes)}
            mem_y = torch.tensor([cls2idx[int(c)] for c in mem_y.tolist()], dtype=torch.long)
    else:
        mem_X = torch.empty(0)
        mem_y = torch.empty(0, dtype=torch.long)

    print(f"[Data] 新类训练: {X_new.shape[0]} | 旧类记忆: {mem_X.shape[0]}")
    return new_train_loader, new_test_loader, old_test_loader, (mem_X, mem_y)


# ======================= 评估工具 =======================
@torch.no_grad()
def eval_old_only_before_train(model: OpenEmbed, test_old_loader: DataLoader, num_old: int, device: str):
    model.eval()
    tot, hit = 0, 0
    for x, y in test_old_loader:
        x = x.to(device)
        y = y.to(device)
        _, logits = model(x)
        pred = logits.argmax(1)
        hit += (pred == y).sum().item()
        tot += y.size(0)
    acc = hit / max(1, tot)
    return acc


@torch.no_grad()
def eval_teacher_upper_bound(cfg: Cfg):
    device = cfg.device
    num_old = len(cfg.known_classes)
    teacher = OpenEmbed(output=num_old).to(device)

    try:
        checkpoint = torch.load(cfg.teacher_model_path, map_location=device)

        # 检查checkpoint的结构
        if 'model' in checkpoint:
            # 如果包含'model'键，提取模型权重
            teacher_sd = checkpoint['model']
        elif 'state_dict' in checkpoint:
            # 如果包含'state_dict'键，提取模型权重
            teacher_sd = checkpoint['state_dict']
        else:
            # 否则假设整个checkpoint就是模型权重
            teacher_sd = checkpoint

        # 加载模型权重
        teacher.load_state_dict(teacher_sd)
        print(f"[Teacher] 成功加载预训练模型，输出维度: {num_old}")

    except Exception as e:
        print(f"Error loading teacher model: {e}")
        print("Teacher evaluation skipped.")
        return

    # 加载旧类测试数据
    X_old_test, y_old_test = load_all_data(cfg.data_root, cfg.known_classes)
    if len(X_old_test) == 0:
        print("No test data available for teacher evaluation")
        return

    test_loader = DataLoader(
        SignalDataset(X_old_test, y_old_test),
        batch_size=cfg.batch_size, shuffle=False
    )

    teacher.eval()
    tot, hit = 0, 0
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        _, logits = teacher(x)
        pred = logits.argmax(1)
        hit += (pred == y).sum().item()
        tot += y.size(0)

    acc = hit / max(1, tot)
    print(f"[Teacher Upper Bound] 旧类别准确率: {acc:.4f} ({acc * 100:.2f}%)")


@torch.no_grad()
def evaluate_on_loaders(model: OpenEmbed, test_old_loader: DataLoader, new_test_loader: DataLoader, num_old: int,
                        device: str):
    model.eval()

    def _eval(loader):
        tot, hit = 0, 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            _, logits = model(x)
            pred = logits.argmax(1)
            hit += (pred == y).sum().item()
            tot += y.size(0)
        return hit, tot

    old_hit, old_tot = _eval(test_old_loader)
    new_hit, new_tot = _eval(new_test_loader)

    acc_old = old_hit / max(1, old_tot)
    acc_new = new_hit / max(1, new_tot)
    acc_all = (old_hit + new_hit) / max(1, old_tot + new_tot)

    print(
        f"[Eval] 旧类别准确率: {acc_old * 100:.2f}% | 新类别准确率: {acc_new * 100:.2f}% | 总体准确率: {acc_all * 100:.2f}%")
    return {"old": acc_old, "new": acc_new, "overall": acc_all}


# ======================= A/GEM 梯度实用 =======================
def flatten_grads(params):
    vec = []
    device = next((p.device for p in params if p is not None), torch.device("cpu"))
    for p in params:
        if p.grad is None:
            vec.append(torch.zeros_like(p).view(-1))
        else:
            vec.append(p.grad.view(-1))
    return torch.cat(vec) if len(vec) else torch.tensor([], device=device)


def load_to_grads(params, grad_vec):
    ptr = 0
    for p in params:
        n = p.numel()
        if p.grad is None:
            p.grad = torch.zeros_like(p.data)
        p.grad.view(-1).copy_(grad_vec[ptr:ptr + n])
        ptr += n


@contextmanager
def stable_layers(model):
    # 冻结 BN running stats + 关闭 Dropout，仅在两次 backward 期间
    bns, dos = [], []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bns.append((m, m.training))
            m.eval()
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            dos.append((m, m.training))
            m.eval()
    try:
        yield
    finally:
        for m, was_train in bns + dos:
            m.train(was_train)


# ======================= GEM 投影（K=1 闭式；K>1 QP/PGD） =======================
def gem_project(g_new: torch.Tensor, mem_grads: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """
    g_new: [P]
    mem_grads: [K, P]  (K个约束向量)
    return: g_proj: [P]
    """
    if mem_grads.numel() == 0:
        return g_new

    # K=1：闭式解
    if mem_grads.dim() == 1 or mem_grads.shape[0] == 1:
        m = mem_grads.view(-1)
        den = torch.dot(m, m)
        if den.item() == 0:
            return g_new
        viol = torch.dot(g_new, m) - margin
        if viol.item() < 0:
            return g_new - (viol / den) * m
        return g_new

    # K>1：简化版本，避免 quadprog 依赖
    # 使用 PGD on dual
    M = mem_grads  # [K,P]
    P = M @ M.t()  # [K,K]
    q = -(M @ g_new)  # [K]

    # 估计 Lipschitz 常数 L = ||P||_2
    def power_iter(A, iters=20):
        v = torch.randn(A.shape[1], device=A.device)
        v = v / (v.norm() + 1e-12)
        for _ in range(iters):
            v = A @ (A.t() @ v)
            v = v / (v.norm() + 1e-12)
        Av = A @ v
        return (Av.norm() / (v.norm() + 1e-12)).item()

    L = max(1.0, power_iter(P))
    step = 1.0 / L
    v = torch.clamp_min(torch.zeros(P.shape[0], device=g_new.device), margin)

    for _ in range(100):
        grad_v = P @ v + q  # ∇
        v = v - step * grad_v
        v = torch.clamp_min(v, margin)

    x = M.t() @ v + g_new
    return x


# ======================= 训练（GEM） =======================
def train_gem(cfg: Cfg):
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = cfg.device
    num_old = len(cfg.known_classes)
    num_new = len(cfg.new_class_labels)
    total_classes = num_old + num_new

    print(f"[Training] 总类别数: {total_classes} (旧: {num_old}, 新: {num_new})")

    new_loader, new_test_loader, test_old_loader, (mem_X, mem_y) = build_loaders_and_memory(cfg)

    # 初始化可视化工具
    visualizer = Visualization(cfg)

    # 学生模型：加载教师骨干 + 扩展分类头
    model = OpenEmbed(output=total_classes).to(device)

    # 加载预训练权重
    try:
        checkpoint = torch.load(cfg.teacher_model_path, map_location=device)

        # 检查checkpoint的结构并提取模型权重
        if 'model' in checkpoint:
            teacher_sd = checkpoint['model']
            print("[Init] 从checkpoint中提取'model'权重")
        elif 'state_dict' in checkpoint:
            teacher_sd = checkpoint['state_dict']
            print("[Init] 从checkpoint中提取'state_dict'权重")
        else:
            teacher_sd = checkpoint
            print("[Init] 使用整个checkpoint作为模型权重")

        # 复制除了最后一层外的所有权重
        model_state = model.state_dict()
        loaded_keys = []

        for key in teacher_sd:
            if key in model_state:
                if model_state[key].shape == teacher_sd[key].shape:
                    model_state[key] = teacher_sd[key]
                    loaded_keys.append(key)
                else:
                    print(f"[Init] 跳过权重 {key}: 形状不匹配 {teacher_sd[key].shape} vs {model_state[key].shape}")
            else:
                print(f"[Init] 跳过未知键: {key}")

        model.load_state_dict(model_state)
        print(f"[Init] 成功加载 {len(loaded_keys)} 个权重参数")

    except Exception as e:
        print(f"Error loading teacher model: {e}")
        print("Training from scratch...")

    # 只训练最后一层（分类头）
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith("fc")  # 根据您的模型结构调整

    # 优化器 + 调度
    if cfg.optimizer_type.lower() == "sgd":
        optimizer = SGD((p for p in model.parameters() if p.requires_grad),
                        lr=cfg.lr, momentum=cfg.momentum, nesterov=cfg.nesterov,
                        weight_decay=cfg.weight_decay)
    else:
        optimizer = AdamW((p for p in model.parameters() if p.requires_grad),
                          lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    ce = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    # 训练前评估
    if cfg.show_teacher_upper_bound:
        eval_teacher_upper_bound(cfg)
    _ = eval_old_only_before_train(model, test_old_loader, num_old, device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    print("\n===== 开始 GEM 训练 =====")
    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        steps = len(new_loader)

        for step, (x_new, y_new) in enumerate(new_loader, start=1):
            x_new, y_new = x_new.to(device), y_new.to(device)
            K = max(1, int(cfg.gem_num_ref_batches))
            ref_grads = []

            with stable_layers(model):
                # --- 新梯度 ---
                optimizer.zero_grad(set_to_none=True)
                _, logits_new = model(x_new)
                loss_new = ce(logits_new, y_new)
                loss_new.backward()
                g_new = flatten_grads(trainable_params)

                # --- 参考梯度（旧样本）---
                if mem_X.numel() > 0:
                    for _ in range(K):
                        bs_ref = min(x_new.shape[0], mem_X.shape[0])
                        # 关键修复：索引必须在CPU上，因为mem_X在CPU上
                        idx = torch.randint(0, mem_X.shape[0], (bs_ref,), device="cpu")
                        x_old = mem_X[idx].to(device, non_blocking=True)
                        y_old = mem_y[idx].to(device, non_blocking=True)

                        optimizer.zero_grad(set_to_none=True)
                        _, logits_old = model(x_old)
                        loss_old = ce(logits_old, y_old)
                        loss_old.backward()
                        ref_grads.append(flatten_grads(trainable_params))

            # --- GEM 投影并更新 ---
            M = torch.stack(ref_grads, dim=0) if len(ref_grads) else torch.empty(0, device=device)
            g_proj = gem_project(g_new, M, margin=cfg.gem_margin) if M.numel() else g_new
            optimizer.zero_grad(set_to_none=True)
            load_to_grads(trainable_params, g_proj)
            optimizer.step()
            running_loss += float(loss_new.item())

            if step % cfg.print_every == 0:
                print(f'Epoch [{epoch + 1}/{cfg.epochs}], Step [{step}/{steps}], Loss: {loss_new.item():.4f}')

        avg_loss_epoch = running_loss / max(1, steps)
        current_lr = scheduler.get_last_lr()[0]

        # 记录训练损失
        visualizer.add_training_loss(epoch + 1, avg_loss_epoch)

        print(f"==> Epoch {epoch + 1}/{cfg.epochs} | Mean New Loss: {avg_loss_epoch:.4f} | LR: {current_lr:.6f}")
        scheduler.step()

        if cfg.do_eval_during_train and ((epoch + 1) % cfg.eval_every == 0):
            metrics = evaluate_on_loaders(model, test_old_loader, new_test_loader, num_old, device)
            visualizer.add_eval_metrics(epoch + 1, metrics)

    # 保存最终模型
    torch.save(model.state_dict(), cfg.final_model)
    print(f"[Save] 最终模型已保存到: {cfg.final_model}")

    # 最终评估
    final_metrics = evaluate_on_loaders(model, test_old_loader, new_test_loader, num_old, device)

    # 生成可视化图表
    print("\n" + "=" * 60)
    print("生成可视化图表...")
    print("=" * 60)

    # 1. 绘制训练损失曲线
    visualizer.plot_training_loss()

    # 2. 绘制准确率演化曲线
    visualizer.plot_accuracy_evolution()

    # 3. 绘制混淆矩阵
    confusion_mat = visualizer.plot_confusion_matrix(model, test_old_loader, new_test_loader, device)

    # 4. 打印最终报告
    visualizer.print_final_report(final_metrics, confusion_mat)

    return final_metrics


# ======================= 主入口 =======================
if __name__ == "__main__":
    cfg = Cfg()

    # 创建输出目录
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.vis_dir, exist_ok=True)

    print("=" * 80)
    print("USTC-TFC2016 增量学习实验")
    print(f"已知类别: {cfg.known_classes}")
    print(f"新增类别: {cfg.new_class_labels}")
    print(f"记忆样本数/类: {cfg.memory_per_class}")
    print(f"设备: {cfg.device}")
    print("=" * 80)

    try:
        metrics = train_gem(cfg)
        print(f"\n增加{cfg.new_class_labels}种未知恶意流量的准确率: {metrics['new']}")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback

        traceback.print_exc()