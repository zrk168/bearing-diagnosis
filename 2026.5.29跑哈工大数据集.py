import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import warnings

warnings.filterwarnings('ignore')

# ================= 配置参数 =================
# 自动获取脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 修正为实际数据相对路径（根据你的目录结构）
DATA_DIR = os.path.join(BASE_DIR, "HIT-bearing-dataset", "原始数据")

# 打印解析后的绝对路径，方便核对
print(f"📂 脚本目录: {BASE_DIR}")
print(f"📂 数据路径: {DATA_DIR}")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"🚀 已启用GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 15
LR_INIT = 1e-3
NUM_CLASSES = 5
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ================= 数据集与标签解析 =================
def parse_label_from_filename(filename):
    """根据文件名自动解析层级标签"""
    base = os.path.basename(filename).lower()
    if 'data1' in base or 'data2' in base:
        return 0  # 正常
    elif 'data3' in base or 'data4' in base:
        if '007' in base:
            return 1  # 内圈轻度
        elif '014' in base:
            return 2  # 内圈中度
        elif '021' in base:
            return 3  # 内圈重度
        else:
            return 1
    elif 'data5' in base:
        return 4  # 外圈（默认单类，若外圈也有程度可在此扩展）
    return 0


class BearingDataset(Dataset):
    def __init__(self, data_dir):
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"❌ 数据路径不存在: {data_dir}\n请确保与脚本同级或修改 DATA_DIR")

        self.samples, self.labels = [], []
        npy_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        if not npy_files:
            raise FileNotFoundError(f"❌ 未在 {data_dir} 中找到 .npy 文件")

        print(f"📂 找到 {len(npy_files)} 个文件，正在加载与预处理...")
        for fpath in tqdm(npy_files, desc="Loading"):
            try:
                data = np.load(fpath, allow_pickle=True)
                # 兼容从.mat转换来的字典结构
                if isinstance(data, dict):
                    data = list(data.values())[0]
                # data shape: [num_segments, 8, 20480]
                label = parse_label_from_filename(fpath)
                for seg in data:
                    # 🛡️ 强制清洗原始数据中的 NaN/Inf/空值
                    seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
                    # 标准化（分母加 1e-8 防除零）
                    seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-8)
                    self.samples.append(seg.astype(np.float32))
                    self.labels.append(label)

            except Exception as e:
                print(f"⚠️ 跳过 {fpath}: {e}")

        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)
        print(
            f"✅ 加载完成 | 样本总数: {len(self.samples)} | 类别分布: {dict(zip(*np.unique(self.labels, return_counts=True)))}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx]), torch.tensor(self.labels[idx], dtype=torch.long)


# ================= CNN-Transformer 模型 =================
class CNNTransformer(nn.Module):
    def __init__(self, in_channels=8, d_model=64, nhead=4, num_layers=3, num_classes=5, dropout=0.15):
        super().__init__()
        # 1D CNN 降维与局部特征提取
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(64)
        )

        self.proj = nn.Linear(128, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)  # [B, 128, 64]
        x = x.permute(0, 2, 1)  # [B, 64, 128]
        x = self.proj(x)  # [B, 64, d_model]
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 65, d_model]
        x = self.transformer(x)
        x = x[:, 0, :]  # 取CLS Token
        return self.classifier(x)


# ================= 早停机制 =================
class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=0.001):
        self.patience, self.min_delta = patience, min_delta
        self.counter, self.best_loss, self.early_stop = 0, None, False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter = 0
        return self.early_stop


# ================= 训练与验证循环 =================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪防震荡
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += inputs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += inputs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


# ================= 可视化与层级诊断映射 =================
def plot_curves(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Val Loss', marker='s')
    plt.title('Loss Curve');
    plt.xlabel('Epoch');
    plt.ylabel('Loss')
    plt.legend();
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc', marker='o')
    plt.plot(val_accs, label='Val Acc', marker='s')
    plt.title('Accuracy Curve');
    plt.xlabel('Epoch');
    plt.ylabel('Accuracy')
    plt.legend();
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    plt.show()


def plot_confusion_matrix(all_preds, all_labels, class_names):
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    plt.title('Validation Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()


def hierarchical_decode(pred_idx):
    """将模型输出映射为你要求的层级诊断逻辑"""
    mapping = {
        0: "状态: 正常",
        1: "状态: 故障 | 类型: 内圈 | 程度: 轻度(007)",
        2: "状态: 故障 | 类型: 内圈 | 程度: 中度(014)",
        3: "状态: 故障 | 类型: 内圈 | 程度: 重度(021)",
        4: "状态: 故障 | 类型: 外圈"
    }
    return mapping.get(pred_idx, "未知")


# ================= 主流程 =================
def main():
    print("🚀 开始加载数据...")
    dataset = BearingDataset(DATA_DIR)

    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=SEED,
                                          stratify=dataset.labels)
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False,
                            pin_memory=True)

    print("🔧 初始化模型...")
    model = CNNTransformer(num_classes=NUM_CLASSES).to(DEVICE)

    # 类别权重缓解不平衡
    class_counts = np.bincount(dataset.labels)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    class_weights = (class_weights / class_weights.sum()) * NUM_CLASSES
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    optimizer = optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopper = EarlyStopping()

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc = 0.0

    print(f"📈 开始训练 (Device: {DEVICE})")
    print("-" * 80)
    for epoch in range(1, EPOCHS + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        v_loss, v_acc, v_preds, v_labels = validate(model, val_loader, criterion, DEVICE)

        train_losses.append(t_loss);
        train_accs.append(t_acc)
        val_losses.append(v_loss);
        val_accs.append(v_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {t_loss:.4f} | Acc: {t_acc:.4f} | "
              f"Val Loss: {v_loss:.4f} | Acc: {v_acc:.4f} | "
              f"LR: {current_lr:.2e}")

        scheduler.step(v_loss)
        if early_stopper(v_loss, model):
            print(f"\n🛑 早停触发于 Epoch {epoch}。最佳验证损失: {early_stopper.best_loss:.4f}")
            break

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), 'best_cnn_transformer.pth')

    print(f"✅ 训练完成！最佳验证准确率: {best_val_acc:.4f}")

    # 加载最优权重评估
    model.load_state_dict(early_stopper.best_model_wts or torch.load('best_cnn_transformer.pth', map_location=DEVICE))
    _, _, all_preds, all_labels = validate(model, val_loader, criterion, DEVICE)

    # 打印层级诊断示例
    print("\n📋 层级诊断逻辑映射示例:")
    for i in range(NUM_CLASSES):
        print(f"  类别 {i} -> {hierarchical_decode(i)}")

    plot_curves(train_losses, val_losses, train_accs, val_accs)
    class_names = ['正常', '内圈-轻', '内圈-中', '内圈-重', '外圈']
    plot_confusion_matrix(all_preds, all_labels, class_names)
    print("📊 图表已保存至当前目录。")


if __name__ == '__main__':
    main()
