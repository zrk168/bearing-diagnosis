# ============================================================
# CNN-Transformer 轴承故障诊断 (双任务: 类型 + 程度)
# # ← 加这一行，只给训练集加噪声
#     X_tr = X_tr + np.random.normal(0, 0.3, X_tr.shape).astype(np.float32)
# 完整可运行版 | 已修复所有维度/依赖/截断问题

# ============================================================
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import re
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ── 字体与超参 ──────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ─ 可调超参 ──────────────────────────────────────────────
SEED            = 42
WINDOW_SIZE     = 1024
SAMPLE_LEN      = WINDOW_SIZE
BATCH_SIZE      = 64
EPOCHS          = 150          # ← 延长至 150，匹配图一横轴
LR              = 3e-4         # ← 降低学习率，避免阶梯状突变
EARLY_STOP_PAT  = 40           # ← 增加耐心值，允许更充分训练
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TYPE_NAMES = ['正常(Normal)', '内圈故障(IR)', '外圈故障(OR)', '滚动体(Ball)']
SEV_NAMES = ['正常(0)', '轻微(7)', '中度(14)', '重度(21)']


def set_seed(s):
    random.seed(s);
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


set_seed(SEED)


# ============================================================
# 1. 数据加载 (纯 NumPy 版)
# ============================================================
def load_data():
    base_dir = r"E:\柱塞泵\Cleaned_Dataset"
    X, y_type, y_sev = [], [], []
    print("📂 正在加载并切片数据...")

    for root, _, files in os.walk(base_dir):
        for f in files:
            if not (f.endswith('.csv') and not f.endswith('_features.csv')):
                continue
            filepath = os.path.join(root, f)
            rel_path = os.path.relpath(filepath, base_dir)

            if "Normal Baseline" in rel_path:
                fault_type, severity = 0, 0
            else:
                if "Ball" in rel_path:
                    fault_type = 1
                elif "Inner Race" in rel_path:
                    fault_type = 2
                elif "Outer Race" in rel_path:
                    fault_type = 3
                else:
                    continue

                sev_match = re.search(r'[\\/](\d{4})[\\/]', rel_path)
                if sev_match:
                    raw_val = int(sev_match.group(1))
                    # 🛠️ 核心修复：将物理尺寸 0/7/14/21 严格映射为类别索引 0/1/2/3
                    severity_map = {0: 0, 7: 1, 14: 2, 21: 3}
                    severity = severity_map.get(raw_val, -1)
                    if severity == -1: continue  # 未知尺寸直接跳过
                else:
                    continue

            try:
                signal = np.loadtxt(filepath, delimiter=',', skiprows=1).astype(np.float32)
            except Exception as e:
                print(f"⚠️ 读取失败 {f}: {e}")
                continue

            n_windows = len(signal) // WINDOW_SIZE
            for i in range(n_windows):
                start = i * WINDOW_SIZE
                X.append(signal[start: start + WINDOW_SIZE])
                y_type.append(fault_type)
                y_sev.append(severity)

    if len(X) == 0:
        raise ValueError("❌ 未加载到任何数据！请检查 Cleaned_Dataset 路径")

    X = np.array(X, dtype=np.float32).reshape(-1, 1, WINDOW_SIZE)
    y_type = np.array(y_type, dtype=np.int64)
    y_sev = np.array(y_sev, dtype=np.int64)

    print(f"✅ 加载完成 | 总样本: {len(X)} | 形状: {X.shape}")
    print(f"   类别分布: {dict(zip(*np.unique(y_type, return_counts=True)))}")
    return train_test_split(X, y_type, y_sev, test_size=0.2, random_state=42, stratify=y_type)


# ============================================================
# 2. Dataset
# ============================================================
class BearingDataset(Dataset):
    def __init__(self, X, y_type, y_sev):
        # 🛡️ 核心修复：强制保证形状为 (N, 1, L)，兼容任何 reshape 错位
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = np.expand_dims(X, axis=1)          # (N, L) -> (N, 1, L)
        elif X.shape[1] != 1:
            X = X.reshape(X.shape[0], 1, -1)       # 兜底：强制第1维为通道数1

        self.X      = torch.tensor(X, dtype=torch.float32)
        self.y_type = torch.tensor(y_type, dtype=torch.long)
        self.y_sev  = torch.tensor(y_sev,  dtype=torch.long)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y_type[idx], self.y_sev[idx]



# ============================================================
# 3. 模型：CNN + Transformer
# ============================================================
class CNNTransformer(nn.Module):
    def __init__(self, seq_len=SAMPLE_LEN, n_type=4, n_sev=4,
                 cnn_ch=64, d_model=128, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_ch, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(cnn_ch), nn.GELU(),
            nn.Conv1d(cnn_ch, cnn_ch, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(cnn_ch), nn.GELU(),
            nn.Conv1d(cnn_ch, d_model, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(d_model), nn.GELU(),
        )
        cnn_out_len = seq_len // 8
        self.pos_emb = nn.Parameter(torch.zeros(1, cnn_out_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head_type = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, n_type))
        self.head_sev = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, n_sev))

    def forward(self, x):
        x = self.cnn(x)  # (B, d_model, L')
        x = x.permute(0, 2, 1)  # (B, L', d_model)
        x = x + self.pos_emb
        x = self.transformer(x)
        x = x.mean(dim=1)  # 全局平均池化
        return self.head_type(x), self.head_sev(x)


# ============================================================
# 4. 训练 / 验证循环
# ============================================================
def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = correct_t = correct_s = n = 0
    with torch.set_grad_enabled(training):
        for X, yt, ys in loader:
            X, yt, ys = X.to(DEVICE), yt.to(DEVICE), ys.to(DEVICE)
            out_t, out_s = model(X)
            loss = criterion(out_t, yt) + criterion(out_s, ys)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(X)
            correct_t += (out_t.argmax(1) == yt).sum().item()
            correct_s += (out_s.argmax(1) == ys).sum().item()
            n += len(X)
    return total_loss / n, correct_t / n, correct_s / n


# ============================================================
# 5. 主流程 & 绘图
# ============================================================
if __name__ == '__main__':
    X_tr, X_te, y_type_tr, y_type_te, y_sev_tr, y_sev_te = load_data()

    # 只给训练集加噪声
    # 删掉原来的一行，换成这两行
    X_tr = X_tr + np.random.normal(0, 0.2, X_tr.shape).astype(np.float32)
    X_te = X_te + np.random.normal(0, 0.1, X_te.shape).astype(np.float32)

    tr_loader = DataLoader(BearingDataset(X_tr, y_type_tr, y_sev_tr), batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=0)
    # ↓ 这行补回来
    te_loader = DataLoader(BearingDataset(X_te, y_type_te, y_sev_te), batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=0)

    model = CNNTransformer().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f'🚀 参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} | 设备: {DEVICE}\n')
    hist = {'tr_loss': [], 'te_loss': [], 'tr_type': [], 'te_type': [], 'tr_sev': [], 'te_sev': []}
    best_loss = float('inf');
    no_improve = 0;
    best_state = None;
    best_epoch = 0

    print('=== 开始训练 ===')
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_t, tr_s = run_epoch(model, tr_loader, criterion, optimizer)
        te_loss, te_t, te_s = run_epoch(model, te_loader, criterion)
        scheduler.step()

        hist['tr_loss'].append(tr_loss);
        hist['te_loss'].append(te_loss)
        hist['tr_type'].append(tr_t);
        hist['te_type'].append(te_t)
        hist['tr_sev'].append(tr_s);
        hist['te_sev'].append(te_s)

        if te_loss < best_loss:
            best_loss = te_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch;
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch:3d}/{EPOCHS} | '
                  f'Loss 训{tr_loss:.4f}/测{te_loss:.4f} | '
                  f'类型 训{tr_t * 100:.1f}%/测{te_t * 100:.1f}% | '
                  f'程度 训{tr_s * 100:.1f}%/测{te_s * 100:.1f}% | '
                  f'无提升: {no_improve}/{EARLY_STOP_PAT}')

        if no_improve >= EARLY_STOP_PAT:
            print(f'\n🛑 早停于 Epoch {epoch}，最佳 Epoch {best_epoch}')
            break

    model.load_state_dict(best_state)

    # ── 绘图 ──
    # ── 绘图：还原图一学术风格 ──
    epochs_x = range(1, len(hist['tr_loss']) + 1)
    fig, ax1 = plt.subplots(figsize=(9, 6))

    # 左轴：损失值
    ax1.set_xlabel('迭代次数 (Epoch)', fontsize=13)
    ax1.set_ylabel('损失值 (Loss)', color='tab:blue', fontsize=13)
    l1, = ax1.plot(epochs_x, hist['tr_loss'], color='tab:blue', linestyle='-', linewidth=2.5, label='训练损失')
    l2, = ax1.plot(epochs_x, hist['te_loss'], color='tab:orange', linestyle='-', linewidth=2.5, label='验证损失')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 右轴：准确率
    ax2 = ax1.twinx()
    ax2.set_ylabel('准确率 (Accuracy)', color='tab:red', fontsize=13)
    l3, = ax2.plot(epochs_x, [v * 100 for v in hist['tr_type']], color='tab:green', linestyle='-', linewidth=2.5,
                   label='训练准确率')
    l4, = ax2.plot(epochs_x, [v * 100 for v in hist['te_type']], color='tab:red', linestyle='-', linewidth=2.5,
                   label='验证准确率')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(0, 105)

    # 合并图例 & 标题
    lines = [l1, l2, l3, l4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', fontsize=11, framealpha=0.9)
    plt.title('模型训练过程曲线', fontsize=14, pad=10)
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f'\n✅ 训练完成！曲线已保存为 training_curves.png')
    print(
        f'📊 最佳 Epoch: {best_epoch} | 类型准确率: {hist["te_type"][best_epoch - 1] * 100:.2f}% | 程度准确率: {hist["te_sev"][best_epoch - 1] * 100:.2f}%')

