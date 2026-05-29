# -*- coding: utf-8 -*-
"""
学习率搜索：分析1-40轮准确率曲线的一阶/二阶导数
"""

import os, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from scipy import stats
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ================================================================
# 配置
# ================================================================
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT       = r'E:\BaiduNetdiskDownload\CRWU'
SAMPLE_LEN      = 1024
SAMPLES_PER_CLS = 500
BATCH_SIZE      = 64
SEED            = 42
NUM_TYPE        = 4
NUM_SEV         = 4
SEARCH_EPOCHS   = 40   # 只跑前40轮

# 待搜索的学习率列表
LR_LIST = [1e-5, 3e-5, 1e-4, 3e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"使用设备: {DEVICE}")

TYPE_NAMES = ['正常(Normal)', '内圈故障(IR)', '外圈故障(OR)', '滚动体(Ball)']
SEV_NAMES  = ['正常(0)', '轻微(1)', '中度(2)', '重度(3)']

# ================================================================
# 以下函数与原脚本完全相同
# ================================================================
def get_label(filepath):
    rel = filepath.replace(DATA_ROOT, '').replace('\\', '/').lower()
    if 'normal baseline' in rel: return (0, 0)
    if '12k drive end' not in rel: return None
    if '0007' in rel: sev = 1
    elif '0014' in rel: sev = 2
    elif '0021' in rel: sev = 3
    else: return None
    if '/inner race/' in rel: return (1, sev)
    if '/outer race/' in rel: return (2, sev)
    if '/ball/' in rel:       return (3, sev)
    return None

def read_mat(fpath):
    try:
        mat = loadmat(fpath)
        de_keys = [k for k in mat if 'de' in k.lower() and not k.startswith('_')]
        if de_keys:
            arr = mat[de_keys[0]].flatten().astype(np.float64)
            if len(arr) >= SAMPLE_LEN: return arr
        arrs = [mat[k].flatten() for k in mat
                if not k.startswith('_') and isinstance(mat[k], np.ndarray)
                and mat[k].size >= SAMPLE_LEN]
        return max(arrs, key=len).astype(np.float64) if arrs else None
    except: return None

def extract_features(sig):
    x, eps = sig.astype(np.float64), 1e-12
    max_val = np.max(x);              max_abs  = np.max(np.abs(x))
    min_val = np.min(x);              mean_val = np.mean(x)
    ptp     = max_val - min_val;      abs_mean = np.mean(np.abs(x))
    rms     = np.sqrt(np.mean(x**2)); sra      = np.mean(np.sqrt(np.abs(x)))**2
    std_val = np.std(x, ddof=1)
    kurt    = stats.kurtosis(x, fisher=False)
    skew_val= stats.skew(x)
    clf     = max_abs / (sra + eps);  wf   = rms / (abs_mean + eps)
    impf    = max_abs / (abs_mean + eps); crf = max_abs / (rms + eps)
    return np.array([max_val, max_abs, min_val, mean_val, ptp,
                     abs_mean, rms, sra, std_val, kurt, skew_val,
                     clf, wf, impf, crf], dtype=np.float32)

def load_data():
    from collections import defaultdict
    groups = defaultdict(list)
    for root, _, files in os.walk(DATA_ROOT):
        for f in files:
            if not f.lower().endswith('.mat'): continue
            fp = os.path.join(root, f)
            tag = get_label(fp)
            if tag is not None: groups[tag].append(fp)
    X_list, yt_list, ys_list = [], [], []
    for (t, s) in sorted(groups.keys()):
        sigs = [sig for fp in groups[(t, s)] for sig in [read_mat(fp)] if sig is not None]
        if not sigs: continue
        full   = np.concatenate(sigs)
        starts = np.arange(0, len(full) - SAMPLE_LEN, SAMPLE_LEN)
        rng    = np.random.default_rng(SEED + t * 10 + s)
        rng.shuffle(starts)
        starts = starts[:SAMPLES_PER_CLS]
        for st in starts:
            X_list.append(extract_features(full[st: st + SAMPLE_LEN]))
            yt_list.append(t); ys_list.append(s)
    return (np.array(X_list, dtype=np.float32),
            np.array(yt_list, dtype=np.int64),
            np.array(ys_list, dtype=np.int64))

def split_dataset(X, y_type, y_sev, ratio=0.8):
    rng = np.random.default_rng(SEED)
    tr_idx, te_idx = [], []
    for t in range(NUM_TYPE):
        for s in range(NUM_SEV):
            idx = np.where((y_type == t) & (y_sev == s))[0]
            if len(idx) == 0: continue
            idx  = rng.permutation(idx)
            n_tr = int(len(idx) * ratio)
            tr_idx.extend(idx[:n_tr].tolist())
            te_idx.extend(idx[n_tr:].tolist())
    return (np.array(rng.permutation(tr_idx)),
            np.array(rng.permutation(te_idx)))

class BearingDS(Dataset):
    def __init__(self, X, yt, ys, mean=None, std=None):
        self.mean = mean if mean is not None else X.mean(0)
        self.std  = std  if std  is not None else X.std(0) + 1e-8
        Xn = (X - self.mean) / self.std
        self.X  = torch.tensor(Xn, dtype=torch.float32).unsqueeze(1)
        self.yt = torch.tensor(yt, dtype=torch.long)
        self.ys = torch.tensor(ys, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.yt[i], self.ys[i]

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act  = nn.GELU()
    def forward(self, x):
        return self.act(self.net(x) + self.skip(x))

class CNNTransformer(nn.Module):
    def __init__(self, seq_len=15, d_model=64, nhead=4, num_layers=2, ff_dim=128, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            ResBlock(1, 16, dropout), ResBlock(16, 32, dropout), ResBlock(32, d_model, dropout))
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
                                         dropout=dropout, batch_first=True, norm_first=True)
        self.trans = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.norm  = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)
        def head(n):
            return nn.Sequential(nn.Linear(d_model, 32), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(32, n))
        self.head_type = head(NUM_TYPE)
        self.head_sev  = head(NUM_SEV)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
    def forward(self, x):
        f = self.cnn(x).permute(0, 2, 1) + self.pos
        f = self.norm(self.trans(f).mean(1))
        return self.head_type(self.drop(f)), self.head_sev(self.drop(f))

def run_epoch(model, loader, criterion, optimizer=None, train=True, scheduler=None):
    model.train() if train else model.eval()
    tot_loss = cor_t = cor_s = total = 0
    with torch.set_grad_enabled(train):
        for Xb, yt, ys in loader:
            Xb, yt, ys = Xb.to(DEVICE), yt.to(DEVICE), ys.to(DEVICE)
            out_t, out_s = model(Xb)
            loss = criterion(out_t, yt) + criterion(out_s, ys)
            if train:
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None: scheduler.step()
            tot_loss += loss.item() * len(Xb)
            cor_t    += (out_t.argmax(1) == yt).sum().item()
            cor_s    += (out_s.argmax(1) == ys).sum().item()
            total    += len(Xb)
    return tot_loss / total, cor_t / total * 100, cor_s / total * 100

# ================================================================
# 对一条曲线求导数统计量
# 用均值衡量"整体趋势"，用标准差衡量"波动程度"
# ================================================================
def deriv_stats(curve):
    arr = np.array(curve)
    d1  = np.gradient(arr)   # 一阶导
    d2  = np.gradient(d1)    # 二阶导
    return d1.mean(), d1.std(), d2.mean(), d2.std()

# ================================================================
# 主流程
# ================================================================
def main():
    print("\n=== 加载数据 ===")
    X, y_type, y_sev = load_data()
    tr_idx, te_idx   = split_dataset(X, y_type, y_sev)

    tr_ds = BearingDS(X[tr_idx], y_type[tr_idx], y_sev[tr_idx])
    te_ds = BearingDS(X[te_idx], y_type[te_idx], y_sev[te_idx],
                      mean=tr_ds.mean, std=tr_ds.std)
    tr_loader = DataLoader(tr_ds, BATCH_SIZE, shuffle=True,  num_workers=0)
    te_loader = DataLoader(te_ds, BATCH_SIZE, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()

    # 结果存储: 每个lr → 4条曲线的导数统计
    results = {}  # lr: {tr_at, te_at, tr_as, te_as} 各含 (d1_mean, d1_std, d2_mean, d2_std)

    for lr in LR_LIST:
        print(f"\n--- LR={lr} ---")
        torch.manual_seed(SEED)
        model     = CNNTransformer().to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr,
            epochs=SEARCH_EPOCHS, steps_per_epoch=len(tr_loader),
            pct_start=0.2, anneal_strategy='cos', div_factor=20, final_div_factor=1e4)

        curves = {k: [] for k in ['tr_at', 'te_at', 'tr_as', 'te_as']}
        for ep in range(1, SEARCH_EPOCHS + 1):
            _, tr_at, tr_as = run_epoch(model, tr_loader, criterion, optimizer, True, scheduler)
            _, te_at, te_as = run_epoch(model, te_loader, criterion, train=False)
            curves['tr_at'].append(tr_at)
            curves['te_at'].append(te_at)
            curves['tr_as'].append(tr_as)
            curves['te_as'].append(te_as)
            if ep % 10 == 0:
                print(f"  Ep{ep:2d} | 类型 训{tr_at:.1f}/测{te_at:.1f} | 程度 训{tr_as:.1f}/测{te_as:.1f}")

        results[lr] = {k: deriv_stats(v) for k, v in curves.items()}

    # ================================================================
    # 绘图
    # ================================================================
    lrs     = LR_LIST
    lr_ticks = [f"{lr:.0e}" for lr in lrs]
    x       = np.arange(len(lrs))

    def get_stat(key, stat_idx):
        return [results[lr][key][stat_idx] for lr in lrs]

    # --- 图1：一阶导数均值（反映上升速度） ---
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(x, get_stat('tr_at', 0), 'b-o',  label='训练集-故障类型')
    ax1.plot(x, get_stat('te_at', 0), 'b--s', label='测试集-故障类型')
    ax2.plot(x, get_stat('tr_as', 0), 'r-o',  label='训练集-故障程度')
    ax2.plot(x, get_stat('te_as', 0), 'r--s', label='测试集-故障程度')
    ax1.set_xlabel('学习率'); ax1.set_ylabel('故障类型准确率一阶导均值 (%/epoch)', color='b')
    ax2.set_ylabel('故障程度准确率一阶导均值 (%/epoch)', color='r')
    ax1.set_xticks(x); ax1.set_xticklabels(lr_ticks, rotation=30)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax1.set_title('不同学习率下 1-40轮 准确率曲线 一阶导数均值')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('lr_search_d1.png', dpi=150); plt.show()
    print("已保存 lr_search_d1.png")

    # --- 图2：二阶导数均值（反映加速度/波动） ---
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(x, get_stat('tr_at', 2), 'b-o',  label='训练集-故障类型')
    ax1.plot(x, get_stat('te_at', 2), 'b--s', label='测试集-故障类型')
    ax2.plot(x, get_stat('tr_as', 2), 'r-o',  label='训练集-故障程度')
    ax2.plot(x, get_stat('te_as', 2), 'r--s', label='测试集-故障程度')
    ax1.set_xlabel('学习率'); ax1.set_ylabel('故障类型准确率二阶导均值 (%/epoch²)', color='b')
    ax2.set_ylabel('故障程度准确率二阶导均值 (%/epoch²)', color='r')
    ax1.set_xticks(x); ax1.set_xticklabels(lr_ticks, rotation=30)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax1.set_title('不同学习率下 1-40轮 准确率曲线 二阶导数均值')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('lr_search_d2.png', dpi=150); plt.show()
    print("已保存 lr_search_d2.png")

    # 打印汇总表
    print(f"\n{'LR':>8} | {'类型-d1均值':>10} {'类型-d1std':>10} | {'程度-d1均值':>10} {'程度-d1std':>10}")
    print("-" * 60)
    for lr in lrs:
        r = results[lr]
        print(f"{lr:8.0e} | {r['te_at'][0]:10.3f} {r['te_at'][1]:10.3f} | "
              f"{r['te_as'][0]:10.3f} {r['te_as'][1]:10.3f}")

if __name__ == '__main__':
    main()
