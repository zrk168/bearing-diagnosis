# -*- coding: utf-8 -*-
"""
CWRU 轴承故障诊断 | CNN-Transformer 双任务分类
修复版：解决数据堆叠报错 / 优化文件名解析 / 防震荡训练 / 完整可视化
"""
import os, re, time
import numpy as np
import scipy.io
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════
# 0. 配置 & 字体
# ══════════════════════════════════════════════════════════
def setup_font():
    for p in ['C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/msyh.ttc']:
        if os.path.exists(p):
            plt.rcParams['font.family'] = fm.FontProperties(fname=p).get_name()
            plt.rcParams['axes.unicode_minus'] = False
            return True
    plt.rcParams['font.family'] = 'DejaVu Sans'
    return False


HAS_CN = setup_font()

CFG = {
    'data_dir': r'E:\柱塞泵\CWRU轴承数据\cwru_data\0HP',
    'save_dir': r'./cwru_results',
    'seg_len': 1024,
    'batch_size': 64,
    'epochs': 100,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'patience': 15,
    'seed': 42,
    'label_smooth': 0.1,
}
os.makedirs(CFG['save_dir'], exist_ok=True)
torch.manual_seed(CFG['seed'])
np.random.seed(CFG['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG['seed'])
    torch.backends.cudnn.benchmark = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[设备] {DEVICE} | [数据路径] {CFG["data_dir"]}\n')


# ═════════════════════════════════════════════════════════
# 1. 数据加载 (修复堆叠报错 & 优化解析逻辑)
# ══════════════════════════════════════════════════════════
def parse_label(fname):
    """解析文件名，返回 (故障类型ID, 故障程度ID)"""
    base = os.path.splitext(fname)[0].lower()
    # 优先匹配故障关键词，避免误判
    if '_b' in base or 'ball' in base:
        f_type = 1
    elif '_ir' in base or 'inner' in base:
        f_type = 2
    elif '_or' in base or 'outer' in base:
        f_type = 3
    else:
        f_type = 0  # 默认为正常

    if '007' in base:
        sev = 1
    elif '014' in base:
        sev = 2
    elif '021' in base:
        sev = 3
    else:
        sev = 0
    return f_type, sev


def load_signal(fpath):
    mat = scipy.io.loadmat(fpath)
    for k in mat:
        if 'DE_time' in k and k[0] != '_': return mat[k].flatten().astype(np.float32)
    for k in mat:
        if isinstance(mat[k], np.ndarray) and mat[k].ndim == 2: return mat[k].flatten().astype(np.float32)
    raise ValueError(f"无法读取信号: {fpath}")


def segment_and_normalize(signal, seg_len):
    n_samples = len(signal) // seg_len
    if n_samples == 0: return np.zeros((0, seg_len), dtype=np.float32)
    segs = signal[:n_samples * seg_len].reshape(n_samples, seg_len)
    mu = segs.mean(axis=1, keepdims=True)
    std = segs.std(axis=1, keepdims=True) + 1e-8
    return (segs - mu) / std


def build_dataset(data_dir, cfg):
    if not os.path.exists(data_dir): raise FileNotFoundError(f"路径不存在: {data_dir}")
    X, Y_type, Y_sev = [], [], []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.mat'): continue
        t, s = parse_label(fname)
        sig = load_signal(os.path.join(data_dir, fname))
        segs = segment_and_normalize(sig, cfg['seg_len'])
        if len(segs) == 0: continue

        X.append(segs)
        Y_type.extend([t] * len(segs))
        Y_sev.extend([s] * len(segs))

    # 修复：使用 vstack 将不同行数的数组垂直拼接
    if len(X) == 0: raise ValueError("未加载到有效数据")
    X_combined = np.vstack(X)
    print(f"[数据] 共加载 {len(X_combined)} 个样本")
    return X_combined.astype(np.float32), np.array(Y_type, dtype=np.int64), np.array(Y_sev, dtype=np.int64)


# ══════════════════════════════════════════════════════════
# 2. Dataset & DataLoader
# ══════════════════════════════════════════════════════════
class BearingDS(Dataset):
    def __init__(self, X, yt, ys):
        self.X = torch.from_numpy(X).unsqueeze(1)
        self.yt = torch.from_numpy(yt)
        self.ys = torch.from_numpy(ys)

    def __len__(self): return len(self.X)

    def __getitem__(self, i): return self.X[i], self.yt[i], self.ys[i]


# ══════════════════════════════════════════════════════════
# 3. CNN-Transformer 模型
# ══════════════════════════════════════════════════════════
class CNNTransformer(nn.Module):
    def __init__(self, in_channels=1, d_model=64, nhead=4, num_layers=2, n_type=4, n_sev=4, dropout=0.15):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, 7, padding=3), nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU(), nn.AdaptiveAvgPool1d(16)
        )
        self.proj = nn.Linear(128, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 16, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                                               dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head_type = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                                       nn.Linear(d_model, n_type))
        self.head_sev = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                                      nn.Linear(d_model, n_sev))

    def forward(self, x):
        f = self.cnn(x).permute(0, 2, 1)
        f = self.proj(f) + self.pos_enc
        f = self.transformer(f).mean(dim=1)
        return self.head_type(f), self.head_sev(f)


# ══════════════════════════════════════════════════════════
# 4. 训练组件
# ══════════════════════════════════════════════════════════
class EarlyStopping:
    def __init__(self, patience, min_delta, save_path):
        self.patience, self.min_delta, self.path = patience, min_delta, save_path
        self.best, self.cnt, self.best_ep = np.inf, 0, 0

    def step(self, val_loss, model, epoch):
        if val_loss < self.best - self.min_delta:
            self.best, self.cnt, self.best_ep = val_loss, 0, epoch
            torch.save(model.state_dict(), self.path)
            return False
        self.cnt += 1
        return self.cnt >= self.patience


def run_epoch(model, loader, crit, opt, device, train=True):
    model.train() if train else model.eval()
    tot_loss = ct = cs = n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X, yt, ys in loader:
            X, yt, ys = X.to(device), yt.to(device), ys.to(device)
            pt, ps = model(X)
            loss = crit(pt, yt) + crit(ps, ys)
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
            tot_loss += loss.item() * len(X)
            ct += (pt.argmax(1) == yt).sum().item()
            cs += (ps.argmax(1) == ys).sum().item()
            n += len(X)
    return tot_loss / n, ct / n, cs / n


# ══════════════════════════════════════════════════════════
# 5. 可视化
# ══════════════════════════════════════════════════════════
def plot_curves(hist, save_dir):
    ep = range(1, len(hist['tl']) + 1)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].plot(ep, hist['tl'], label='Train');
    ax[0].plot(ep, hist['vl'], label='Val')
    ax[0].set_title('损失曲线');
    ax[0].set_xlabel('Epoch');
    ax[0].set_ylabel('Loss');
    ax[0].legend();
    ax[0].grid(alpha=0.3)
    ax[1].plot(ep, hist['tat'], label='Type');
    ax[1].plot(ep, hist['tas'], label='Severity')
    ax[1].set_title('训练集准确率');
    ax[1].set_xlabel('Epoch');
    ax[1].set_ylabel('Acc');
    ax[1].legend();
    ax[1].grid(alpha=0.3)
    ax[2].plot(ep, hist['vat'], label='Type');
    ax[2].plot(ep, hist['vas'], label='Severity')
    ax[2].set_title('验证集准确率');
    ax[2].set_xlabel('Epoch');
    ax[2].set_ylabel('Acc');
    ax[2].legend();
    ax[2].grid(alpha=0.3)
    plt.tight_layout();
    plt.savefig(os.path.join(save_dir, 'curves.png'), dpi=150);
    plt.close()


def plot_cm(y_true, y_pred, names, title, fpath):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.title(title);
    plt.colorbar()
    plt.xticks(range(len(names)), names, rotation=30, ha='right')
    plt.yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout();
    plt.savefig(fpath, dpi=150);
    plt.close()


# ══════════════════════════════════════════════════════════
# 6. 主流程
# ══════════════════════════════════════════════════════════
def main():
    print('[1/5] 加载数据...')
    X, yt, ys = build_dataset(CFG['data_dir'], CFG)

    print('[2/5] 划分数据集...')
    X_tr, X_vl, yt_tr, yt_vl, ys_tr, ys_vl = train_test_split(X, yt, ys, test_size=0.2, random_state=CFG['seed'],
                                                              stratify=yt)
    print(f'  训练集: {len(X_tr)} | 验证集: {len(X_vl)}')

    tr_loader = DataLoader(BearingDS(X_tr, yt_tr, ys_tr), batch_size=CFG['batch_size'], shuffle=True, num_workers=0)
    vl_loader = DataLoader(BearingDS(X_vl, yt_vl, ys_vl), batch_size=CFG['batch_size'], shuffle=False, num_workers=0)

    type_names = ['Normal', 'Ball', 'InnerRace', 'OuterRace']
    sev_names = ['Normal', '轻度(007)', '中度(014)', '重度(021)']

    print('[3/5] 初始化模型...')
    model = CNNTransformer(n_type=4, n_sev=4).to(DEVICE)
    total_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  参数量: {total_p:,} | 设备: {DEVICE}')

    crit = nn.CrossEntropyLoss(label_smoothing=CFG['label_smooth'])
    opt = optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG['epochs'], eta_min=1e-5)
    es = EarlyStopping(CFG['patience'], 1e-4, os.path.join(CFG['save_dir'], 'best_model.pth'))

    hist = {'tl': [], 'vl': [], 'tat': [], 'vat': [], 'tas': [], 'vas': [], 'lr': []}

    print('[4/5] 开始训练...\n')
    header = f"{'Epoch':>5} | {'LR':>9} | {'TrLoss':>8} | {'ValLoss':>8} | {'TrAcc_T':>7} | {'ValAcc_T':>8} | {'TrAcc_S':>7} | {'ValAcc_S':>8} | {'Time':>5}"
    print(header)
    print('-' * len(header))

    for ep in range(1, CFG['epochs'] + 1):
        t0 = time.time()
        tl, tat, tas = run_epoch(model, tr_loader, crit, opt, DEVICE, train=True)
        vl, vat, vas = run_epoch(model, vl_loader, crit, None, DEVICE, train=False)
        sched.step()
        cur_lr = opt.param_groups[0]['lr']
        elapsed = time.time() - t0

        hist['tl'].append(tl);
        hist['vl'].append(vl)
        hist['tat'].append(tat);
        hist['vat'].append(vat)
        hist['tas'].append(tas);
        hist['vas'].append(vas)
        hist['lr'].append(cur_lr)

        stop = es.step(vl, model, ep)
        marker = ' ★' if es.cnt == 0 else ''

        print(
            f"  {ep:5d} | {cur_lr:9.2e} | {tl:8.4f} | {vl:8.4f} | {tat * 100:6.2f}% | {vat * 100:7.2f}% | {tas * 100:6.2f}% | {vas * 100:7.2f}% | {elapsed:4.1f}s{marker}")

        if stop:
            print(f'\n[早停触发] 验证集损失连续 {CFG["patience"]} 轮未下降，停止训练。')
            print(f'[最优轮次] Epoch {es.best_ep} | 最优验证损失: {es.best:.4f}')
            break

    print('\n[5/5] 训练结束，生成报告...')
    model.load_state_dict(torch.load(es.path, map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        X_vl_t = torch.from_numpy(X_vl).unsqueeze(1).to(DEVICE)
        pt, ps = model(X_vl_t)
        pred_t = pt.argmax(1).cpu().numpy()
        pred_s = ps.argmax(1).cpu().numpy()

    print('\n' + '=' * 65)
    print('  📊 故障类型分类报告')
    print('=' * 65)
    print(classification_report(yt_vl, pred_t, target_names=type_names))

    print('\n' + '=' * 65)
    print('  📊 故障程度分类报告 (Severity)')
    print('=' * 65)
    print(classification_report(ys_vl, pred_s, target_names=sev_names))

    # ── 绘制混淆矩阵 ──────────────────────────────────────
    plot_cm(yt_vl, pred_t, type_names, '故障类型混淆矩阵', os.path.join(CFG['save_dir'], 'cm_type.png'))
    plot_cm(ys_vl, pred_s, sev_names, '故障程度混淆矩阵', os.path.join(CFG['save_dir'], 'cm_severity.png'))

    # ── 绘制训练曲线 ──────────────────────────────────────
    plot_curves(hist, CFG['save_dir'])

    print('\n' + '█' * 65)
    print('  ✅ 全部完成！结果已保存至:')
    print(f'  📁 {os.path.abspath(CFG["save_dir"])}')
    print('   curves.png          (损失与准确率曲线)')
    print('   cm_type.png         (故障类型混淆矩阵)')
    print('   cm_severity.png     (故障程度混淆矩阵)')
    print('   best_model.pth      (最优模型权重)')
    print('█' * 65)


if __name__ == '__main__':
    main()

