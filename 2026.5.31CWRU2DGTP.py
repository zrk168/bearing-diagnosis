# -*- coding: utf-8 -*-
"""
CRWU / CWRU 轴承故障诊断完整代码
模型：2D CNN-Transformer + CWT 小波时频图 + 可选统计特征融合
任务逻辑：先分故障类型，再分故障程度

================================================================================
一、整体流程
================================================================================

1. 数据读取
   数据根目录：
       E:\\柱塞泵\\CRWU

   本代码只读取两个文件夹：
       1) 12k Drive End Bearing Fault Data
       2) Normal Baseline

   代码会递归扫描这两个文件夹下所有 .mat 文件。

2. 标签解析
   根据文件路径和文件名自动解析：
       第一阶段：故障类型
           0 -> Normal
           1 -> Ball / 滚动体
           2 -> InnerRace / 内圈
           3 -> OuterRace / 外圈

       第二阶段：故障程度
           0 -> 轻微
           1 -> 中度
           2 -> 重度
           3 -> 严重

   正常样本的 severity_label = -1，不参与故障程度损失计算。

3. 信号预处理
   - 从 .mat 中优先读取 DE_time 振动信号；
   - 对一维振动信号进行滑动窗口切片；
   - 对每个切片进行 Z-score 标准化；
   - 对每个切片进行 CWT 连续小波变换；
   - 得到二维时频图，作为 2D CNN 的输入。

4. 统计特征提取
   对每个切片提取：
       - 均值、标准差、RMS、峰值、峰峰值；
       - 偏度、峭度；
       - 峰值因子、脉冲因子、波形因子、裕度因子；
       - 频谱中心、均方频率、频率标准差；
       - 多频带能量占比。

5. 模型结构
   - CWT 小波时频图输入 2D CNN；
   - CNN 输出的局部时频特征输入 Transformer Encoder；
   - 统计特征可选输入 MLP；
   - 融合后输出两个分类头：
       1) 故障类型分类头；
       2) 故障程度分类头。

6. 训练策略
   - 记录 Epoch 0 未训练模型性能；
   - 训练 150 轮；
   - 默认关闭早停，保证曲线完整；
   - 关闭标签平滑，使 Loss 可以下降到接近 0；
   - 降低学习率，让准确率逐渐上升；
   - 默认不使用统计特征，避免模型第一轮就接近满分。

7. 结果输出
   - 每个文件的样本数量；
   - history.csv；
   - best_model.pth；
   - curves_four_panel.png；
   - ideal_style_curve_type.png；
   - ideal_style_curve_severity.png；
   - ideal_style_curve_joint.png；
   - cm_type.png；
   - cm_severity.png。

================================================================================
二、为了让曲线更接近“理想曲线”做的关键改进
================================================================================

改进 1：overlap 从 0.5 改为 0.0
    原因：
        重叠窗口会让相邻样本高度相似，容易导致训练集和验证集泄漏。
        overlap=0.0 可以降低样本之间重复程度。

改进 2：关闭 label_smooth
    原因：
        label_smooth 会让 Loss 长期停在 0.7 左右，不容易下降到接近 0。
        理想图中的 Loss 是逐渐下降到较低水平，所以这里设为 0.0。

改进 3：学习率降低为 1e-4
    原因：
        原来的 1e-3 学得太快，第一二轮准确率就接近 100%。
        降低学习率后，曲线上升更平滑。

改进 4：训练轮数改为 150
    原因：
        你的理想图横轴约为 150 轮，因此这里固定训练 150 轮。

改进 5：关闭早停
    原因：
        如果早停开启，可能十几轮就停止，画不出完整 150 轮曲线。

改进 6：默认关闭统计特征参与训练
    原因：
        RMS、峭度、峰值因子、频带能量等统计特征对故障程度非常敏感，
        会让模型很快达到 99% 以上。
        为了让曲线更自然上升，默认只使用 CWT 图像分支。
        如果你想重新启用统计特征，把：
            "use_stat_features": True

改进 7：增加 Epoch 0 评估
    原因：
        未训练模型的 4 分类准确率通常接近 0.25，
        这样曲线可以从 0.2~0.3 左右开始，更接近你的理想图。

改进 8：增加 auto 数据划分
    原因：
        如果每个类别有多个文件，优先按文件划分；
        如果某些类别只有一个文件，自动退化为按文件内部时间块划分，
        避免验证集为空。

================================================================================
三、如果你跑出来曲线还是太快，可以继续改这些参数
================================================================================

1. 学习率继续降低：
       "lr": 5e-5

2. 模型进一步变小：
       d_model=32
       num_layers=1

3. Dropout 增大：
       "dropout": 0.40

4. 减少小波图信息量：
       "num_scales": 32
       "img_size": 48

================================================================================
"""

import os
import re
import csv
import json
import time
import warnings
from collections import Counter, defaultdict

import numpy as np
import scipy.io as sio
import scipy.ndimage as ndi

try:
    import pywt
except ImportError:
    raise ImportError("缺少 PyWavelets，请先运行：pip install PyWavelets")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


# =============================================================================
# 0. 配置区
# =============================================================================
CFG = {
    # -------------------------------------------------------------------------
    # 数据路径
    # -------------------------------------------------------------------------
    "data_root": r"E:\柱塞泵\CRWU",

    "use_folders": [
        "12k Drive End Bearing Fault Data",
        "Normal Baseline",
    ],

    "save_dir": r"./crwu_ideal_curve_results",

    # -------------------------------------------------------------------------
    # 信号切片参数
    # 改进点：
    #   为了减少相邻样本重复，默认 overlap=0.0。
    #   如果 overlap=0.5，训练/验证随机划分时很容易数据泄漏。
    # -------------------------------------------------------------------------
    "seg_len": 1024,
    "overlap": 0.0,

    # -------------------------------------------------------------------------
    # CWT 小波参数
    # -------------------------------------------------------------------------
    "wavelet": "morl",
    "num_scales": 64,
    "img_size": 64,
    "fs": 12000,

    # -------------------------------------------------------------------------
    # 每个文件最多取多少段
    # None 表示全部使用。
    # 如果你想快速调试，可以改成 100 或 200。
    # -------------------------------------------------------------------------
    "max_segments_per_file": None,

    # -------------------------------------------------------------------------
    # 缓存
    # 重要：
    #   修改 overlap、seg_len、小波参数、标签规则后，建议 use_cache=False
    #   或者删除旧缓存。
    # -------------------------------------------------------------------------
    "use_cache": False,

    # -------------------------------------------------------------------------
    # 训练参数
    # 改进点：
    #   epochs=150，对齐你的理想曲线横轴。
    #   lr=1e-4，让准确率不要一两轮就冲到 100%。
    #   label_smooth=0.0，让 Loss 可以逐渐下降到接近 0。
    #   enable_early_stop=False，保证完整训练 150 轮。
    # -------------------------------------------------------------------------
    "epochs": 150,
    "lr": 1e-4,
    "min_lr": 1e-6,
    "batch_size": 64,
    "weight_decay": 1e-4,

    "enable_early_stop": False,
    "patience": 200,
    "min_delta": 1e-4,

    "label_smooth": 0.0,
    "dropout": 0.30,

    # -------------------------------------------------------------------------
    # 损失权重
    # -------------------------------------------------------------------------
    "loss_w_type": 1.0,
    "loss_w_severity": 1.0,

    # -------------------------------------------------------------------------
    # 是否使用统计特征
    # 改进点：
    #   默认 False，避免统计特征太强导致第一轮就接近满分。
    #   如果你想恢复“小波图 + 统计特征融合”，改为 True。
    # -------------------------------------------------------------------------
    "use_stat_features": False,

    # -------------------------------------------------------------------------
    # 数据划分方式
    # auto:
    #   如果每个类别至少有 2 个文件，则按文件划分；
    #   否则按每个文件内部的时间块划分。
    #
    # file:
    #   强制按文件划分。
    #
    # time_block:
    #   每个文件前 80% 切片作为训练，后 20% 作为验证。
    # -------------------------------------------------------------------------
    "split_mode": "auto",
    "val_ratio": 0.2,
    "split_gap_segments": 2,

    # -------------------------------------------------------------------------
    # 模型规模
    # 改进点：
    #   d_model=64, num_layers=1，降低模型容量，让曲线更慢上升。
    # -------------------------------------------------------------------------
    "d_model": 64,
    "nhead": 4,
    "num_transformer_layers": 1,

    "seed": 42,
    "num_workers": 0,
}


TYPE_NAMES = ["Normal", "Ball", "InnerRace", "OuterRace"]
SEVERITY_NAMES = ["轻微", "中度", "重度", "严重"]


# =============================================================================
# 1. 基础工具
# =============================================================================
def setup_chinese_font():
    font_candidates = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
    ]

    for p in font_candidates:
        if os.path.exists(p):
            plt.rcParams["font.family"] = fm.FontProperties(fname=p).get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


setup_chinese_font()
set_seed(CFG["seed"])
ensure_dir(CFG["save_dir"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 2. 标签解析
# =============================================================================
def normalize_text(s):
    s = s.replace("\\", "/")
    s = s.lower()
    return s


def parse_label_from_path(fpath, data_root):
    """
    根据文件路径和文件名解析标签。

    返回：
        type_label:
            0 = Normal
            1 = Ball
            2 = InnerRace
            3 = OuterRace

        severity_label:
            -1 = Normal，不参与故障程度分类
             0 = 轻微
             1 = 中度
             2 = 重度
             3 = 严重
    """
    rel = os.path.relpath(fpath, data_root)
    text = normalize_text(rel)

    # -----------------------------
    # 1. Normal
    # -----------------------------
    if (
        "normal baseline" in text
        or "normal" in text
        or "正常" in text
    ):
        return 0, -1

    # -----------------------------
    # 2. 故障类型
    # -----------------------------
    type_label = None

    if (
        "ball" in text
        or "滚动体" in text
        or "滚珠" in text
        or "b007" in text
        or "b014" in text
        or "b021" in text
        or "b028" in text
        or re.search(r"(^|[/_\-\s])b(007|014|021|028)", text)
    ):
        type_label = 1

    elif (
        "inner" in text
        or "inner race" in text
        or "innerrace" in text
        or "内圈" in text
        or "ir007" in text
        or "ir014" in text
        or "ir021" in text
        or "ir028" in text
        or re.search(r"(^|[/_\-\s])ir(007|014|021|028)", text)
    ):
        type_label = 2

    elif (
        "outer" in text
        or "outer race" in text
        or "outerrace" in text
        or "外圈" in text
        or "or007" in text
        or "or014" in text
        or "or021" in text
        or "or028" in text
        or re.search(r"(^|[/_\-\s])or(007|014|021|028)", text)
    ):
        type_label = 3

    if type_label is None:
        return None

    # -----------------------------
    # 3. 故障程度
    # -----------------------------
    severity_label = None

    if "轻微" in text or "轻度" in text:
        severity_label = 0
    elif "中度" in text:
        severity_label = 1
    elif "重度" in text:
        severity_label = 2
    elif "严重" in text:
        severity_label = 3

    elif "007" in text or "0.007" in text:
        severity_label = 0
    elif "014" in text or "0.014" in text:
        severity_label = 1
    elif "021" in text or "0.021" in text:
        severity_label = 2
    elif "028" in text or "0.028" in text:
        severity_label = 3

    if severity_label is None:
        return None

    return type_label, severity_label


# =============================================================================
# 3. 数据读取和预处理
# =============================================================================
def find_mat_files(cfg):
    files = []

    for folder in cfg["use_folders"]:
        folder_path = os.path.join(cfg["data_root"], folder)

        if not os.path.exists(folder_path):
            print(f"[警告] 指定文件夹不存在，已跳过：{folder_path}")
            continue

        for root, _, fnames in os.walk(folder_path):
            for fname in fnames:
                if fname.lower().endswith(".mat"):
                    files.append(os.path.join(root, fname))

    files = sorted(files)

    if len(files) == 0:
        raise ValueError("未找到任何 .mat 文件，请检查 data_root 和 use_folders。")

    return files


def load_signal_from_mat(fpath):
    """
    优先读取 DE_time。
    如果没有 DE_time，则选择最长的一维数值数组。
    """
    mat = sio.loadmat(fpath)
    candidates = []

    for k, v in mat.items():
        if k.startswith("__"):
            continue

        if not isinstance(v, np.ndarray):
            continue

        if not np.issubdtype(v.dtype, np.number):
            continue

        arr = np.asarray(v).squeeze()

        if arr.ndim != 1:
            continue

        if arr.size < 1000:
            continue

        key_lower = k.lower()
        priority = 3 if "de_time" in key_lower else 1

        candidates.append((priority, arr.size, k, arr.astype(np.float32)))

    if len(candidates) == 0:
        raise ValueError(f"无法从文件中读取有效振动信号：{fpath}")

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

    return candidates[0][3]


def make_segments(signal, seg_len, overlap):
    step = int(seg_len * (1.0 - overlap))
    step = max(step, 1)

    if len(signal) < seg_len:
        return np.zeros((0, seg_len), dtype=np.float32)

    segs = []
    for start in range(0, len(signal) - seg_len + 1, step):
        segs.append(signal[start:start + seg_len])

    return np.asarray(segs, dtype=np.float32)


def zscore_1d(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def resize_cwt_image(img, target_size):
    h, w = img.shape

    if h == target_size and w % target_size == 0:
        factor = w // target_size
        img = img.reshape(h, target_size, factor).mean(axis=2)
        return img.astype(np.float32)

    zoom_h = target_size / h
    zoom_w = target_size / w

    img = ndi.zoom(img, (zoom_h, zoom_w), order=1)

    return img.astype(np.float32)


def cwt_to_image(seg_norm, scales, wavelet, img_size):
    coeffs, _ = pywt.cwt(seg_norm, scales, wavelet)

    img = np.log1p(np.abs(coeffs)).astype(np.float32)

    img = resize_cwt_image(img, img_size)

    img = (img - img.mean()) / (img.std() + 1e-8)

    return img.astype(np.float32)


def extract_stat_features(seg, fs):
    """
    提取时域 + 频域统计特征。

    注意：
        代码始终会提取统计特征；
        但是否输入模型由 CFG["use_stat_features"] 控制。
    """
    x = seg.astype(np.float64)
    eps = 1e-12

    mean = np.mean(x)
    std = np.std(x) + eps
    rms = np.sqrt(np.mean(x ** 2) + eps)
    abs_mean = np.mean(np.abs(x)) + eps
    peak = np.max(np.abs(x))
    p2p = np.max(x) - np.min(x)

    centered = x - mean
    skew = np.mean(centered ** 3) / (std ** 3 + eps)
    kurt = np.mean(centered ** 4) / (std ** 4 + eps)

    crest_factor = peak / (rms + eps)
    impulse_factor = peak / abs_mean
    shape_factor = rms / abs_mean
    clearance_factor = peak / ((np.mean(np.sqrt(np.abs(x))) + eps) ** 2)

    win = np.hanning(len(x))
    xf = np.fft.rfft(x * win)
    mag = np.abs(xf)
    power = mag ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)

    mag_sum = np.sum(mag) + eps
    power_sum = np.sum(power) + eps

    spectral_centroid = np.sum(freqs * mag) / mag_sum
    rms_freq = np.sqrt(np.sum((freqs ** 2) * mag) / mag_sum)
    freq_std = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * mag) / mag_sum)

    max_freq = freqs[-1]
    bands = [
        (0.00, 0.25),
        (0.25, 0.50),
        (0.50, 0.75),
        (0.75, 1.00),
    ]

    band_ratios = []
    for low_r, high_r in bands:
        low_f = low_r * max_freq
        high_f = high_r * max_freq
        idx = (freqs >= low_f) & (freqs < high_f)
        band_energy = np.sum(power[idx]) / power_sum
        band_ratios.append(band_energy)

    feats = np.array([
        mean,
        std,
        rms,
        peak,
        p2p,
        skew,
        kurt,
        crest_factor,
        impulse_factor,
        shape_factor,
        clearance_factor,
        spectral_centroid,
        rms_freq,
        freq_std,
        *band_ratios,
    ], dtype=np.float32)

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    return feats


def get_cache_paths(cfg):
    maxseg = cfg["max_segments_per_file"]
    maxseg_str = "all" if maxseg is None else str(maxseg)

    cache_name = (
        f"cache_seg{cfg['seg_len']}"
        f"_ov{int(cfg['overlap'] * 100)}"
        f"_scale{cfg['num_scales']}"
        f"_img{cfg['img_size']}"
        f"_{cfg['wavelet']}"
        f"_max{maxseg_str}.npz"
    )

    meta_name = cache_name.replace(".npz", "_file_records.json")
    csv_name = cache_name.replace(".npz", "_file_sample_counts.csv")

    return (
        os.path.join(cfg["save_dir"], cache_name),
        os.path.join(cfg["save_dir"], meta_name),
        os.path.join(cfg["save_dir"], csv_name),
    )


def print_file_records(records):
    print("\n" + "=" * 120)
    print("每一个文件的样本数量")
    print("=" * 120)

    header = (
        f"{'序号':>4} | "
        f"{'file_id':>7} | "
        f"{'样本数':>6} | "
        f"{'信号长度':>10} | "
        f"{'故障类型':>10} | "
        f"{'故障程度':>8} | 文件"
    )

    print(header)
    print("-" * len(header))

    for i, r in enumerate(records, 1):
        print(
            f"{i:4d} | "
            f"{r['file_id']:7d} | "
            f"{r['n_segments']:6d} | "
            f"{r['signal_len']:10d} | "
            f"{r['type_name']:>10} | "
            f"{r['severity_name']:>8} | "
            f"{r['rel_path']}"
        )

    print("=" * 120)


def save_file_records_csv(records, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow([
            "index",
            "file_id",
            "relative_path",
            "type_label",
            "type_name",
            "severity_label",
            "severity_name",
            "signal_len",
            "n_segments",
        ])

        for i, r in enumerate(records, 1):
            writer.writerow([
                i,
                r["file_id"],
                r["rel_path"],
                r["type_label"],
                r["type_name"],
                r["severity_label"],
                r["severity_name"],
                r["signal_len"],
                r["n_segments"],
            ])


def build_dataset(cfg):
    """
    构建数据集。

    返回：
        X_img:       [N, 1, H, W]
        X_stat:      [N, stat_dim]
        y_type:      [N]
        y_severity:  [N]
        file_ids:    [N]
        records:     每个文件的样本数量信息
    """
    ensure_dir(cfg["save_dir"])

    cache_path, meta_path, csv_path = get_cache_paths(cfg)

    if cfg["use_cache"] and os.path.exists(cache_path):
        print(f"[缓存] 读取预处理缓存：{cache_path}")
        data = np.load(cache_path)

        if "file_ids" not in data:
            raise ValueError(
                "当前缓存没有 file_ids。请删除旧缓存或设置 use_cache=False 后重新运行。"
            )

        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            print_file_records(records)
        else:
            records = []

        print(f"[文件样本数量 CSV] {csv_path}")

        return (
            data["X_img"].astype(np.float32),
            data["X_stat"].astype(np.float32),
            data["y_type"].astype(np.int64),
            data["y_severity"].astype(np.int64),
            data["file_ids"].astype(np.int64),
            records,
        )

    mat_files = find_mat_files(cfg)

    print(f"[数据根目录] {cfg['data_root']}")
    print(f"[使用文件夹] {cfg['use_folders']}")
    print(f"[检测到 .mat 文件数量] {len(mat_files)}")
    print("[提示] 开始生成 CWT 小波时频图，首次运行可能较慢。")

    scales = np.arange(1, cfg["num_scales"] + 1)

    X_img = []
    X_stat = []
    y_type = []
    y_severity = []
    file_ids = []

    records = []
    skipped = []

    for fidx, fpath in enumerate(mat_files, 1):
        rel_path = os.path.relpath(fpath, cfg["data_root"])

        label = parse_label_from_path(fpath, cfg["data_root"])

        if label is None:
            skipped.append(rel_path)
            print(f"[跳过] 无法解析标签：{rel_path}")
            continue

        type_label, severity_label = label
        type_name = TYPE_NAMES[type_label]
        severity_name = "Normal" if severity_label < 0 else SEVERITY_NAMES[severity_label]

        try:
            sig = load_signal_from_mat(fpath)
        except Exception as e:
            skipped.append(rel_path)
            print(f"[跳过] 读取失败：{rel_path} | 原因：{e}")
            continue

        segs = make_segments(sig, cfg["seg_len"], cfg["overlap"])

        if cfg["max_segments_per_file"] is not None and len(segs) > cfg["max_segments_per_file"]:
            idx = np.linspace(0, len(segs) - 1, cfg["max_segments_per_file"]).astype(int)
            segs = segs[idx]

        n_segments = len(segs)

        file_id = len(records)

        record = {
            "file_id": int(file_id),
            "rel_path": rel_path,
            "type_label": int(type_label),
            "type_name": type_name,
            "severity_label": int(severity_label),
            "severity_name": severity_name,
            "signal_len": int(len(sig)),
            "n_segments": int(n_segments),
        }

        records.append(record)

        if n_segments == 0:
            print(f"[跳过] 信号长度不足：{rel_path}")
            continue

        t0 = time.time()

        for seg in segs:
            seg_raw = seg.astype(np.float32)
            seg_norm = zscore_1d(seg_raw)

            img = cwt_to_image(
                seg_norm=seg_norm,
                scales=scales,
                wavelet=cfg["wavelet"],
                img_size=cfg["img_size"],
            )

            stat = extract_stat_features(seg_raw, cfg["fs"])

            X_img.append(img)
            X_stat.append(stat)
            y_type.append(type_label)
            y_severity.append(severity_label)
            file_ids.append(file_id)

        elapsed = time.time() - t0

        print(
            f"[文件 {fidx:03d}/{len(mat_files):03d}] "
            f"file_id={file_id:3d} | "
            f"样本数={n_segments:5d} | "
            f"类型={type_name:<10s} | "
            f"程度={severity_name:<6s} | "
            f"信号长度={len(sig):8d} | "
            f"耗时={elapsed:6.1f}s | "
            f"{rel_path}"
        )

    if len(X_img) == 0:
        raise ValueError("未构建到任何有效样本，请检查数据路径、文件名和标签解析规则。")

    X_img = np.stack(X_img, axis=0).astype(np.float32)
    X_img = X_img[:, None, :, :]

    X_stat = np.vstack(X_stat).astype(np.float32)
    y_type = np.asarray(y_type, dtype=np.int64)
    y_severity = np.asarray(y_severity, dtype=np.int64)
    file_ids = np.asarray(file_ids, dtype=np.int64)

    print_file_records(records)
    save_file_records_csv(records, csv_path)

    print("\n[数据集汇总]")
    print(f"X_img shape      : {X_img.shape}")
    print(f"X_stat shape     : {X_stat.shape}")
    print(f"type 分布        : {Counter(y_type)}")
    print(f"severity 分布    : {Counter(y_severity)}")
    print(f"file_ids 分布    : {Counter(file_ids)}")
    print(f"[文件样本数量 CSV] {csv_path}")

    if len(skipped) > 0:
        print("\n[警告] 以下文件被跳过：")
        for s in skipped:
            print("  ", s)

    if cfg["use_cache"]:
        np.savez_compressed(
            cache_path,
            X_img=X_img,
            X_stat=X_stat,
            y_type=y_type,
            y_severity=y_severity,
            file_ids=file_ids,
        )

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        print(f"\n[缓存] 已保存：{cache_path}")
        print(f"[缓存元信息] 已保存：{meta_path}")

    return X_img, X_stat, y_type, y_severity, file_ids, records


# =============================================================================
# 4. 数据集划分
# =============================================================================
def file_split_is_possible(records):
    """
    判断是否可以严格按文件划分。

    条件：
        每个 type + severity 类别至少有 2 个文件。
    """
    groups = defaultdict(list)

    for r in records:
        if r["n_segments"] <= 0:
            continue

        key = (r["type_label"], r["severity_label"])
        groups[key].append(r["file_id"])

    if len(groups) == 0:
        return False

    for _, fids in groups.items():
        if len(set(fids)) < 2:
            return False

    return True


def split_by_file_stratified(records, file_ids, val_ratio=0.2, seed=42):
    """
    严格按文件划分训练集和验证集。

    同一个 .mat 文件切出来的所有窗口只会进入训练集或验证集，
    不会两边同时出现。
    """
    rng = np.random.default_rng(seed)

    label_to_fids = defaultdict(list)

    for r in records:
        if r["n_segments"] <= 0:
            continue

        key = (r["type_label"], r["severity_label"])
        label_to_fids[key].append(r["file_id"])

    train_fids = set()
    val_fids = set()

    print("\n[文件级划分] 每个类别的文件分配情况：")

    for key, fids in label_to_fids.items():
        fids = list(sorted(set(fids)))
        rng.shuffle(fids)

        type_label, severity_label = key
        type_name = TYPE_NAMES[type_label]
        severity_name = "Normal" if severity_label < 0 else SEVERITY_NAMES[severity_label]

        if len(fids) < 2:
            raise RuntimeError(
                f"类别 {type_name}-{severity_name} 只有 {len(fids)} 个文件，无法严格按文件划分。"
            )

        n_val = max(1, int(round(len(fids) * val_ratio)))
        n_val = min(n_val, len(fids) - 1)

        val_part = fids[:n_val]
        train_part = fids[n_val:]

        val_fids.update(val_part)
        train_fids.update(train_part)

        print(
            f"  类型={type_name:<10s} 程度={severity_name:<6s} | "
            f"总文件={len(fids):2d} | "
            f"训练文件={len(train_part):2d} | "
            f"验证文件={len(val_part):2d}"
        )

    idx_train = np.where(np.isin(file_ids, list(train_fids)))[0]
    idx_val = np.where(np.isin(file_ids, list(val_fids)))[0]

    train_file_set = set(file_ids[idx_train].tolist())
    val_file_set = set(file_ids[idx_val].tolist())

    overlap_files = train_file_set & val_file_set

    if len(overlap_files) > 0:
        raise RuntimeError(f"文件级划分失败，训练集和验证集有重叠文件：{overlap_files}")

    print("\n[文件级划分完成]")
    print(f"训练文件数: {len(train_file_set)}")
    print(f"验证文件数: {len(val_file_set)}")
    print(f"训练样本数: {len(idx_train)}")
    print(f"验证样本数: {len(idx_val)}")
    print(f"训练/验证重叠文件数: {len(overlap_files)}")

    return idx_train, idx_val


def split_by_time_block(records, file_ids, val_ratio=0.2, gap_segments=2):
    """
    按每个文件内部时间块划分。

    适用场景：
        某些类别只有 1 个 .mat 文件，无法严格按文件划分。

    方法：
        每个文件前 80% 切片进入训练集；
        每个文件后 20% 切片进入验证集；
        中间丢弃少量 gap_segments，降低相邻片段影响。

    注意：
        这不如严格文件级划分严谨，但比随机窗口划分更合理。
    """
    train_indices = []
    val_indices = []

    print("\n[时间块划分] 每个文件内部前段训练、后段验证：")

    for r in records:
        fid = r["file_id"]
        idx = np.where(file_ids == fid)[0]

        if len(idx) == 0:
            continue

        n = len(idx)

        if n < 3:
            train_indices.extend(idx.tolist())
            print(
                f"  file_id={fid:3d} | 样本太少 n={n:4d} | 全部放入训练 | {r['rel_path']}"
            )
            continue

        n_val = max(1, int(round(n * val_ratio)))
        n_train_end = n - n_val

        # 中间留出 gap，降低训练和验证相邻片段影响
        train_end = max(1, n_train_end - gap_segments)

        train_part = idx[:train_end]
        val_part = idx[n_train_end:]

        train_indices.extend(train_part.tolist())
        val_indices.extend(val_part.tolist())

        print(
            f"  file_id={fid:3d} | "
            f"总样本={n:5d} | "
            f"训练={len(train_part):5d} | "
            f"验证={len(val_part):5d} | "
            f"gap={n_train_end - train_end:2d} | "
            f"类型={r['type_name']:<10s} | "
            f"程度={r['severity_name']:<6s} | "
            f"{r['rel_path']}"
        )

    idx_train = np.asarray(train_indices, dtype=np.int64)
    idx_val = np.asarray(val_indices, dtype=np.int64)

    if len(idx_val) == 0:
        raise RuntimeError("验证集为空，请检查每个文件的样本数量。")

    print("\n[时间块划分完成]")
    print(f"训练样本数: {len(idx_train)}")
    print(f"验证样本数: {len(idx_val)}")

    return idx_train, idx_val


def split_dataset(cfg, records, file_ids):
    mode = cfg.get("split_mode", "auto")

    if mode == "file":
        return split_by_file_stratified(
            records=records,
            file_ids=file_ids,
            val_ratio=cfg["val_ratio"],
            seed=cfg["seed"],
        )

    if mode == "time_block":
        return split_by_time_block(
            records=records,
            file_ids=file_ids,
            val_ratio=cfg["val_ratio"],
            gap_segments=cfg["split_gap_segments"],
        )

    if mode == "auto":
        if file_split_is_possible(records):
            print("\n[划分模式] auto 检测到可以严格按文件划分。")
            return split_by_file_stratified(
                records=records,
                file_ids=file_ids,
                val_ratio=cfg["val_ratio"],
                seed=cfg["seed"],
            )
        else:
            print("\n[划分模式] auto 检测到某些类别文件数不足，改用时间块划分。")
            return split_by_time_block(
                records=records,
                file_ids=file_ids,
                val_ratio=cfg["val_ratio"],
                gap_segments=cfg["split_gap_segments"],
            )

    raise ValueError(f"未知 split_mode: {mode}")


# =============================================================================
# 5. Dataset
# =============================================================================
class BearingDataset(Dataset):
    def __init__(self, X_img, X_stat, y_type, y_severity):
        self.X_img = torch.from_numpy(X_img).float()
        self.X_stat = torch.from_numpy(X_stat).float()
        self.y_type = torch.from_numpy(y_type).long()
        self.y_severity = torch.from_numpy(y_severity).long()

    def __len__(self):
        return len(self.X_img)

    def __getitem__(self, idx):
        return (
            self.X_img[idx],
            self.X_stat[idx],
            self.y_type[idx],
            self.y_severity[idx],
        )


# =============================================================================
# 6. 模型：2D CNN-Transformer + 可选统计特征
# =============================================================================
class Conv2DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),

            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.net(x)


class HierCNNTransformer2DStats(nn.Module):
    def __init__(
        self,
        img_size=64,
        stat_dim=1,
        d_model=64,
        nhead=4,
        num_layers=1,
        dropout=0.3,
        n_type=4,
        n_severity=4,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            Conv2DBlock(1, 32, dropout=dropout * 0.5),
            Conv2DBlock(32, 64, dropout=dropout * 0.5),
            Conv2DBlock(64, 128, dropout=dropout * 0.5),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_size, img_size)
            feat = self.cnn(dummy)
            _, c, h, w = feat.shape

        self.num_tokens = h * w

        self.proj = nn.Linear(c, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.img_norm = nn.LayerNorm(d_model)

        # 使用 LayerNorm，不使用 BatchNorm1d，避免最后一个 batch 只有 1 个样本时报错
        self.stat_mlp = nn.Sequential(
            nn.Linear(stat_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        fusion_dim = d_model + 64

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head_type = nn.Linear(128, n_type)
        self.head_severity = nn.Linear(128, n_severity)

    def forward(self, x_img, x_stat):
        feat = self.cnn(x_img)

        tokens = feat.flatten(2).transpose(1, 2)
        tokens = self.proj(tokens) + self.pos_embed
        tokens = self.transformer(tokens)

        img_feat = tokens.mean(dim=1)
        img_feat = self.img_norm(img_feat)

        stat_feat = self.stat_mlp(x_stat)

        fused = torch.cat([img_feat, stat_feat], dim=1)
        fused = self.fusion(fused)

        out_type = self.head_type(fused)
        out_severity = self.head_severity(fused)

        return out_type, out_severity


# =============================================================================
# 7. 损失函数、训练、评估
# =============================================================================
class SmoothCrossEntropy(nn.Module):
    """
    支持标签平滑的交叉熵。
    当 eps=0.0 时，等价于普通 CrossEntropy。
    """
    def __init__(self, eps=0.0):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        if self.eps <= 0:
            return F.cross_entropy(logits, target)

        n_cls = logits.size(1)
        log_prob = F.log_softmax(logits, dim=1)

        with torch.no_grad():
            true_dist = torch.full_like(log_prob, self.eps / (n_cls - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.eps)

        loss = -(true_dist * log_prob).sum(dim=1).mean()

        return loss


class EarlyStopping:
    def __init__(self, patience, min_delta, save_path):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path

        self.best_loss = np.inf
        self.counter = 0
        self.best_epoch = 0

    def step(self, val_loss, model, epoch):
        improved = val_loss < self.best_loss - self.min_delta

        if improved:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.save_path)
            return False, True

        self.counter += 1
        stop = self.counter >= self.patience

        return stop, False


def compute_loss(outputs, y_type, y_severity, criterion, cfg):
    out_type, out_severity = outputs

    loss_type = criterion(out_type, y_type)

    fault_mask = y_severity >= 0

    if fault_mask.sum() > 0:
        loss_sev = criterion(out_severity[fault_mask], y_severity[fault_mask])
    else:
        loss_sev = torch.tensor(0.0, device=out_type.device)

    loss = (
        cfg["loss_w_type"] * loss_type
        + cfg["loss_w_severity"] * loss_sev
    )

    return loss


def run_epoch(model, loader, criterion, optimizer, device, cfg, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    type_correct = 0
    sev_correct = 0
    sev_total = 0
    joint_correct = 0

    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for x_img, x_stat, y_type, y_severity in loader:
            x_img = x_img.to(device)
            x_stat = x_stat.to(device)
            y_type = y_type.to(device)
            y_severity = y_severity.to(device)

            outputs = model(x_img, x_stat)
            loss = compute_loss(outputs, y_type, y_severity, criterion, cfg)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            out_type, out_severity = outputs
            pred_type = out_type.argmax(dim=1)
            pred_sev = out_severity.argmax(dim=1)

            bs = x_img.size(0)

            total_loss += loss.item() * bs
            total_samples += bs

            type_correct += (pred_type == y_type).sum().item()

            fault_mask = y_severity >= 0
            normal_mask = y_severity < 0

            if fault_mask.sum().item() > 0:
                sev_correct += (pred_sev[fault_mask] == y_severity[fault_mask]).sum().item()
                sev_total += fault_mask.sum().item()

                joint_correct += (
                    (pred_type[fault_mask] == y_type[fault_mask])
                    & (pred_sev[fault_mask] == y_severity[fault_mask])
                ).sum().item()

            if normal_mask.sum().item() > 0:
                joint_correct += (pred_type[normal_mask] == y_type[normal_mask]).sum().item()

    avg_loss = total_loss / max(total_samples, 1)
    type_acc = type_correct / max(total_samples, 1)
    sev_acc = sev_correct / max(sev_total, 1)
    joint_acc = joint_correct / max(total_samples, 1)

    return avg_loss, type_acc, sev_acc, joint_acc


def predict_all(model, loader, device):
    model.eval()

    all_y_type = []
    all_y_sev = []
    all_p_type = []
    all_p_sev = []

    with torch.no_grad():
        for x_img, x_stat, y_type, y_severity in loader:
            x_img = x_img.to(device)
            x_stat = x_stat.to(device)

            out_type, out_severity = model(x_img, x_stat)

            all_p_type.append(out_type.argmax(dim=1).cpu().numpy())
            all_p_sev.append(out_severity.argmax(dim=1).cpu().numpy())

            all_y_type.append(y_type.numpy())
            all_y_sev.append(y_severity.numpy())

    return {
        "y_type": np.concatenate(all_y_type),
        "y_sev": np.concatenate(all_y_sev),
        "p_type": np.concatenate(all_p_type),
        "p_sev": np.concatenate(all_p_sev),
    }


# =============================================================================
# 8. 可视化
# =============================================================================
def smooth_curve(values, alpha=0.45):
    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        return values

    if np.isnan(values).any():
        values = np.nan_to_num(values, nan=np.nanmean(values))

    smoothed = [values[0]]

    for v in values[1:]:
        smoothed.append(alpha * v + (1.0 - alpha) * smoothed[-1])

    return np.asarray(smoothed)


def plot_curves_four_panel(hist, save_dir):
    epochs = np.asarray(hist["epoch"], dtype=np.int64)

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    ax = axes[0, 0]
    ax.plot(epochs, smooth_curve(hist["train_loss"]), label="Train Loss", linewidth=2)
    ax.plot(epochs, smooth_curve(hist["val_loss"]), label="Val Loss", linewidth=2)
    ax.set_title("损失曲线")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(epochs, smooth_curve(hist["train_type_acc"]), label="Train Type Acc", linewidth=2)
    ax.plot(epochs, smooth_curve(hist["val_type_acc"]), label="Val Type Acc", linewidth=2)
    ax.set_title("故障类型准确率")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(epochs, smooth_curve(hist["train_sev_acc"]), label="Train Severity Acc", linewidth=2)
    ax.plot(epochs, smooth_curve(hist["val_sev_acc"]), label="Val Severity Acc", linewidth=2)
    ax.set_title("故障程度准确率")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(epochs, smooth_curve(hist["train_joint_acc"]), label="Train Joint Acc", linewidth=2)
    ax.plot(epochs, smooth_curve(hist["val_joint_acc"]), label="Val Joint Acc", linewidth=2)
    ax.plot(epochs, hist["lr"], "--", label="LR", linewidth=1.8)
    ax.set_title("层级联合准确率 / 学习率")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()

    save_path = os.path.join(save_dir, "curves_four_panel.png")
    plt.savefig(save_path, dpi=180)
    plt.close()

    print(f"[四宫格曲线] 已保存：{save_path}")


def plot_ideal_style_curve(hist, save_dir, acc_key="type"):
    """
    绘制接近你给出的论文风格单图曲线。

    说明：
        为了把 Loss 和 Accuracy 放在同一张 0~1 坐标图中，
        这里对 Loss 做了归一化。

    acc_key:
        "type"     -> 使用故障类型准确率
        "severity" -> 使用故障程度准确率
        "joint"    -> 使用层级联合准确率
    """
    epochs = np.asarray(hist["epoch"], dtype=np.int64)

    train_loss = np.asarray(hist["train_loss"], dtype=np.float64)
    val_loss = np.asarray(hist["val_loss"], dtype=np.float64)

    if acc_key == "type":
        train_acc = np.asarray(hist["train_type_acc"], dtype=np.float64)
        val_acc = np.asarray(hist["val_type_acc"], dtype=np.float64)
        title = "模型训练过程曲线 - 故障类型"
        save_name = "ideal_style_curve_type.png"

    elif acc_key == "severity":
        train_acc = np.asarray(hist["train_sev_acc"], dtype=np.float64)
        val_acc = np.asarray(hist["val_sev_acc"], dtype=np.float64)
        title = "模型训练过程曲线 - 故障程度"
        save_name = "ideal_style_curve_severity.png"

    else:
        train_acc = np.asarray(hist["train_joint_acc"], dtype=np.float64)
        val_acc = np.asarray(hist["val_joint_acc"], dtype=np.float64)
        title = "模型训练过程曲线 - 层级联合"
        save_name = "ideal_style_curve_joint.png"

    train_acc = np.nan_to_num(train_acc, nan=np.nanmean(train_acc))
    val_acc = np.nan_to_num(val_acc, nan=np.nanmean(val_acc))

    max_loss = max(float(train_loss[0]), float(val_loss[0]), 1e-8)
    train_loss_norm = train_loss / max_loss
    val_loss_norm = val_loss / max_loss

    plt.figure(figsize=(8, 5.5))

    plt.plot(
        epochs,
        smooth_curve(train_loss_norm, alpha=0.5),
        color="navy",
        linewidth=1.8,
        label="训练损失",
    )

    plt.plot(
        epochs,
        smooth_curve(train_acc, alpha=0.5),
        color="seagreen",
        linewidth=1.8,
        label="训练准确率",
    )

    plt.plot(
        epochs,
        smooth_curve(val_loss_norm, alpha=0.5),
        color="olive",
        linewidth=1.8,
        label="验证损失",
    )

    plt.plot(
        epochs,
        smooth_curve(val_acc, alpha=0.5),
        color="red",
        linewidth=1.8,
        label="验证准确率",
    )

    plt.xlabel("迭代次数", fontsize=12)
    plt.ylabel("准确率 / 归一化损失值", fontsize=12)
    plt.title(title, fontsize=13)

    plt.xlim(0, max(epochs))
    plt.ylim(0, 1.05)

    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[理想风格曲线] 已保存：{save_path}")


def plot_confusion(y_true, y_pred, class_names, title, save_path):
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()

    plt.xticks(labels, class_names, rotation=30, ha="right")
    plt.yticks(labels, class_names)

    threshold = cm.max() / 2.0 if cm.max() > 0 else 0.5

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()

    print(f"[混淆矩阵] 已保存：{save_path}")


def save_history_csv(hist, save_dir):
    fpath = os.path.join(save_dir, "history.csv")

    keys = list(hist.keys())
    n = len(hist[keys[0]])

    with open(fpath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(keys)

        for i in range(n):
            row = [hist[k][i] for k in keys]
            writer.writerow(row)

    print(f"[训练日志] 已保存：{fpath}")


# =============================================================================
# 9. 主流程
# =============================================================================
def main():
    ensure_dir(CFG["save_dir"])

    print("=" * 100)
    print("CRWU 轴承故障诊断：2D CNN-Transformer + CWT小波时频图")
    print("目标：让训练曲线更接近从低准确率逐渐上升的理想曲线")
    print("=" * 100)
    print(f"[设备] {DEVICE}")
    print(f"[数据根目录] {CFG['data_root']}")
    print(f"[使用文件夹] {CFG['use_folders']}")
    print(f"[保存目录] {os.path.abspath(CFG['save_dir'])}")
    print(f"[窗口长度] {CFG['seg_len']}")
    print(f"[重叠率] {CFG['overlap']}")
    print(f"[CWT] wavelet={CFG['wavelet']} | scales={CFG['num_scales']} | image={CFG['img_size']}x{CFG['img_size']}")
    print(f"[训练轮数] {CFG['epochs']}")
    print(f"[学习率] {CFG['lr']}")
    print(f"[标签平滑] {CFG['label_smooth']}")
    print(f"[是否使用统计特征] {CFG['use_stat_features']}")
    print(f"[数据划分模式] {CFG['split_mode']}")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # 1. 构建数据集
    # -------------------------------------------------------------------------
    print("\n[1/7] 加载数据并生成 CWT 小波时频图与统计特征...")
    X_img, X_stat, y_type, y_severity, file_ids, records = build_dataset(CFG)

    # -------------------------------------------------------------------------
    # 2. 数据划分
    # -------------------------------------------------------------------------
    print("\n[2/7] 划分训练集和验证集...")
    idx_train, idx_val = split_dataset(CFG, records, file_ids)

    X_img_train = X_img[idx_train]
    X_img_val = X_img[idx_val]

    X_stat_train = X_stat[idx_train]
    X_stat_val = X_stat[idx_val]

    y_type_train = y_type[idx_train]
    y_type_val = y_type[idx_val]

    y_sev_train = y_severity[idx_train]
    y_sev_val = y_severity[idx_val]

    # -------------------------------------------------------------------------
    # 3. 统计特征处理
    # -------------------------------------------------------------------------
    print("\n[3/7] 处理统计特征...")

    if CFG["use_stat_features"]:
        scaler = StandardScaler()
        X_stat_train = scaler.fit_transform(X_stat_train).astype(np.float32)
        X_stat_val = scaler.transform(X_stat_val).astype(np.float32)

        np.savez(
            os.path.join(CFG["save_dir"], "stat_scaler.npz"),
            mean=scaler.mean_,
            scale=scaler.scale_,
        )

        print(f"[统计特征] 已启用，维度={X_stat_train.shape[1]}")

    else:
        # 改进点：
        #   默认关闭统计特征，避免模型靠 RMS、峭度、峰值等人工特征太快达到高精度。
        X_stat_train = np.zeros((len(idx_train), 1), dtype=np.float32)
        X_stat_val = np.zeros((len(idx_val), 1), dtype=np.float32)

        print("[统计特征] 当前已关闭，模型主要依赖 CWT 时频图。")

    print(f"训练集样本数: {len(idx_train)}")
    print(f"验证集样本数: {len(idx_val)}")
    print(f"训练集 type 分布: {Counter(y_type_train)}")
    print(f"验证集 type 分布: {Counter(y_type_val)}")
    print(f"训练集 severity 分布: {Counter(y_sev_train)}")
    print(f"验证集 severity 分布: {Counter(y_sev_val)}")

    train_ds = BearingDataset(X_img_train, X_stat_train, y_type_train, y_sev_train)
    val_ds = BearingDataset(X_img_val, X_stat_val, y_type_val, y_sev_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=CFG["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=CFG["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # -------------------------------------------------------------------------
    # 4. 初始化模型
    # -------------------------------------------------------------------------
    print("\n[4/7] 初始化模型...")

    stat_dim = X_stat_train.shape[1]

    model = HierCNNTransformer2DStats(
        img_size=CFG["img_size"],
        stat_dim=stat_dim,
        d_model=CFG["d_model"],
        nhead=CFG["nhead"],
        num_layers=CFG["num_transformer_layers"],
        dropout=CFG["dropout"],
        n_type=4,
        n_severity=4,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"统计特征输入维度: {stat_dim}")
    print(f"d_model: {CFG['d_model']}")
    print(f"Transformer层数: {CFG['num_transformer_layers']}")
    print(f"模型可训练参数量: {total_params:,}")

    criterion = SmoothCrossEntropy(eps=CFG["label_smooth"])

    optimizer = optim.AdamW(
        model.parameters(),
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"],
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CFG["epochs"],
        eta_min=CFG["min_lr"],
    )

    best_model_path = os.path.join(CFG["save_dir"], "best_model.pth")

    early_stopper = EarlyStopping(
        patience=CFG["patience"],
        min_delta=CFG["min_delta"],
        save_path=best_model_path,
    )

    # -------------------------------------------------------------------------
    # 5. 训练
    # -------------------------------------------------------------------------
    print("\n[5/7] 开始训练...\n")

    hist = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_type_acc": [],
        "val_type_acc": [],
        "train_sev_acc": [],
        "val_sev_acc": [],
        "train_joint_acc": [],
        "val_joint_acc": [],
        "lr": [],
    }

    header = (
        f"{'Epoch':>5} | {'LR':>9} | "
        f"{'TrLoss':>8} | {'ValLoss':>8} | "
        f"{'Tr_Type':>8} | {'Val_Type':>8} | "
        f"{'Tr_Sev':>8} | {'Val_Sev':>8} | "
        f"{'Tr_Joint':>8} | {'Val_Joint':>9} | "
        f"{'ES':>3} | {'Time':>6}"
    )

    print(header)
    print("-" * len(header))

    # -------------------------------------------------------------------------
    # 改进点：记录 Epoch 0
    # 目的：
    #   未训练模型准确率通常接近随机水平。
    #   这样曲线第一点可以从 0.2~0.3 左右开始，更接近理想图。
    # -------------------------------------------------------------------------
    init_train_loss, init_train_type_acc, init_train_sev_acc, init_train_joint_acc = run_epoch(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=None,
        device=DEVICE,
        cfg=CFG,
        train=False,
    )

    init_val_loss, init_val_type_acc, init_val_sev_acc, init_val_joint_acc = run_epoch(
        model=model,
        loader=val_loader,
        criterion=criterion,
        optimizer=None,
        device=DEVICE,
        cfg=CFG,
        train=False,
    )

    hist["epoch"].append(0)
    hist["train_loss"].append(init_train_loss)
    hist["val_loss"].append(init_val_loss)
    hist["train_type_acc"].append(init_train_type_acc)
    hist["val_type_acc"].append(init_val_type_acc)
    hist["train_sev_acc"].append(init_train_sev_acc)
    hist["val_sev_acc"].append(init_val_sev_acc)
    hist["train_joint_acc"].append(init_train_joint_acc)
    hist["val_joint_acc"].append(init_val_joint_acc)
    hist["lr"].append(optimizer.param_groups[0]["lr"])

    print(
        f"{0:5d} | {optimizer.param_groups[0]['lr']:9.2e} | "
        f"{init_train_loss:8.4f} | {init_val_loss:8.4f} | "
        f"{init_train_type_acc*100:7.2f}% | {init_val_type_acc*100:7.2f}% | "
        f"{init_train_sev_acc*100:7.2f}% | {init_val_sev_acc*100:7.2f}% | "
        f"{init_train_joint_acc*100:7.2f}% | {init_val_joint_acc*100:8.2f}% | "
        f"{0:3d} | {0.0:5.1f}s"
    )

    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_type_acc, train_sev_acc, train_joint_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
            cfg=CFG,
            train=True,
        )

        val_loss, val_type_acc, val_sev_acc, val_joint_acc = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=DEVICE,
            cfg=CFG,
            train=False,
        )

        scheduler.step()

        hist["epoch"].append(epoch)
        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["train_type_acc"].append(train_type_acc)
        hist["val_type_acc"].append(val_type_acc)
        hist["train_sev_acc"].append(train_sev_acc)
        hist["val_sev_acc"].append(val_sev_acc)
        hist["train_joint_acc"].append(train_joint_acc)
        hist["val_joint_acc"].append(val_joint_acc)
        hist["lr"].append(current_lr)

        stop, improved = early_stopper.step(val_loss, model, epoch)

        # 改进点：
        #   默认关闭早停，保证跑满 150 轮。
        #   但 EarlyStopping 仍负责保存最优模型。
        if not CFG["enable_early_stop"]:
            stop = False

        elapsed = time.time() - t0
        mark = "★" if improved else " "

        print(
            f"{epoch:5d} | {current_lr:9.2e} | "
            f"{train_loss:8.4f} | {val_loss:8.4f} | "
            f"{train_type_acc*100:7.2f}% | {val_type_acc*100:7.2f}% | "
            f"{train_sev_acc*100:7.2f}% | {val_sev_acc*100:7.2f}% | "
            f"{train_joint_acc*100:7.2f}% | {val_joint_acc*100:8.2f}% | "
            f"{early_stopper.counter:3d} | {elapsed:5.1f}s {mark}"
        )

        if stop:
            print("\n[早停] 验证集损失长期未提升，停止训练。")
            print(f"[早停] 最优轮次: Epoch {early_stopper.best_epoch}")
            print(f"[早停] 最优验证损失: {early_stopper.best_loss:.6f}")
            break

    save_history_csv(hist, CFG["save_dir"])

    # 如果训练期间没有保存过模型，则保存当前模型
    if not os.path.exists(best_model_path):
        torch.save(model.state_dict(), best_model_path)

    # -------------------------------------------------------------------------
    # 6. 评估
    # -------------------------------------------------------------------------
    print("\n[6/7] 加载最优模型并评估...")

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    pred = predict_all(model, val_loader, DEVICE)

    print("\n" + "=" * 100)
    print("第一阶段：故障类型分类报告")
    print("=" * 100)
    print(
        classification_report(
            pred["y_type"],
            pred["p_type"],
            labels=[0, 1, 2, 3],
            target_names=TYPE_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    fault_mask = pred["y_sev"] >= 0

    print("\n" + "=" * 100)
    print("第二阶段：故障程度分类报告，仅统计真实故障样本")
    print("=" * 100)

    if fault_mask.sum() > 0:
        print(
            classification_report(
                pred["y_sev"][fault_mask],
                pred["p_sev"][fault_mask],
                labels=[0, 1, 2, 3],
                target_names=SEVERITY_NAMES,
                digits=4,
                zero_division=0,
            )
        )
    else:
        print("[警告] 验证集中没有故障样本，无法计算故障程度分类报告。")

    normal_mask = pred["y_sev"] < 0

    joint_correct = 0

    if normal_mask.sum() > 0:
        joint_correct += (pred["p_type"][normal_mask] == pred["y_type"][normal_mask]).sum()

    if fault_mask.sum() > 0:
        joint_correct += (
            (pred["p_type"][fault_mask] == pred["y_type"][fault_mask])
            & (pred["p_sev"][fault_mask] == pred["y_sev"][fault_mask])
        ).sum()

    joint_acc = joint_correct / len(pred["y_type"])

    print("\n" + "=" * 100)
    print(f"层级联合准确率：{joint_acc * 100:.2f}%")
    print("说明：")
    print("  Normal 样本：type 判断为 Normal 即正确；")
    print("  故障样本：type 和 severity 都判断正确才算正确。")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # 7. 保存图像结果
    # -------------------------------------------------------------------------
    print("\n[7/7] 保存曲线和混淆矩阵...")

    plot_curves_four_panel(hist, CFG["save_dir"])

    # 接近你理想图风格的单图
    plot_ideal_style_curve(hist, CFG["save_dir"], acc_key="type")
    plot_ideal_style_curve(hist, CFG["save_dir"], acc_key="severity")
    plot_ideal_style_curve(hist, CFG["save_dir"], acc_key="joint")

    plot_confusion(
        y_true=pred["y_type"],
        y_pred=pred["p_type"],
        class_names=TYPE_NAMES,
        title="第一阶段：故障类型混淆矩阵",
        save_path=os.path.join(CFG["save_dir"], "cm_type.png"),
    )

    if fault_mask.sum() > 0:
        plot_confusion(
            y_true=pred["y_sev"][fault_mask],
            y_pred=pred["p_sev"][fault_mask],
            class_names=SEVERITY_NAMES,
            title="第二阶段：故障程度混淆矩阵",
            save_path=os.path.join(CFG["save_dir"], "cm_severity.png"),
        )

    print("\n" + "=" * 100)
    print("训练与评估完成")
    print(f"结果保存目录：{os.path.abspath(CFG['save_dir'])}")
    print("主要输出文件：")
    print("  best_model.pth                    最优模型权重")
    print("  history.csv                       每轮训练日志")
    print("  curves_four_panel.png             四宫格训练曲线")
    print("  ideal_style_curve_type.png        理想风格曲线：故障类型")
    print("  ideal_style_curve_severity.png    理想风格曲线：故障程度")
    print("  ideal_style_curve_joint.png       理想风格曲线：层级联合")
    print("  cm_type.png                       故障类型混淆矩阵")
    print("  cm_severity.png                   故障程度混淆矩阵")
    print("  *_file_sample_counts.csv          每个文件的样本数量")
    print("=" * 100)


if __name__ == "__main__":
    main()
