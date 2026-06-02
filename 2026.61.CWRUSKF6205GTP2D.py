# -*- coding: utf-8 -*-
"""
CWRU 轴承故障诊断：2D CNN-Transformer + CWT 小波时频图 + 统计特征融合

================================================================================
一、数据结构说明
================================================================================

数据根目录：
    E:\\柱塞泵\\CWRU轴承数据\\cwru_data

目录下包含四种工况：
    0HP
    1HP
    2HP
    3HP

当前代码默认只使用一种工况：
    0HP

如果想切换工况，只需要修改 CFG["work_condition"]：
    "work_condition": "1HP"
    "work_condition": "2HP"
    "work_condition": "3HP"

================================================================================
二、当前任务逻辑
================================================================================

本代码采用层级诊断逻辑：

第一阶段：
    判断 Normal 还是 Fault
        0 -> Normal
        1 -> Fault

第二阶段：
    如果是 Fault，判断故障类型
        0 -> Ball / 滚动体
        1 -> InnerRace / 内圈
        2 -> OuterRace / 外圈

第三阶段：
    如果是 Fault，判断故障程度
        0 -> 轻度，对应 007
        1 -> 中度，对应 014
        2 -> 重度，对应 021

正常样本：
    y_binary = 0
    y_type = -1
    y_severity = -1

故障样本：
    y_binary = 1
    y_type in [0, 1, 2]
    y_severity in [0, 1, 2]

================================================================================
三、整体流程
================================================================================

1. 读取指定工况文件夹，例如：
       E:\\柱塞泵\\CWRU轴承数据\\cwru_data\\0HP

2. 递归扫描该工况下所有 .mat 文件。

3. 根据文件路径和文件名自动解析标签：
       Normal
       Ball / InnerRace / OuterRace
       007 / 014 / 021

4. 从 .mat 文件中优先读取 DE_time 振动信号。
   如果没有 DE_time，则选择最长的一维数值数组。

5. 将一维振动信号切成多个片段。

6. 对每个片段：
       1) Z-score 标准化；
       2) 连续小波变换 CWT；
       3) 生成二维小波时频图；
       4) 提取时域 + 频域统计特征。

7. 模型输入：
       1) CWT 二维时频图 -> 2D CNN -> Transformer；
       2) 统计特征 -> MLP；
       3) 两路特征融合。

8. 模型输出三个头：
       1) Normal/Fault 二分类头；
       2) 故障类型分类头；
       3) 故障程度分类头。

9. 损失函数：
       binary loss 对所有样本计算；
       type loss 只对故障样本计算；
       severity loss 只对故障样本计算。

10. 训练过程中详细打印：
       Epoch
       学习率
       训练损失
       验证损失
       Normal/Fault 准确率
       故障类型准确率
       故障程度准确率
       层级联合准确率
       早停计数
       单轮耗时

11. 训练结束输出：
       训练/验证损失曲线
       准确率曲线
       Normal/Fault 混淆矩阵
       故障类型混淆矩阵
       故障程度混淆矩阵
       最终层级诊断混淆矩阵
       分类报告
       history.csv
       best_model.pth

================================================================================
四、为了让训练曲线不要过于震荡，代码做了这些改进
================================================================================

改进 1：使用 AdamW 优化器
    比普通 Adam 更稳定，并带有权重衰减。

改进 2：使用较小学习率
    默认 lr = 3e-4，避免一开始震荡过大。

改进 3：使用 Warmup + Cosine 学习率
    前几轮学习率逐渐升高，后面平滑下降。

改进 4：加入 Dropout
    降低过拟合和曲线抖动。



改进 6：梯度裁剪
    防止梯度突然变大造成曲线震荡。

改进 7：统计特征标准化
    统计特征只用训练集拟合 StandardScaler，避免数据泄漏。

改进 8：默认采用 time_block 划分
    每个文件前段作为训练，后段作为验证，中间留 gap，
    比完全随机窗口划分更稳，也能减少相邻窗口泄漏。

改进 9：早停机制
    验证损失长期不提升时自动停止，避免后面白跑。

改进 10：曲线绘图时做轻微平滑
    只用于显示，不影响真实训练日志。

================================================================================
五、后续如何改进
================================================================================

1. 换工况：
       CFG["work_condition"] = "1HP"

2. 想让模型更强：
       CFG["d_model"] = 128
       CFG["num_transformer_layers"] = 2

3. 想让训练更稳：
       CFG["lr"] = 1e-4
       CFG["dropout"] = 0.30
       CFG["label_smooth"] = 0.08

4. 想让 loss 降得更低：
       CFG["label_smooth"] = 0.0

5. 想加快调试：
       CFG["max_segments_per_file"] = 100

6. 想提高时频图分辨率：
       CFG["img_size"] = 96
       CFG["num_scales"] = 96

7. 如果修改了切片长度、小波参数、工况或标签规则：
       建议删除缓存，或者设置：
       CFG["use_cache"] = False
"""

import os
import re
import csv
import json
import time
import math
import warnings
from collections import Counter

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
    # 数据根目录
    # -------------------------------------------------------------------------
    "data_root": r"E:\柱塞泵\CWRU轴承数据\cwru_data",

    # -------------------------------------------------------------------------
    # 当前只做一种工况
    # 可选：0HP / 1HP / 2HP / 3HP
    # -------------------------------------------------------------------------
    "work_condition": "0HP",

    # -------------------------------------------------------------------------
    # 结果保存目录
    # -------------------------------------------------------------------------
    "save_dir": r"./cwru_2d_cwt_cnn_transformer_results",

    # -------------------------------------------------------------------------
    # 信号切片参数
    # overlap 建议先用 0.0，曲线更真实，也减少相邻窗口泄漏。
    # 如果样本太少，可以改成 0.5。
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
    # 每个文件最多取多少个片段
    # None 表示全部使用。
    # 调试时可以改成 100。
    # -------------------------------------------------------------------------
    "max_segments_per_file": None,

    # -------------------------------------------------------------------------
    # 缓存
    # 第一次建议 False 或 True 都可以。
    # 如果改了 seg_len、overlap、img_size、num_scales、work_condition，
    # 会自动生成不同缓存名。
    # -------------------------------------------------------------------------
    "use_cache": True,

    # -------------------------------------------------------------------------
    # 数据划分
    # time_block:
    #   每个文件前段训练，后段验证，中间留 gap，减少窗口泄漏。
    #
    # random:
    #   所有窗口随机划分，不推荐作为最终实验，但调试方便。
    # -------------------------------------------------------------------------
    "split_mode": "time_block",
    "val_ratio": 0.2,
    "split_gap_segments": 2,

    # -------------------------------------------------------------------------
    # 训练参数
    # -------------------------------------------------------------------------
    "batch_size": 64,
    "epochs": 100,
    "lr": 3e-4,
    "min_lr": 1e-6,
    "warmup_epochs": 5,
    "weight_decay": 1e-4,

    # -------------------------------------------------------------------------
    # 早停
    # -------------------------------------------------------------------------
    "enable_early_stop": True,
    "patience": 15,
    "min_delta": 1e-4,

    # -------------------------------------------------------------------------
    # 稳定训练相关参数
    # -------------------------------------------------------------------------
    "label_smooth": 0.05,
    "dropout": 0.25,
    "grad_clip": 1.0,

    # -------------------------------------------------------------------------
    # 多任务损失权重
    # -------------------------------------------------------------------------
    "loss_w_binary": 1.0,
    "loss_w_type": 1.0,
    "loss_w_severity": 1.0,

    # -------------------------------------------------------------------------
    # 模型规模
    # -------------------------------------------------------------------------
    "d_model": 96,
    "nhead": 4,
    "num_transformer_layers": 2,

    # -------------------------------------------------------------------------
    # 随机种子与 DataLoader
    # Windows 下 num_workers 建议先用 0
    # -------------------------------------------------------------------------
    "seed": 42,
    "num_workers": 0,
}


BINARY_NAMES = ["Normal", "Fault"]
TYPE_NAMES = ["Ball", "InnerRace", "OuterRace"]
SEVERITY_NAMES = ["轻度", "中度", "重度"]

FINAL_NAMES = [
    "Normal",
    "Ball-轻度",
    "Ball-中度",
    "Ball-重度",
    "InnerRace-轻度",
    "InnerRace-中度",
    "InnerRace-重度",
    "OuterRace-轻度",
    "OuterRace-中度",
    "OuterRace-重度",
]


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


def safe_condition_name(name):
    return name.replace("\\", "_").replace("/", "_").replace(":", "_")


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


def parse_label_from_path(fpath, condition_dir):
    """
    从文件路径中解析：
        y_binary:
            0 = Normal
            1 = Fault

        y_type:
            -1 = Normal
             0 = Ball
             1 = InnerRace
             2 = OuterRace

        y_severity:
            -1 = Normal
             0 = 007 轻度
             1 = 014 中度
             2 = 021 重度
    """
    rel = os.path.relpath(fpath, condition_dir)
    text = normalize_text(rel)

    # -------------------------------------------------------------------------
    # 1. 正常样本
    # -------------------------------------------------------------------------
    if (
        "normal" in text
        or "baseline" in text
        or "正常" in text
    ):
        return 0, -1, -1

    # -------------------------------------------------------------------------
    # 2. 故障类型
    # -------------------------------------------------------------------------
    y_type = None

    # Ball
    if (
        "ball" in text
        or "滚动体" in text
        or "滚珠" in text
        or "b007" in text
        or "b014" in text
        or "b021" in text
        or re.search(r"(^|[/_\-\s])b(007|014|021)", text)
    ):
        y_type = 0

    # InnerRace
    elif (
        "inner" in text
        or "inner race" in text
        or "innerrace" in text
        or "内圈" in text
        or "ir007" in text
        or "ir014" in text
        or "ir021" in text
        or re.search(r"(^|[/_\-\s])ir(007|014|021)", text)
    ):
        y_type = 1

    # OuterRace
    elif (
        "outer" in text
        or "outer race" in text
        or "outerrace" in text
        or "外圈" in text
        or "or007" in text
        or "or014" in text
        or "or021" in text
        or re.search(r"(^|[/_\-\s])or(007|014|021)", text)
    ):
        y_type = 2

    if y_type is None:
        return None

    # -------------------------------------------------------------------------
    # 3. 故障程度
    # -------------------------------------------------------------------------
    y_severity = None

    if "007" in text or "0.007" in text or "轻度" in text or "轻微" in text:
        y_severity = 0
    elif "014" in text or "0.014" in text or "中度" in text:
        y_severity = 1
    elif "021" in text or "0.021" in text or "重度" in text:
        y_severity = 2

    # 如果数据中存在 028，而当前任务只做 007/014/021，则跳过。
    if y_severity is None:
        return None

    return 1, y_type, y_severity


# =============================================================================
# 3. 数据读取与预处理
# =============================================================================
def get_condition_dir(cfg):
    condition_dir = os.path.join(cfg["data_root"], cfg["work_condition"])

    if not os.path.exists(condition_dir):
        print("[错误] 指定工况文件夹不存在：")
        print(condition_dir)

        if os.path.exists(cfg["data_root"]):
            print("\n当前 data_root 下可用文件夹：")
            for name in os.listdir(cfg["data_root"]):
                p = os.path.join(cfg["data_root"], name)
                if os.path.isdir(p):
                    print("  ", name)

        raise FileNotFoundError(condition_dir)

    return condition_dir


def find_mat_files(cfg):
    condition_dir = get_condition_dir(cfg)

    files = []

    for root, _, fnames in os.walk(condition_dir):
        for fname in fnames:
            if fname.lower().endswith(".mat"):
                files.append(os.path.join(root, fname))

    files = sorted(files)

    if len(files) == 0:
        raise ValueError(f"在工况目录中没有找到 .mat 文件：{condition_dir}")

    return condition_dir, files


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

        if "de_time" in key_lower:
            priority = 4
        elif "fe_time" in key_lower:
            priority = 3
        elif "ba_time" in key_lower:
            priority = 2
        else:
            priority = 1

        candidates.append((priority, arr.size, k, arr.astype(np.float32)))

    if len(candidates) == 0:
        raise ValueError(f"无法从文件中读取有效振动信号：{fpath}")

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

    return candidates[0][3], candidates[0][2]


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

    condition = safe_condition_name(cfg["work_condition"])

    cache_name = (
        f"cache_{condition}"
        f"_seg{cfg['seg_len']}"
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
        f"{'二分类':>8} | "
        f"{'故障类型':>10} | "
        f"{'故障程度':>8} | "
        f"{'变量名':>18} | 文件"
    )

    print(header)
    print("-" * len(header))

    for i, r in enumerate(records, 1):
        print(
            f"{i:4d} | "
            f"{r['file_id']:7d} | "
            f"{r['n_segments']:6d} | "
            f"{r['signal_len']:10d} | "
            f"{r['binary_name']:>8} | "
            f"{r['type_name']:>10} | "
            f"{r['severity_name']:>8} | "
            f"{r['signal_key']:>18} | "
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
            "y_binary",
            "binary_name",
            "y_type",
            "type_name",
            "y_severity",
            "severity_name",
            "signal_key",
            "signal_len",
            "n_segments",
        ])

        for i, r in enumerate(records, 1):
            writer.writerow([
                i,
                r["file_id"],
                r["rel_path"],
                r["y_binary"],
                r["binary_name"],
                r["y_type"],
                r["type_name"],
                r["y_severity"],
                r["severity_name"],
                r["signal_key"],
                r["signal_len"],
                r["n_segments"],
            ])


def build_dataset(cfg):
    """
    构建数据集。

    返回：
        X_img:      [N, 1, H, W]
        X_stat:     [N, stat_dim]
        y_binary:   [N]
        y_type:     [N]
        y_severity: [N]
        file_ids:   [N]
        records:    文件级记录
    """
    ensure_dir(cfg["save_dir"])

    cache_path, meta_path, csv_path = get_cache_paths(cfg)

    if cfg["use_cache"] and os.path.exists(cache_path):
        print(f"[缓存] 读取预处理缓存：{cache_path}")

        data = np.load(cache_path)

        if not os.path.exists(meta_path):
            raise ValueError("找到缓存数据，但找不到文件记录 json。请删除缓存后重新运行。")

        with open(meta_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        print_file_records(records)
        print(f"[文件样本数量 CSV] {csv_path}")

        return (
            data["X_img"].astype(np.float32),
            data["X_stat"].astype(np.float32),
            data["y_binary"].astype(np.int64),
            data["y_type"].astype(np.int64),
            data["y_severity"].astype(np.int64),
            data["file_ids"].astype(np.int64),
            records,
        )

    condition_dir, mat_files = find_mat_files(cfg)

    print(f"[数据根目录] {cfg['data_root']}")
    print(f"[当前工况] {cfg['work_condition']}")
    print(f"[当前工况路径] {condition_dir}")
    print(f"[检测到 .mat 文件数量] {len(mat_files)}")
    print("[提示] 开始生成 CWT 小波时频图，首次运行可能较慢。")

    scales = np.arange(1, cfg["num_scales"] + 1)

    X_img = []
    X_stat = []
    y_binary = []
    y_type = []
    y_severity = []
    file_ids = []

    records = []
    skipped = []

    for fidx, fpath in enumerate(mat_files, 1):
        rel_path = os.path.relpath(fpath, condition_dir)

        label = parse_label_from_path(fpath, condition_dir)

        if label is None:
            skipped.append(rel_path)
            print(f"[跳过] 无法解析标签或不是当前三种程度 007/014/021：{rel_path}")
            continue

        cur_binary, cur_type, cur_sev = label

        binary_name = BINARY_NAMES[cur_binary]
        type_name = "Normal" if cur_type < 0 else TYPE_NAMES[cur_type]
        severity_name = "Normal" if cur_sev < 0 else SEVERITY_NAMES[cur_sev]

        try:
            sig, signal_key = load_signal_from_mat(fpath)
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
            "y_binary": int(cur_binary),
            "binary_name": binary_name,
            "y_type": int(cur_type),
            "type_name": type_name,
            "y_severity": int(cur_sev),
            "severity_name": severity_name,
            "signal_key": signal_key,
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
            y_binary.append(cur_binary)
            y_type.append(cur_type)
            y_severity.append(cur_sev)
            file_ids.append(file_id)

        elapsed = time.time() - t0

        print(
            f"[文件 {fidx:03d}/{len(mat_files):03d}] "
            f"file_id={file_id:3d} | "
            f"样本数={n_segments:5d} | "
            f"二分类={binary_name:<6s} | "
            f"类型={type_name:<10s} | "
            f"程度={severity_name:<6s} | "
            f"变量={signal_key:<18s} | "
            f"信号长度={len(sig):8d} | "
            f"耗时={elapsed:6.1f}s | "
            f"{rel_path}"
        )

    if len(X_img) == 0:
        raise ValueError("未构建到任何有效样本，请检查路径、文件命名和标签解析规则。")

    X_img = np.stack(X_img, axis=0).astype(np.float32)
    X_img = X_img[:, None, :, :]

    X_stat = np.vstack(X_stat).astype(np.float32)
    y_binary = np.asarray(y_binary, dtype=np.int64)
    y_type = np.asarray(y_type, dtype=np.int64)
    y_severity = np.asarray(y_severity, dtype=np.int64)
    file_ids = np.asarray(file_ids, dtype=np.int64)

    print_file_records(records)
    save_file_records_csv(records, csv_path)

    print("\n[数据集汇总]")
    print(f"X_img shape       : {X_img.shape}")
    print(f"X_stat shape      : {X_stat.shape}")
    print(f"y_binary 分布     : {Counter(y_binary)}")
    print(f"y_type 分布       : {Counter(y_type)}")
    print(f"y_severity 分布   : {Counter(y_severity)}")
    print(f"file_ids 分布     : {Counter(file_ids)}")
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
            y_binary=y_binary,
            y_type=y_type,
            y_severity=y_severity,
            file_ids=file_ids,
        )

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        print(f"\n[缓存] 已保存：{cache_path}")
        print(f"[缓存元信息] 已保存：{meta_path}")

    return X_img, X_stat, y_binary, y_type, y_severity, file_ids, records


# =============================================================================
# 4. 数据划分
# =============================================================================
def split_by_time_block(records, file_ids, val_ratio=0.2, gap_segments=2):
    """
    每个文件内部按时间顺序划分：
        前面部分 -> 训练集
        后面部分 -> 验证集
        中间留出 gap_segments 个片段不用，降低相邻片段影响。
    """
    train_indices = []
    val_indices = []

    print("\n[时间块划分] 每个文件前段训练、后段验证：")

    for r in records:
        fid = r["file_id"]
        idx = np.where(file_ids == fid)[0]

        if len(idx) == 0:
            continue

        n = len(idx)

        if n < 5:
            train_indices.extend(idx.tolist())
            print(
                f"  file_id={fid:3d} | 样本太少 n={n:4d} | 全部放入训练 | {r['rel_path']}"
            )
            continue

        split = int(round(n * (1.0 - val_ratio)))

        train_end = max(1, split - gap_segments)
        val_start = min(n - 1, split + gap_segments)

        train_part = idx[:train_end]
        val_part = idx[val_start:]

        if len(val_part) == 0:
            val_part = idx[-1:]
            train_part = idx[:-1]

        train_indices.extend(train_part.tolist())
        val_indices.extend(val_part.tolist())

        print(
            f"  file_id={fid:3d} | "
            f"总样本={n:5d} | "
            f"训练={len(train_part):5d} | "
            f"验证={len(val_part):5d} | "
            f"gap={val_start - train_end:2d} | "
            f"二分类={r['binary_name']:<6s} | "
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


def split_random_stratified(y_binary, y_type, y_severity, val_ratio=0.2, seed=42):
    """
    随机分层划分。
    不作为默认方式，因为窗口随机划分可能导致相邻片段泄漏。
    """
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(y_binary))

    key = y_binary * 100 + (y_type + 1) * 10 + (y_severity + 1)
    counts = Counter(key)

    if min(counts.values()) < 2:
        stratify = y_binary
    else:
        stratify = key

    idx_train, idx_val = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )

    return idx_train, idx_val


def split_dataset(cfg, records, file_ids, y_binary, y_type, y_severity):
    if cfg["split_mode"] == "time_block":
        return split_by_time_block(
            records=records,
            file_ids=file_ids,
            val_ratio=cfg["val_ratio"],
            gap_segments=cfg["split_gap_segments"],
        )

    if cfg["split_mode"] == "random":
        return split_random_stratified(
            y_binary=y_binary,
            y_type=y_type,
            y_severity=y_severity,
            val_ratio=cfg["val_ratio"],
            seed=cfg["seed"],
        )

    raise ValueError(f"未知 split_mode: {cfg['split_mode']}")


# =============================================================================
# 5. Dataset
# =============================================================================
class BearingDataset(Dataset):
    def __init__(self, X_img, X_stat, y_binary, y_type, y_severity):
        self.X_img = torch.from_numpy(X_img).float()
        self.X_stat = torch.from_numpy(X_stat).float()
        self.y_binary = torch.from_numpy(y_binary).long()
        self.y_type = torch.from_numpy(y_type).long()
        self.y_severity = torch.from_numpy(y_severity).long()

    def __len__(self):
        return len(self.X_img)

    def __getitem__(self, idx):
        return (
            self.X_img[idx],
            self.X_stat[idx],
            self.y_binary[idx],
            self.y_type[idx],
            self.y_severity[idx],
        )


# =============================================================================
# 6. 模型：2D CNN-Transformer + 统计特征融合
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


class CNNTransformerCWTStats(nn.Module):
    def __init__(
        self,
        img_size=64,
        stat_dim=18,
        d_model=96,
        nhead=4,
        num_layers=2,
        dropout=0.25,
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

        # 第一阶段：Normal / Fault
        self.head_binary = nn.Linear(128, 2)

        # 第二阶段：故障类型，Ball / InnerRace / OuterRace
        self.head_type = nn.Linear(128, 3)

        # 第三阶段：故障程度，007 / 014 / 021
        self.head_severity = nn.Linear(128, 3)

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

        out_binary = self.head_binary(fused)
        out_type = self.head_type(fused)
        out_severity = self.head_severity(fused)

        return out_binary, out_type, out_severity


# =============================================================================
# 7. 损失函数、早停、训练与评估
# =============================================================================
class SmoothCrossEntropy(nn.Module):
    def __init__(self, eps=0.05):
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


def get_lr_for_epoch(epoch, cfg):
    """
    Warmup + Cosine 学习率。
    epoch 从 1 开始。
    """
    base_lr = cfg["lr"]
    min_lr = cfg["min_lr"]
    warmup_epochs = cfg["warmup_epochs"]
    total_epochs = cfg["epochs"]

    if warmup_epochs > 0 and epoch <= warmup_epochs:
        start_factor = 0.2
        factor = start_factor + (1.0 - start_factor) * epoch / warmup_epochs
        return base_lr * factor

    if total_epochs <= warmup_epochs:
        return base_lr

    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    progress = min(max(progress, 0.0), 1.0)

    lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    return lr


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def compute_loss(outputs, y_binary, y_type, y_severity, criterion, cfg):
    out_binary, out_type, out_severity = outputs

    loss_binary = criterion(out_binary, y_binary)

    fault_mask = y_binary == 1

    if fault_mask.sum() > 0:
        loss_type = criterion(out_type[fault_mask], y_type[fault_mask])
        loss_severity = criterion(out_severity[fault_mask], y_severity[fault_mask])
    else:
        loss_type = torch.tensor(0.0, device=out_binary.device)
        loss_severity = torch.tensor(0.0, device=out_binary.device)

    loss = (
        cfg["loss_w_binary"] * loss_binary
        + cfg["loss_w_type"] * loss_type
        + cfg["loss_w_severity"] * loss_severity
    )

    return loss


def run_epoch(model, loader, criterion, optimizer, device, cfg, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    binary_correct = 0

    type_correct = 0
    type_total = 0

    severity_correct = 0
    severity_total = 0

    joint_correct = 0

    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for x_img, x_stat, y_binary, y_type, y_severity in loader:
            x_img = x_img.to(device)
            x_stat = x_stat.to(device)
            y_binary = y_binary.to(device)
            y_type = y_type.to(device)
            y_severity = y_severity.to(device)

            outputs = model(x_img, x_stat)

            loss = compute_loss(
                outputs=outputs,
                y_binary=y_binary,
                y_type=y_type,
                y_severity=y_severity,
                criterion=criterion,
                cfg=cfg,
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["grad_clip"])
                optimizer.step()

            out_binary, out_type, out_severity = outputs

            pred_binary = out_binary.argmax(dim=1)
            pred_type = out_type.argmax(dim=1)
            pred_severity = out_severity.argmax(dim=1)

            bs = x_img.size(0)

            total_loss += loss.item() * bs
            total_samples += bs

            binary_correct += (pred_binary == y_binary).sum().item()

            normal_mask = y_binary == 0
            fault_mask = y_binary == 1

            # 故障类型准确率，只统计真实故障样本
            if fault_mask.sum().item() > 0:
                type_correct += (pred_type[fault_mask] == y_type[fault_mask]).sum().item()
                type_total += fault_mask.sum().item()

                severity_correct += (pred_severity[fault_mask] == y_severity[fault_mask]).sum().item()
                severity_total += fault_mask.sum().item()

                joint_correct += (
                    (pred_binary[fault_mask] == y_binary[fault_mask])
                    & (pred_type[fault_mask] == y_type[fault_mask])
                    & (pred_severity[fault_mask] == y_severity[fault_mask])
                ).sum().item()

            # 正常样本只要判断为 Normal 就算层级诊断正确
            if normal_mask.sum().item() > 0:
                joint_correct += (pred_binary[normal_mask] == y_binary[normal_mask]).sum().item()

    avg_loss = total_loss / max(total_samples, 1)
    binary_acc = binary_correct / max(total_samples, 1)
    type_acc = type_correct / max(type_total, 1)
    severity_acc = severity_correct / max(severity_total, 1)
    joint_acc = joint_correct / max(total_samples, 1)

    return avg_loss, binary_acc, type_acc, severity_acc, joint_acc


def predict_all(model, loader, device):
    model.eval()

    all_y_binary = []
    all_y_type = []
    all_y_sev = []

    all_p_binary = []
    all_p_type = []
    all_p_sev = []

    with torch.no_grad():
        for x_img, x_stat, y_binary, y_type, y_severity in loader:
            x_img = x_img.to(device)
            x_stat = x_stat.to(device)

            out_binary, out_type, out_severity = model(x_img, x_stat)

            all_p_binary.append(out_binary.argmax(dim=1).cpu().numpy())
            all_p_type.append(out_type.argmax(dim=1).cpu().numpy())
            all_p_sev.append(out_severity.argmax(dim=1).cpu().numpy())

            all_y_binary.append(y_binary.numpy())
            all_y_type.append(y_type.numpy())
            all_y_sev.append(y_severity.numpy())

    return {
        "y_binary": np.concatenate(all_y_binary),
        "y_type": np.concatenate(all_y_type),
        "y_sev": np.concatenate(all_y_sev),
        "p_binary": np.concatenate(all_p_binary),
        "p_type": np.concatenate(all_p_type),
        "p_sev": np.concatenate(all_p_sev),
    }


def make_final_label(y_binary, y_type, y_sev):
    """
    最终层级标签：
        0 -> Normal
        1 -> Ball-轻度
        2 -> Ball-中度
        3 -> Ball-重度
        4 -> InnerRace-轻度
        ...
        9 -> OuterRace-重度
    """
    final = np.zeros_like(y_binary, dtype=np.int64)

    fault_mask = y_binary == 1

    final[fault_mask] = 1 + y_type[fault_mask] * 3 + y_sev[fault_mask]

    return final


def make_pred_final_label(p_binary, p_type, p_sev):
    final = np.zeros_like(p_binary, dtype=np.int64)

    fault_mask = p_binary == 1

    final[fault_mask] = 1 + p_type[fault_mask] * 3 + p_sev[fault_mask]

    return final


# =============================================================================
# 8. 可视化
# =============================================================================
def smooth_curve(values, alpha=0.35):
    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        return values

    if np.isnan(values).any():
        values = np.nan_to_num(values, nan=np.nanmean(values))

    smoothed = [values[0]]

    for v in values[1:]:
        smoothed.append(alpha * v + (1.0 - alpha) * smoothed[-1])

    return np.asarray(smoothed)


def save_history_csv(hist, save_dir):
    fpath = os.path.join(save_dir, "history.csv")

    keys = list(hist.keys())
    n = len(hist[keys[0]])

    with open(fpath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(keys)

        for i in range(n):
            writer.writerow([hist[k][i] for k in keys])

    print(f"[训练日志] 已保存：{fpath}")


def plot_loss_curve(hist, save_dir):
    epochs = np.asarray(hist["epoch"])

    plt.figure(figsize=(8, 5))

    plt.plot(epochs, hist["train_loss"], alpha=0.25, color="tab:blue")
    plt.plot(epochs, hist["val_loss"], alpha=0.25, color="tab:orange")

    plt.plot(epochs, hist["train_loss"], label="Train Loss", color="tab:blue", linewidth=2)
    plt.plot(epochs, hist["val_loss"], label="Val Loss", color="tab:orange", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("训练与验证损失曲线")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(save_path, dpi=180)
    plt.close()

    print(f"[损失曲线] 已保存：{save_path}")


def plot_accuracy_curve(hist, save_dir):
    epochs = np.asarray(hist["epoch"])

    plt.figure(figsize=(9, 6))

    plt.plot(epochs, hist["train_binary_acc"], label="Train Normal/Fault Acc", linewidth=2)
    plt.plot(epochs, hist["val_binary_acc"], label="Val Normal/Fault Acc", linewidth=2)

    plt.plot(epochs, hist["train_type_acc"], label="Train Type Acc", linewidth=2)
    plt.plot(epochs, hist["val_type_acc"], label="Val Type Acc", linewidth=2)

    plt.plot(epochs, hist["train_severity_acc"], label="Train Severity Acc", linewidth=2)
    plt.plot(epochs, hist["val_severity_acc"], label="Val Severity Acc", linewidth=2)

    plt.plot(epochs, hist["train_joint_acc"], label="Train Joint Acc", linewidth=2, linestyle="--")
    plt.plot(epochs, hist["val_joint_acc"], label="Val Joint Acc", linewidth=2, linestyle="--")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("训练与验证准确率曲线")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, "accuracy_curve.png")
    plt.savefig(save_path, dpi=180)
    plt.close()

    print(f"[准确率曲线] 已保存：{save_path}")


def plot_lr_curve(hist, save_dir):
    epochs = np.asarray(hist["epoch"])

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, hist["lr"], linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("学习率变化曲线")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "lr_curve.png")
    plt.savefig(save_path, dpi=180)
    plt.close()

    print(f"[学习率曲线] 已保存：{save_path}")


def plot_confusion(y_true, y_pred, class_names, title, save_path):
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 7))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()

    plt.xticks(labels, class_names, rotation=35, ha="right")
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


# =============================================================================
# 9. 主流程
# =============================================================================
def main():
    ensure_dir(CFG["save_dir"])

    print("=" * 110)
    print("CWRU 轴承故障诊断：2D CNN-Transformer + CWT 小波时频图 + 统计特征融合")
    print("层级任务：Normal/Fault -> 故障类型 -> 故障程度")
    print("=" * 110)
    print(f"[设备] {DEVICE}")
    print(f"[数据根目录] {CFG['data_root']}")
    print(f"[当前工况] {CFG['work_condition']}")
    print(f"[保存目录] {os.path.abspath(CFG['save_dir'])}")
    print(f"[窗口长度] {CFG['seg_len']}")
    print(f"[重叠率] {CFG['overlap']}")
    print(f"[CWT] wavelet={CFG['wavelet']} | scales={CFG['num_scales']} | image={CFG['img_size']}x{CFG['img_size']}")
    print(f"[划分方式] {CFG['split_mode']}")
    print(f"[训练轮数] {CFG['epochs']}")
    print(f"[学习率] {CFG['lr']}")
    print(f"[标签平滑] {CFG['label_smooth']}")
    print(f"[早停] enable={CFG['enable_early_stop']} | patience={CFG['patience']}")
    print("=" * 110)

    # -------------------------------------------------------------------------
    # 1. 构建数据集
    # -------------------------------------------------------------------------
    print("\n[1/7] 加载数据并生成 CWT 小波时频图与统计特征...")
    X_img, X_stat, y_binary, y_type, y_severity, file_ids, records = build_dataset(CFG)

    # -------------------------------------------------------------------------
    # 2. 划分训练集和验证集
    # -------------------------------------------------------------------------
    print("\n[2/7] 划分训练集和验证集...")
    idx_train, idx_val = split_dataset(
        cfg=CFG,
        records=records,
        file_ids=file_ids,
        y_binary=y_binary,
        y_type=y_type,
        y_severity=y_severity,
    )

    X_img_train = X_img[idx_train]
    X_img_val = X_img[idx_val]

    X_stat_train = X_stat[idx_train]
    X_stat_val = X_stat[idx_val]

    y_binary_train = y_binary[idx_train]
    y_binary_val = y_binary[idx_val]

    y_type_train = y_type[idx_train]
    y_type_val = y_type[idx_val]

    y_sev_train = y_severity[idx_train]
    y_sev_val = y_severity[idx_val]

    # -------------------------------------------------------------------------
    # 3. 统计特征标准化
    # -------------------------------------------------------------------------
    print("\n[3/7] 统计特征标准化...")

    scaler = StandardScaler()
    X_stat_train = scaler.fit_transform(X_stat_train).astype(np.float32)
    X_stat_val = scaler.transform(X_stat_val).astype(np.float32)

    np.savez(
        os.path.join(CFG["save_dir"], "stat_scaler.npz"),
        mean=scaler.mean_,
        scale=scaler.scale_,
    )

    print(f"统计特征维度: {X_stat_train.shape[1]}")
    print(f"训练集样本数: {len(idx_train)}")
    print(f"验证集样本数: {len(idx_val)}")
    print(f"训练集 y_binary 分布: {Counter(y_binary_train)}")
    print(f"验证集 y_binary 分布: {Counter(y_binary_val)}")
    print(f"训练集 y_type 分布: {Counter(y_type_train)}")
    print(f"验证集 y_type 分布: {Counter(y_type_val)}")
    print(f"训练集 y_severity 分布: {Counter(y_sev_train)}")
    print(f"验证集 y_severity 分布: {Counter(y_sev_val)}")

    train_ds = BearingDataset(
        X_img_train,
        X_stat_train,
        y_binary_train,
        y_type_train,
        y_sev_train,
    )

    val_ds = BearingDataset(
        X_img_val,
        X_stat_val,
        y_binary_val,
        y_type_val,
        y_sev_val,
    )

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

    model = CNNTransformerCWTStats(
        img_size=CFG["img_size"],
        stat_dim=X_stat_train.shape[1],
        d_model=CFG["d_model"],
        nhead=CFG["nhead"],
        num_layers=CFG["num_transformer_layers"],
        dropout=CFG["dropout"],
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"d_model: {CFG['d_model']}")
    print(f"Transformer 层数: {CFG['num_transformer_layers']}")
    print(f"模型可训练参数量: {total_params:,}")

    criterion = SmoothCrossEntropy(eps=CFG["label_smooth"])

    optimizer = optim.AdamW(
        model.parameters(),
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"],
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
        "lr": [],
        "train_loss": [],
        "val_loss": [],
        "train_binary_acc": [],
        "val_binary_acc": [],
        "train_type_acc": [],
        "val_type_acc": [],
        "train_severity_acc": [],
        "val_severity_acc": [],
        "train_joint_acc": [],
        "val_joint_acc": [],
    }

    header = (
        f"{'Epoch':>5} | {'LR':>9} | "
        f"{'TrLoss':>8} | {'ValLoss':>8} | "
        f"{'Tr_Bin':>8} | {'Val_Bin':>8} | "
        f"{'Tr_Type':>8} | {'Val_Type':>8} | "
        f"{'Tr_Sev':>8} | {'Val_Sev':>8} | "
        f"{'Tr_Joint':>8} | {'Val_Joint':>9} | "
        f"{'ES':>3} | {'Time':>6}"
    )

    print(header)
    print("-" * len(header))

    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()

        current_lr = get_lr_for_epoch(epoch, CFG)
        set_optimizer_lr(optimizer, current_lr)

        train_loss, train_bin_acc, train_type_acc, train_sev_acc, train_joint_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
            cfg=CFG,
            train=True,
        )

        val_loss, val_bin_acc, val_type_acc, val_sev_acc, val_joint_acc = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=DEVICE,
            cfg=CFG,
            train=False,
        )

        hist["epoch"].append(epoch)
        hist["lr"].append(current_lr)
        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["train_binary_acc"].append(train_bin_acc)
        hist["val_binary_acc"].append(val_bin_acc)
        hist["train_type_acc"].append(train_type_acc)
        hist["val_type_acc"].append(val_type_acc)
        hist["train_severity_acc"].append(train_sev_acc)
        hist["val_severity_acc"].append(val_sev_acc)
        hist["train_joint_acc"].append(train_joint_acc)
        hist["val_joint_acc"].append(val_joint_acc)

        stop, improved = early_stopper.step(val_loss, model, epoch)

        if not CFG["enable_early_stop"]:
            stop = False

        elapsed = time.time() - t0
        mark = "★" if improved else " "

        print(
            f"{epoch:5d} | {current_lr:9.2e} | "
            f"{train_loss:8.4f} | {val_loss:8.4f} | "
            f"{train_bin_acc*100:7.2f}% | {val_bin_acc*100:7.2f}% | "
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

    if not os.path.exists(best_model_path):
        torch.save(model.state_dict(), best_model_path)

    # -------------------------------------------------------------------------
    # 6. 加载最优模型并评估
    # -------------------------------------------------------------------------
    print("\n[6/7] 加载最优模型并评估...")

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    pred = predict_all(model, val_loader, DEVICE)

    print("\n" + "=" * 100)
    print("第一阶段：Normal/Fault 分类报告")
    print("=" * 100)
    print(
        classification_report(
            pred["y_binary"],
            pred["p_binary"],
            labels=[0, 1],
            target_names=BINARY_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    fault_mask = pred["y_binary"] == 1

    print("\n" + "=" * 100)
    print("第二阶段：故障类型分类报告，仅统计真实故障样本")
    print("=" * 100)
    print(
        classification_report(
            pred["y_type"][fault_mask],
            pred["p_type"][fault_mask],
            labels=[0, 1, 2],
            target_names=TYPE_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    print("\n" + "=" * 100)
    print("第三阶段：故障程度分类报告，仅统计真实故障样本")
    print("=" * 100)
    print(
        classification_report(
            pred["y_sev"][fault_mask],
            pred["p_sev"][fault_mask],
            labels=[0, 1, 2],
            target_names=SEVERITY_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    y_final = make_final_label(
        pred["y_binary"],
        pred["y_type"],
        pred["y_sev"],
    )

    p_final = make_pred_final_label(
        pred["p_binary"],
        pred["p_type"],
        pred["p_sev"],
    )

    final_acc = (y_final == p_final).mean()

    print("\n" + "=" * 100)
    print(f"最终层级联合准确率：{final_acc * 100:.2f}%")
    print("规则：")
    print("  Normal 样本：预测为 Normal 即正确；")
    print("  Fault 样本：必须 Normal/Fault、故障类型、故障程度全部正确才算正确。")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # 7. 保存图像结果
    # -------------------------------------------------------------------------
    print("\n[7/7] 保存曲线和混淆矩阵...")

    plot_loss_curve(hist, CFG["save_dir"])
    plot_accuracy_curve(hist, CFG["save_dir"])
    plot_lr_curve(hist, CFG["save_dir"])

    plot_confusion(
        y_true=pred["y_binary"],
        y_pred=pred["p_binary"],
        class_names=BINARY_NAMES,
        title="Normal/Fault 混淆矩阵",
        save_path=os.path.join(CFG["save_dir"], "cm_binary.png"),
    )

    plot_confusion(
        y_true=pred["y_type"][fault_mask],
        y_pred=pred["p_type"][fault_mask],
        class_names=TYPE_NAMES,
        title="故障类型混淆矩阵",
        save_path=os.path.join(CFG["save_dir"], "cm_type.png"),
    )

    plot_confusion(
        y_true=pred["y_sev"][fault_mask],
        y_pred=pred["p_sev"][fault_mask],
        class_names=SEVERITY_NAMES,
        title="故障程度混淆矩阵",
        save_path=os.path.join(CFG["save_dir"], "cm_severity.png"),
    )

    plot_confusion(
        y_true=y_final,
        y_pred=p_final,
        class_names=FINAL_NAMES,
        title="最终层级诊断混淆矩阵",
        save_path=os.path.join(CFG["save_dir"], "cm_final_hierarchical.png"),
    )

    print("\n" + "=" * 100)
    print("训练与评估完成")
    print(f"结果保存目录：{os.path.abspath(CFG['save_dir'])}")
    print("主要输出文件：")
    print("  best_model.pth                    最优模型权重")
    print("  history.csv                       每轮训练日志")
    print("  loss_curve.png                    训练/验证损失曲线")
    print("  accuracy_curve.png                准确率曲线")
    print("  lr_curve.png                      学习率曲线")
    print("  cm_binary.png                     Normal/Fault 混淆矩阵")
    print("  cm_type.png                       故障类型混淆矩阵")
    print("  cm_severity.png                   故障程度混淆矩阵")
    print("  cm_final_hierarchical.png         最终层级诊断混淆矩阵")
    print("  stat_scaler.npz                   统计特征标准化参数")
    print("  *_file_sample_counts.csv          每个文件的样本数量")
    print("=" * 100)


if __name__ == "__main__":
    main()
