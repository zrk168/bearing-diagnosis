import os
import re
import warnings

import numpy as np
import torch
import pywt

from PIL import Image
from scipy.io import loadmat

warnings.filterwarnings("ignore")


# ===================== 配置区 =====================
DATA_ROOT = r"E:\柱塞泵\CWRU轴承数据\cwru_data"

CONFIG = {
    # 三路原始数据文件夹
    "x_dir": os.path.join(DATA_ROOT, "0HP_x"),
    "y_dir": os.path.join(DATA_ROOT, "0HP_y"),
    "z_dir": os.path.join(DATA_ROOT, "0HP_z"),

    # 输出 RGB 图像文件夹
    "rgb_image_dir": os.path.join(DATA_ROOT, "0HP_rgb_images"),

    # 输出训练缓存文件夹
    "cache_dir": os.path.join(DATA_ROOT, "0HP_rgb_cache"),

    # 图像大小
    "image_size": 64,

    # 每个样本截取的数据长度
    "window_size": 2048,

    # 滑动步长
    "step_size": 1024,

    # 每个原始文件最多生成多少张图
    "max_segments_per_file": 200,

    # CWT 小波参数
    "wavelet": "morl",
    "num_scales": 64,

    # 是否重新生成
    "force_rebuild": True,

    # 如果 .mat 文件里变量名固定，可以写这里，例如 "X100_DE_time"
    # 如果不确定，保持 None，程序会自动找最长的一维数值变量
    "mat_variable_name": None,
}


os.makedirs(CONFIG["rgb_image_dir"], exist_ok=True)
os.makedirs(CONFIG["cache_dir"], exist_ok=True)


# ===================== 标签规则 =====================
"""
标签格式：
    [is_fault, fault_type, severity]

is_fault:
    0 = 正常
    1 = 故障

fault_type:
    0 = 内圈
    1 = 滚动体
    2 = 外圈

severity:
    0 = 轻微
    1 = 中度
    2 = 重度
"""

ID_LABEL_MAP = {
    # 正常
    "100": (0, 0, 0),

    # 内圈故障
    "105": (1, 0, 0),
    "118": (1, 0, 1),
    "130": (1, 0, 2),

    # 滚动体故障
    "169": (1, 1, 0),
    "185": (1, 1, 1),
    "197": (1, 1, 2),

    # 外圈故障
    "222": (1, 2, 0),
    "234": (1, 2, 1),
    "246": (1, 2, 2),
}

KEYWORD_LABEL_MAP = {
    "normal": (0, 0, 0),
    "healthy": (0, 0, 0),
    "正常": (0, 0, 0),

    "ir007": (1, 0, 0),
    "ir014": (1, 0, 1),
    "ir021": (1, 0, 2),

    "b007": (1, 1, 0),
    "b014": (1, 1, 1),
    "b021": (1, 1, 2),

    "or007": (1, 2, 0),
    "or014": (1, 2, 1),
    "or021": (1, 2, 2),

    "inner007": (1, 0, 0),
    "inner014": (1, 0, 1),
    "inner021": (1, 0, 2),

    "ball007": (1, 1, 0),
    "ball014": (1, 1, 1),
    "ball021": (1, 1, 2),

    "outer007": (1, 2, 0),
    "outer014": (1, 2, 1),
    "outer021": (1, 2, 2),
}


# 如果你的文件名完全不含 100/105/118/130/007/014/021 等标签信息，
# 可以手动在这里写标签。
# 例如：
# MANUAL_LABELS = {
#     "cwt_outer_race_examples_real": (1, 2, 0),
# }
MANUAL_LABELS = {}


# ===================== 工具函数 =====================
def normalize_name(name):
    name = name.lower()
    name = name.replace(" ", "")
    name = name.replace("_", "")
    name = name.replace("-", "")
    name = name.replace("（", "(").replace("）", ")")
    return name


def infer_label(stem):
    if stem in MANUAL_LABELS:
        return MANUAL_LABELS[stem]

    clean = normalize_name(stem)

    for key, label in ID_LABEL_MAP.items():
        if key in clean:
            return label

    for key, label in KEYWORD_LABEL_MAP.items():
        if key in clean:
            return label

    normal_keys = ["normal", "healthy", "health", "正常", "无故障"]
    if any(key in clean for key in normal_keys):
        return (0, 0, 0)

    fault_type = None
    severity = None

    if any(key in clean for key in ["ir", "inner", "innerrace", "内圈", "内环"]):
        fault_type = 0
    elif any(key in clean for key in ["ball", "rolling", "滚动体", "滚珠", "球"]):
        fault_type = 1
    elif any(key in clean for key in ["or", "outer", "outerrace", "外圈", "外环"]):
        fault_type = 2

    if any(key in clean for key in ["007", "light", "slight", "minor", "轻微", "轻度"]):
        severity = 0
    elif any(key in clean for key in ["014", "medium", "middle", "中度"]):
        severity = 1
    elif any(key in clean for key in ["021", "heavy", "severe", "serious", "重度", "严重"]):
        severity = 2

    if fault_type is not None and severity is not None:
        return (1, fault_type, severity)

    return None


def scan_data_files(folder):
    exts = {".mat", ".csv", ".txt", ".npy", ".npz"}
    files = {}

    for name in os.listdir(folder):
        path = os.path.join(folder, name)

        if not os.path.isfile(path):
            continue

        stem, ext = os.path.splitext(name)

        if ext.lower() in exts:
            files[stem] = path

    return files


def load_signal(path, mat_variable_name=None):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        data = np.load(path)
        return to_1d_signal(data)

    if ext == ".npz":
        data = np.load(path)
        best = None
        best_len = 0

        for key in data.files:
            arr = np.asarray(data[key])
            sig = to_1d_signal(arr)
            if len(sig) > best_len:
                best = sig
                best_len = len(sig)

        if best is None:
            raise ValueError(f"npz 文件中未找到有效数据: {path}")

        return best

    if ext == ".csv":
        data = np.loadtxt(path, delimiter=",")
        return to_1d_signal(data)

    if ext == ".txt":
        try:
            data = np.loadtxt(path)
        except Exception:
            data = np.loadtxt(path, delimiter=",")
        return to_1d_signal(data)

    if ext == ".mat":
        mat = loadmat(path)

        if mat_variable_name is not None:
            if mat_variable_name not in mat:
                raise KeyError(f"{path} 中不存在变量 {mat_variable_name}")
            return to_1d_signal(mat[mat_variable_name])

        best = None
        best_len = 0
        best_key = None

        for key, value in mat.items():
            if key.startswith("__"):
                continue

            arr = np.asarray(value)

            if not np.issubdtype(arr.dtype, np.number):
                continue

            sig = to_1d_signal(arr)

            if len(sig) > best_len:
                best = sig
                best_len = len(sig)
                best_key = key

        if best is None:
            raise ValueError(f"mat 文件中未找到有效一维数值信号: {path}")

        print(f"  MAT变量自动选择: {os.path.basename(path)} -> {best_key}, 长度={best_len}")

        return best

    raise ValueError(f"不支持的文件格式: {path}")


def to_1d_signal(data):
    arr = np.asarray(data, dtype=np.float32)

    if arr.ndim == 0:
        raise ValueError("数据为空或不是有效数组")

    if arr.ndim == 1:
        sig = arr

    elif arr.ndim == 2:
        if 1 in arr.shape:
            sig = arr.reshape(-1)
        else:
            # 多列数据时默认取第一列
            sig = arr[:, 0]

    else:
        sig = arr.reshape(-1)

    sig = sig.astype(np.float32)
    sig = sig[np.isfinite(sig)]

    if len(sig) == 0:
        raise ValueError("信号中没有有效数值")

    return sig


def standardize_signal(sig):
    sig = np.asarray(sig, dtype=np.float32)
    sig = sig - np.mean(sig)
    std = np.std(sig)

    if std < 1e-8:
        return sig

    return sig / std


def cwt_to_image(signal, image_size=64, wavelet="morl", num_scales=64):
    signal = standardize_signal(signal)

    scales = np.arange(1, num_scales + 1)
    coeffs, _ = pywt.cwt(signal, scales, wavelet)

    cwt_abs = np.abs(coeffs).astype(np.float32)

    low = np.percentile(cwt_abs, 1)
    high = np.percentile(cwt_abs, 99)

    cwt_abs = np.clip(cwt_abs, low, high)
    cwt_abs = (cwt_abs - low) / (high - low + 1e-8)

    img = Image.fromarray((cwt_abs * 255).astype(np.uint8))
    img = img.resize((image_size, image_size), Image.BILINEAR)

    return np.asarray(img, dtype=np.uint8)


def make_rgb_image(x_seg, y_seg, z_seg):
    r = cwt_to_image(
        x_seg,
        image_size=CONFIG["image_size"],
        wavelet=CONFIG["wavelet"],
        num_scales=CONFIG["num_scales"],
    )

    g = cwt_to_image(
        y_seg,
        image_size=CONFIG["image_size"],
        wavelet=CONFIG["wavelet"],
        num_scales=CONFIG["num_scales"],
    )

    b = cwt_to_image(
        z_seg,
        image_size=CONFIG["image_size"],
        wavelet=CONFIG["wavelet"],
        num_scales=CONFIG["num_scales"],
    )

    rgb = np.stack([r, g, b], axis=-1)
    return rgb


def label_to_folder(label):
    is_fault, fault_type, severity = label

    if is_fault == 0:
        return "normal"

    type_names = {
        0: "inner",
        1: "ball",
        2: "outer",
    }

    severity_names = {
        0: "light",
        1: "medium",
        2: "heavy",
    }

    return f"{type_names[fault_type]}_{severity_names[severity]}"


def remove_old_outputs():
    cache_file = os.path.join(CONFIG["cache_dir"], "dataset_rgb.pt")

    if CONFIG["force_rebuild"] and os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"已删除旧缓存: {cache_file}")


# ===================== 主生成流程 =====================
def generate_rgb_dataset():
    print("=" * 80)
    print("开始从原始数据生成 RGB-CWT 图像")
    print("=" * 80)

    remove_old_outputs()

    x_files = scan_data_files(CONFIG["x_dir"])
    y_files = scan_data_files(CONFIG["y_dir"])
    z_files = scan_data_files(CONFIG["z_dir"])

    print(f"x 通道原始数据文件数: {len(x_files)}")
    print(f"y 通道原始数据文件数: {len(y_files)}")
    print(f"z 通道原始数据文件数: {len(z_files)}")

    common_stems = sorted(set(x_files.keys()) & set(y_files.keys()) & set(z_files.keys()))

    print(f"三通道同名匹配文件数: {len(common_stems)}")

    if not common_stems:
        raise RuntimeError(
            "三个文件夹中没有找到同名数据文件。\n"
            "请确认 0HP_x、0HP_y、0HP_z 中同一条样本的文件名完全一致。"
        )

    all_images = []
    all_labels = []

    skipped_no_label = 0
    skipped_error = 0
    total_saved = 0

    for file_index, stem in enumerate(common_stems, start=1):
        print("-" * 80)
        print(f"[{file_index}/{len(common_stems)}] 正在处理: {stem}")

        label = infer_label(stem)

        if label is None:
            skipped_no_label += 1
            print(f"  跳过：无法从文件名识别标签 -> {stem}")
            print("  解决方法：重命名文件，或在 MANUAL_LABELS 中手动指定标签。")
            continue

        try:
            x_signal = load_signal(x_files[stem], CONFIG["mat_variable_name"])
            y_signal = load_signal(y_files[stem], CONFIG["mat_variable_name"])
            z_signal = load_signal(z_files[stem], CONFIG["mat_variable_name"])

            min_len = min(len(x_signal), len(y_signal), len(z_signal))

            x_signal = x_signal[:min_len]
            y_signal = y_signal[:min_len]
            z_signal = z_signal[:min_len]

            if min_len < CONFIG["window_size"]:
                print(f"  跳过：信号长度 {min_len} 小于窗口长度 {CONFIG['window_size']}")
                skipped_error += 1
                continue

            class_folder = label_to_folder(label)
            save_dir = os.path.join(CONFIG["rgb_image_dir"], class_folder)
            os.makedirs(save_dir, exist_ok=True)

            starts = list(range(0, min_len - CONFIG["window_size"] + 1, CONFIG["step_size"]))

            if CONFIG["max_segments_per_file"] is not None:
                starts = starts[:CONFIG["max_segments_per_file"]]

            print(f"  信号长度: {min_len}")
            print(f"  生成片段数: {len(starts)}")
            print(f"  标签: {label} -> {class_folder}")

            for seg_idx, start in enumerate(starts):
                end = start + CONFIG["window_size"]

                x_seg = x_signal[start:end]
                y_seg = y_signal[start:end]
                z_seg = z_signal[start:end]

                rgb = make_rgb_image(x_seg, y_seg, z_seg)

                image_name = f"{stem}_seg{seg_idx:04d}.png"
                image_path = os.path.join(save_dir, image_name)

                Image.fromarray(rgb).save(image_path)

                tensor = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
                label_tensor = torch.tensor(label, dtype=torch.long)

                all_images.append(tensor)
                all_labels.append(label_tensor)

                total_saved += 1

        except Exception as e:
            skipped_error += 1
            print(f"  处理失败: {stem}")
            print(f"  错误信息: {e}")

    print("=" * 80)
    print("生成完成")
    print(f"成功保存 RGB 图像数: {total_saved}")
    print(f"无标签跳过文件数: {skipped_no_label}")
    print(f"读取/处理失败文件数: {skipped_error}")
    print(f"RGB 图像目录: {CONFIG['rgb_image_dir']}")

    if not all_images:
        print("没有生成任何有效样本。")
        print("请重点检查：")
        print("1. 文件名是否包含 normal / 100 / 007 / 014 / 021 / IR / B / OR 等标签信息；")
        print("2. 或者在 MANUAL_LABELS 中手动填写标签；")
        print("3. 三个文件夹中同一数据文件名是否完全一致。")
        raise RuntimeError("未生成任何 RGB 样本。")

    x_tensor = torch.stack(all_images)
    y_tensor = torch.stack(all_labels)

    cache_file = os.path.join(CONFIG["cache_dir"], "dataset_rgb.pt")

    torch.save(
        {
            "x": x_tensor,
            "y": y_tensor,
        },
        cache_file,
    )

    print(f"训练缓存已保存: {cache_file}")
    print(f"x shape: {x_tensor.shape}")
    print(f"y shape: {y_tensor.shape}")
    print("=" * 80)


if __name__ == "__main__":
    generate_rgb_dataset()
