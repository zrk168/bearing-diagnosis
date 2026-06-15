import os
import re
import shutil
import warnings

import numpy as np
from PIL import Image
from scipy.io import loadmat

warnings.filterwarnings("ignore")

# ================= 配置 =================
DATA_ROOT = r"E:\柱塞泵\CWRU轴承数据\cwru_data"

CONFIG = {
    "x_dir": os.path.join(DATA_ROOT, "0HP_x"),
    "y_dir": os.path.join(DATA_ROOT, "0HP_y"),
    "z_dir": os.path.join(DATA_ROOT, "0HP_z"),
    "output_dir": os.path.join(DATA_ROOT, "0HP_rgb_xyz_one_group"),
    "image_size": 64,
    "window_size": 4096,
    "force_rebuild": True,
    "mat_variable_name": None,
}

PARTS = ("ball", "outer", "inner")
SEVERITIES = ("normal", "light", "medium", "heavy")

TARGET_KEYS = {
    (part, severity)
    for part in PARTS
    for severity in SEVERITIES
}

# CWRU 常用编号
# 97/98/99/100 通常是正常数据，不同编号对应不同负载
ID_LABEL_MAP = {
    "97": ("normal", "normal"),
    "98": ("normal", "normal"),
    "99": ("normal", "normal"),
    "100": ("normal", "normal"),

    "105": ("inner", "light"),
    "118": ("inner", "medium"),
    "130": ("inner", "heavy"),

    "169": ("ball", "light"),
    "185": ("ball", "medium"),
    "197": ("ball", "heavy"),

    "222": ("outer", "light"),
    "234": ("outer", "medium"),
    "246": ("outer", "heavy"),
}

KEYWORD_LABEL_MAP = {
    "normal": ("normal", "normal"),
    "healthy": ("normal", "normal"),

    "ir007": ("inner", "light"),
    "ir014": ("inner", "medium"),
    "ir021": ("inner", "heavy"),
    "inner007": ("inner", "light"),
    "inner014": ("inner", "medium"),
    "inner021": ("inner", "heavy"),

    "b007": ("ball", "light"),
    "b014": ("ball", "medium"),
    "b021": ("ball", "heavy"),
    "ball007": ("ball", "light"),
    "ball014": ("ball", "medium"),
    "ball021": ("ball", "heavy"),

    "or007": ("outer", "light"),
    "or014": ("outer", "medium"),
    "or021": ("outer", "heavy"),
    "outer007": ("outer", "light"),
    "outer014": ("outer", "medium"),
    "outer021": ("outer", "heavy"),
}


def normalize_name(name):
    return (
        name.lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("（", "(")
        .replace("）", ")")
    )


def normalize_number_token(token):
    value = int(token)
    return str(value)


def infer_label(stem):
    clean = normalize_name(stem)

    for key, label in KEYWORD_LABEL_MAP.items():
        if key in clean:
            return label

    numbers = re.findall(r"\d+", stem)
    number_tokens = set()

    for number in numbers:
        try:
            number_tokens.add(normalize_number_token(number))
        except ValueError:
            pass

    for key, label in ID_LABEL_MAP.items():
        if key in number_tokens:
            return label

    return None


def scan_files(folder):
    exts = {".mat", ".csv", ".txt", ".npy", ".npz"}
    result = {}

    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue

        stem, ext = os.path.splitext(name)
        if ext.lower() in exts:
            result[stem] = path

    return result


def to_1d_signal(data):
    arr = np.asarray(data, dtype=np.float32)

    if arr.ndim == 1:
        sig = arr
    elif arr.ndim == 2:
        if 1 in arr.shape:
            sig = arr.reshape(-1)
        else:
            sig = arr[:, 0]
    else:
        sig = arr.reshape(-1)

    sig = sig[np.isfinite(sig)]

    if len(sig) == 0:
        raise ValueError("空信号")

    return sig


def load_signal(path, mat_variable_name=None):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        return to_1d_signal(np.load(path))

    if ext == ".npz":
        data = np.load(path)
        best = None
        best_len = 0

        for key in data.files:
            sig = to_1d_signal(data[key])
            if len(sig) > best_len:
                best = sig
                best_len = len(sig)

        if best is None:
            raise ValueError(f"npz 无有效数据: {path}")

        return best

    if ext == ".csv":
        return to_1d_signal(np.loadtxt(path, delimiter=","))

    if ext == ".txt":
        try:
            return to_1d_signal(np.loadtxt(path))
        except Exception:
            return to_1d_signal(np.loadtxt(path, delimiter=","))

    if ext == ".mat":
        mat = loadmat(path)

        if mat_variable_name is not None:
            if mat_variable_name not in mat:
                raise KeyError(f"{path} 中没有变量 {mat_variable_name}")
            return to_1d_signal(mat[mat_variable_name])

        best = None
        best_len = 0

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

        if best is None:
            raise ValueError(f"mat 无有效数值变量: {path}")

        return best

    raise ValueError(f"不支持格式: {path}")


def minmax_norm(sig):
    sig = np.asarray(sig, dtype=np.float32)
    sig = sig - np.min(sig)

    denom = np.max(sig) - np.min(sig)
    if denom < 1e-8:
        return np.zeros_like(sig, dtype=np.float32)

    return sig / denom


def signal_to_channel(sig, image_size):
    pixels = image_size * image_size

    sig = np.asarray(sig, dtype=np.float32)

    if len(sig) < pixels:
        sig = np.pad(sig, (0, pixels - len(sig)), mode="constant")
    else:
        sig = sig[:pixels]

    sig = minmax_norm(sig)
    channel = sig.reshape(image_size, image_size)

    return (channel * 255).astype(np.uint8)


def xyz_to_rgb(x_seg, y_seg, z_seg):
    r = signal_to_channel(x_seg, CONFIG["image_size"])
    g = signal_to_channel(y_seg, CONFIG["image_size"])
    b = signal_to_channel(z_seg, CONFIG["image_size"])

    rgb = np.stack([r, g, b], axis=-1)
    return rgb


def get_save_targets(label):
    part, severity = label

    if part == "normal":
        return [
            (target_part, "normal")
            for target_part in PARTS
        ]

    return [(part, severity)]


def clear_output():
    if CONFIG["force_rebuild"] and os.path.exists(CONFIG["output_dir"]):
        shutil.rmtree(CONFIG["output_dir"])

    os.makedirs(CONFIG["output_dir"], exist_ok=True)


def generate():
    clear_output()

    x_files = scan_files(CONFIG["x_dir"])
    y_files = scan_files(CONFIG["y_dir"])
    z_files = scan_files(CONFIG["z_dir"])

    common = sorted(set(x_files) & set(y_files) & set(z_files))

    print(f"同名文件数: {len(common)}")

    if not common:
        raise RuntimeError("三个文件夹里没有找到同名文件。")

    completed = set()
    saved = 0

    for idx, stem in enumerate(common, 1):
        if completed == TARGET_KEYS:
            break

        try:
            label = infer_label(stem)

            if label is None:
                print(f"跳过 {stem}: 无法识别标签")
                continue

            save_targets = get_save_targets(label)
            save_targets = [
                target for target in save_targets
                if target not in completed
            ]

            if not save_targets:
                continue

            x = load_signal(x_files[stem], CONFIG["mat_variable_name"])
            y = load_signal(y_files[stem], CONFIG["mat_variable_name"])
            z = load_signal(z_files[stem], CONFIG["mat_variable_name"])

            min_len = min(len(x), len(y), len(z), CONFIG["window_size"])

            if min_len <= 0:
                print(f"跳过 {stem}: 空数据")
                continue

            x_seg = x[:min_len]
            y_seg = y[:min_len]
            z_seg = z[:min_len]

            rgb = xyz_to_rgb(x_seg, y_seg, z_seg)
            image = Image.fromarray(rgb)

            print(f"[{idx}/{len(common)}] {stem} -> 保存 {len(save_targets)} 张")

            for part, severity in save_targets:
                save_dir = os.path.join(CONFIG["output_dir"], part, severity)
                os.makedirs(save_dir, exist_ok=True)

                out_name = f"{part}_{severity}_{stem}.png"
                out_path = os.path.join(save_dir, out_name)

                image.save(out_path)

                completed.add((part, severity))
                saved += 1

        except Exception as e:
            print(f"处理失败 {stem}: {e}")

    print("=" * 60)
    print(f"完成，已输出: {saved} 张")
    print(f"输出目录: {CONFIG['output_dir']}")

    missing = TARGET_KEYS - completed

    if missing:
        print("以下类别没有找到对应数据:")
        for part, severity in sorted(missing):
            print(f"{part}/{severity}")
    else:
        print("12 个类别全部生成完成。")


if __name__ == "__main__":
    generate()
