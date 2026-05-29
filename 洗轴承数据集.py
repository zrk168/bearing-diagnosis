import os
import numpy as np
import scipy.io as sio
import pandas as pd
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# ================= 配置区 =================
# 基于脚本所在位置自动定位（相对引用）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(BASE_DIR, "CRWU")
DEST_ROOT = os.path.join(BASE_DIR, "Cleaned_Dataset")

# 🎯 仅处理这两个指定文件夹
TARGET_FOLDERS = [
    "12k Drive End Bearing Fault Data",
    "Normal Baseline"
]

WINDOW_SIZE = 1024  # 滑动窗口大小
IQR_FACTOR = 1.5  # 异常窗口剔除阈值
MIN_CLEAN_WINDOWS = 10  # 清洗后最少保留窗口数


# ==========================================

def get_signal_from_mat(mat_path):
    """自动识别 .mat 中的振动信号列"""
    mat = sio.loadmat(mat_path)
    keys = [k for k in mat.keys() if not k.startswith('__')]
    for k in keys:
        arr = mat[k]
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 1:
            return arr.flatten(), k
    return mat[keys[0]].flatten(), keys[0]


def purify_signal(signal):
    """信号级提纯：去直流 + 剔除电气毛刺"""
    signal = signal - np.mean(signal)
    std_val = np.std(signal)
    spike_mask = np.abs(signal) > 5 * std_val
    if np.any(spike_mask):
        signal[spike_mask] = np.median(signal[~spike_mask])
    return signal


def process_file(src_mat, dest_csv):
    """分段 -> 特征计算 -> IQR过滤 -> 保存"""
    try:
        signal, _ = get_signal_from_mat(src_mat)
    except Exception as e:
        print(f"❌ 读取失败 {os.path.basename(src_mat)}: {e}")
        return False

    signal = purify_signal(signal)
    n_windows = len(signal) // WINDOW_SIZE
    if n_windows < 5:
        return False

    rms_list, kurt_list = [], []
    for i in range(n_windows):
        seg = signal[i * WINDOW_SIZE: (i + 1) * WINDOW_SIZE]
        rms_list.append(np.sqrt(np.mean(seg ** 2)))
        kurt_list.append(stats.kurtosis(seg))

    rms_arr, kurt_arr = np.array(rms_list), np.array(kurt_list)

    # IQR 异常窗口过滤
    def iqr_mask(arr):
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        return (arr >= q1 - IQR_FACTOR * iqr) & (arr <= q3 + IQR_FACTOR * iqr)

    valid_mask = iqr_mask(rms_arr) & iqr_mask(kurt_arr)

    clean_windows = [signal[i * WINDOW_SIZE:(i + 1) * WINDOW_SIZE] for i, v in enumerate(valid_mask) if v]
    if len(clean_windows) < MIN_CLEAN_WINDOWS:
        return False

    # 保持原始目录结构输出
    os.makedirs(os.path.dirname(dest_csv), exist_ok=True)
    pd.DataFrame(np.concatenate(clean_windows), columns=['Amplitude']).to_csv(dest_csv, index=False)

    # 同步输出特征表（仅含有效窗口）
    feat_df = pd.DataFrame({
        'Window_ID': np.where(valid_mask)[0],
        'RMS': rms_arr[valid_mask],
        'Kurtosis': kurt_arr[valid_mask]
    })
    feat_df.to_csv(dest_csv.replace('.csv', '_features.csv'), index=False)
    return True


if __name__ == "__main__":
    if not os.path.exists(SRC_ROOT):
        raise FileNotFoundError(f"⚠️ 未找到源目录: {SRC_ROOT}\n请将脚本放在 E:\\柱塞泵\\ 下运行")

    print(f"📂 源目录: {SRC_ROOT}")
    print(f"💾 输出目录: {DEST_ROOT}")
    print(f"🎯 限定处理: {', '.join(TARGET_FOLDERS)}")
    print("🔄 开始提纯...\n")

    total = success = skipped = 0
    for target in TARGET_FOLDERS:
        target_path = os.path.join(SRC_ROOT, target)
        if not os.path.exists(target_path):
            print(f"⚠️  跳过不存在的文件夹: {target}")
            continue

        for root, _, files in os.walk(target_path):
            for f in files:
                if f.endswith('.mat'):
                    src = os.path.join(root, f)
                    rel = os.path.relpath(src, SRC_ROOT)
                    dest = os.path.join(DEST_ROOT, rel.replace('.mat', '.csv'))

                    total += 1
                    print(f"⏳ {rel}", end=" ... ")
                    if process_file(src, dest):
                        success += 1
                        print("✅")
                    else:
                        skipped += 1
                        print("⏭️")

    print("\n" + "=" * 50)
    print(f"📊 提纯完成 | 总计: {total} | 成功: {success} | 跳过: {skipped}")
    print(f"📁 输出路径: {DEST_ROOT}")
    print("=" * 50)
