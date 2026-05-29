import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')

# ================= 配置区 =================
# 根据截图路径修改为你的实际 .mat 文件
MAT_FILE_PATH = r"E:\柱塞泵\CRWU\12k Drive End Bearing Fault Data\Ball\0007\B007_0.mat"
USE_GPU = False  # 振动信号特征回归数据量小，CPU 通常比 GPU 快 3~5 倍
WINDOW_SIZE = 1024  # 滑动窗口大小（CWRU 12kHz 采样常用 1024 点/窗）


# ==========================================

def load_cwru_mat(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到文件: {filepath}")

    mat = sio.loadmat(filepath)
    # 过滤掉 MATLAB 内置元数据
    keys = [k for k in mat.keys() if not k.startswith('__')]
    print(f"📂 文件包含变量: {keys}")

    # 自动寻找振动信号（通常名称含 _time / _DE / _FE，且为二维列向量）
    signal_key = None
    for k in keys:
        arr = mat[k]
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 1:
            if any(tag in k.lower() for tag in ['_time', '_de', '_fe', '_ba']):
                signal_key = k
                break
    if signal_key is None:
        signal_key = keys[0]  # 兜底选择第一个数组

    signal = mat[signal_key].flatten()
    print(f"✅ 成功加载信号变量 '{signal_key}'，长度: {len(signal)}")
    return signal


def compute_rms_trend(signal, window_size=1024):
    """计算滑动窗口 RMS，生成有物理意义的回归趋势"""
    n_windows = len(signal) // window_size
    rms_values = []
    for i in range(n_windows):
        segment = signal[i * window_size: (i + 1) * window_size]
        rms = np.sqrt(np.mean(segment ** 2))
        rms_values.append(rms)
    return np.array(rms_values)


def train_regression(X, y, use_gpu=False):
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print("⚡ 使用 GPU 训练...")
        X_t = torch.tensor(X, dtype=torch.float32, device=device).view(-1, 1)
        y_t = torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1)
        model = nn.Linear(1, 1).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        for epoch in range(300):
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(X_t), y_t)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            y_pred = model(X_t).cpu().numpy().ravel()
        return model.weight.item(), model.bias.item(), y_pred
    else:
        print(" 使用 CPU 训练 (推荐)...")
        model = LinearRegression()
        model.fit(X.reshape(-1, 1), y)
        y_pred = model.predict(X.reshape(-1, 1))
        return model.coef_[0], model.intercept_, y_pred


if __name__ == "__main__":
    # 1. 加载数据
    signal = load_cwru_mat(MAT_FILE_PATH)

    # 2. 提取特征（RMS趋势）
    print("📈 正在计算 RMS 趋势特征...")
    y_rms = compute_rms_trend(signal, WINDOW_SIZE)
    X_rms = np.arange(len(y_rms))  # 窗口索引作为自变量

    # 3. 线性回归
    slope, intercept, y_pred = train_regression(X_rms, y_rms, USE_GPU)

    # 4. 绘图
    plt.figure(figsize=(10, 7))

    # 上图：原始信号片段（仅画前 5000 点避免卡顿）
    plt.subplot(2, 1, 1)
    plt.plot(signal[:5000], linewidth=0.5, color='gray')
    plt.title("原始振动信号 (前 5000 采样点)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)

    # 下图：RMS 趋势 + 线性回归
    plt.subplot(2, 1, 2)
    plt.scatter(X_rms, y_rms, alpha=0.7, label='RMS 特征点', s=20, edgecolor='k')
    plt.plot(X_rms, y_pred, color='red', linewidth=2,
             label=f'线性回归: y = {slope:.4f}x + {intercept:.4f}')
    plt.title("RMS 趋势线性回归曲线")
    plt.xlabel("时间窗口索引")
    plt.ylabel("RMS 值")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 5. 输出评估指标
    r2 = 1 - np.sum((y_rms - y_pred) ** 2) / np.sum((y_rms - np.mean(y_rms)) ** 2)
    print(f"✅ 回归完成 | 斜率: {slope:.6f} | 截距: {intercept:.6f} | R²: {r2:.4f}")
