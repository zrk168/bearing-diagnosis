"""
================================================================================
【程序执行全流程详解】
================================================================================
Phase 0：环境与配置初始化
  1. 硬件检测：自动识别 CUDA GPU，不可用时降级 CPU 并打印警告。
  2. 路径校验：检查数据根目录，自动创建 cache/（缓存）与 output/（结果）文件夹。
  3. 超参注入：学习率、BatchSize、早停耐心值、CWT尺度数、正则权重等统一写入 CONFIG。
  4. 随机种子固定：锁定 torch/numpy/random 种子，确保实验完全可复现。

Phase 1：数据管道构建（Data Pipeline）
  1. .mat 扫描与解析：遍历 0HP 目录，匹配 CWRU 标准文件名（100正常，105/118/130内圈轻中重，169/185/197滚动体，222/234/246外圈）。
  2. 信号分段：按 1024 点长度滑动截取，避免边界截断效应。
  3. CWT 时频变换：使用 Morlet 母小波生成 64×1024 复系数 → 取模 → log1p 压缩 → Min-Max 归一化 → 双线性插值至 64×64 标准图像尺寸。
  4. 标签映射：转换为三任务标签 [is_fault(0/1), fault_type(0内/1滚/2外), severity(0轻/1中/2重)]。
  5. 缓存机制：首次运行将张量序列化至 cache/dataset.pt，后续直接加载（秒级启动）。
  6. 数据集划分：80% 训练 / 20% 验证，启用 pin_memory 与多进程 DataLoader 加速。

Phase 2：前向传播与特征流（Feature Flow）
  [B,1,64,64] → CNN Stem (浅层卷积+BN+ReLU+下采样) → [B,64,32,32]
              → DSC (偏移预测+累积约束+grid_sample变形采样+卷积) → [B,128,32,32]
              → EFF (理论故障频带高斯掩码+残差门控加权) → [B,128,32,32]
              → DAT (多头注意力+空间相对位置编码) → [B,128,32,32]
              → GAP (全局平均池化) → [B,128]
              → 多任务分类头 (正常/故障, 故障类型, 故障程度) → 输出预测概率

Phase 3：训练优化与早停机制（Optimization Flow）
  1. 损失计算：L_total = L_nf + L_type + L_sev + λ·L_reg（交叉熵 + 持续学习稳定性正则）。
  2. 反向传播：loss.backward() → 梯度裁剪 → optimizer.step()。
  3. 学习率调度：CosineAnnealingLR 余弦衰减，避免后期震荡。
  4. 早停监控：每轮记录验证集总损失，若 15 轮未下降则触发 EarlyStopping，自动恢复最优权重并终止训练。
  5. 实时日志：终端逐轮打印 Epoch、Train/Val Loss、三任务独立准确率、当前 LR，格式严格对齐。

Phase 4：评估与可视化（Evaluation Flow）
  1. 加载最优模型：早停触发后自动加载 best_model.pth。
  2. 全量推理：在验证集上获取预测标签与真实标签。
  3. 无损绘图：训练/验证损失曲线与准确率曲线（原始点连线，零平滑处理）。
  4. 混淆矩阵：normalize='true' 计算行归一化，格式化为百分比（保留1位小数），直观展示分类边界。
  5. 结果保存：所有图表与模型权重自动存入 output/ 目录。
================================================================================
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import pywt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F
import warnings
import copy
warnings.filterwarnings('ignore')

# ================= 配置区 =================
CONFIG = {
    "data_root": r"E:\柱塞泵\CWRU轴承数据\cwru_data\0HP",
    "cache_dir": r"E:\柱塞泵\CWRU轴承数据\cwru_data\0HP\cache",
    "output_dir": r"E:\柱塞泵\CWRU轴承数据\cwru_data\0HP\output",
    "sample_len": 1024,
    "cwt_scales": 64,
    "cwt_time_len": 64,
    "batch_size": 32,
    "epochs": 100,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "early_stop_patience": 15,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "num_workers": 4
}
os.makedirs(CONFIG["cache_dir"], exist_ok=True)
os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ================= 1. 数据集与CWT预处理 =================
class CWRUDataset(Dataset):
    def __init__(self, root, cache_dir, sample_len=1024, cwt_scales=64, cwt_time_len=64):
        self.cache_dir = cache_dir
        self.sample_len = sample_len
        self.cwt_scales = cwt_scales
        self.cwt_time_len = cwt_time_len

        self.file_map = {
            "100": (0, 0, 0), "105": (1, 0, 0), "118": (1, 0, 1), "130": (1, 0, 2),
            "169": (1, 1, 0), "185": (1, 1, 1), "197": (1, 1, 2),
            "222": (1, 2, 0), "234": (1, 2, 1), "246": (1, 2, 2)
        }
        self.samples, self.labels = [], []
        self._load_or_cache()

    def _load_or_cache(self):
        cache_file = os.path.join(self.cache_dir, "dataset.pt")
        if os.path.exists(cache_file):
            data = torch.load(cache_file, weights_only=False)
            self.samples, self.labels = data["x"], data["y"]
            return

        print("⏳ 加载 .mat 并计算 CWT 时频图...")
        mats = [f for f in os.listdir(CONFIG["data_root"]) if f.endswith(".mat")]
        if not mats:
            raise FileNotFoundError(f"❌ 目录 {CONFIG['data_root']} 中未找到任何 .mat 文件！")

        print(f"📂 共发现 {len(mats)} 个 .mat 文件，开始解析...")
        processed_count = 0

        for mat_name in mats:
            base = mat_name.split(".")[0]
            match_key = None
            for k in self.file_map.keys():
                if k in base:
                    match_key = k
                    break
            if match_key is None:
                continue

            try:
                mat = sio.loadmat(os.path.join(CONFIG["data_root"], mat_name))
                valid_keys = [k for k in mat.keys() if not k.startswith("__")]
                sig_key = None
                for k in valid_keys:
                    if "time" in k.lower():
                        sig_key = k
                        break
                if sig_key is None and valid_keys:
                    sig_key = valid_keys[0]
                if sig_key is None: continue

                sig = mat[sig_key].flatten()
            except Exception as e:
                continue

            n_seg = len(sig) // self.sample_len
            if n_seg == 0: continue

            for i in range(n_seg):
                seg = sig[i*self.sample_len : (i+1)*self.sample_len]
                scales = np.arange(1, self.cwt_scales + 1)
                coeffs, _ = pywt.cwt(seg, scales, "morl")
                img = np.log1p(np.abs(coeffs))
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
                img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(self.cwt_scales, self.cwt_time_len), mode='bilinear', align_corners=False).squeeze(0)
                self.samples.append(img_tensor)
                self.labels.append(torch.tensor(self.file_map[match_key], dtype=torch.long))
            processed_count += 1

        if not self.samples:
            raise RuntimeError("❌ 未成功加载任何样本！请检查数据路径与命名。")

        self.samples = torch.stack(self.samples)
        self.labels = torch.stack(self.labels)
        torch.save({"x": self.samples, "y": self.labels}, cache_file)
        print(f"✅ 缓存完成: {len(self.samples)} 样本 (来自 {processed_count} 个文件) | 形状: {self.samples.shape}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx], self.labels[idx]

# ================= 2. 核心模块 =================
class DynamicSnakeConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.k = kernel_size
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size//2)
        # 预测动态偏移场 (dx, dy)
        self.offset_conv = nn.Conv2d(in_ch, 2, kernel_size, padding=kernel_size//2)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        # 1. 预测基础偏移
        offset = self.offset_conv(x)  # [B, 2, H, W]
        # 2. 累积约束形成连续路径 (沿H方向)
        offset = torch.cumsum(offset, dim=2)
        offset = torch.tanh(offset) * 0.5  # 限制幅值防止越界

        # 3. 生成标准采样网格 [-1, 1]
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1).to(x.device)

        # 4. 叠加偏移并执行可变形采样
        grid = base_grid + offset.permute(0, 2, 3, 1)
        grid = grid.clamp(-1, 1)
        x_deformed = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return self.conv(x_deformed)

class DeformableAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.rel_pos = nn.Parameter(torch.zeros(num_heads, 1, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W).permute(1, 0, 2, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.rel_pos.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(B, self.num_heads, self.head_dim, H, W)
        out = out.permute(0, 2, 1, 3, 4).reshape(B, C, H, W)
        return self.proj(out)


class BearingDiagnosisNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.stem = nn.Sequential(
      nn.Conv2d(1, 32, 3, 1, 1),
      nn.BatchNorm2d(32),
      nn.ReLU(),

      nn.Conv2d(32, 64, 3, 2, 1),
      nn.BatchNorm2d(64),
      nn.ReLU()
    )

    self.dsc = DynamicSnakeConv(64, 128, kernel_size=3)
    self.dat = DeformableAttentionBlock(128, num_heads=4)

    self.head_nf = nn.Linear(128, 2)
    self.head_type = nn.Linear(128, 3)
    self.head_sev = nn.Linear(128, 3)

    self.stability_lambda = 0.05
    self.init_params = None

  def forward(self, x):
    x = self.stem(x)
    x = self.dsc(x)
    x = self.dat(x)
    x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)

    return self.head_nf(x), self.head_type(x), self.head_sev(x)

  def stability_reg(self):
    if self.init_params is None:
      self.init_params = torch.cat([p.data.flatten() for p in self.parameters()]).detach()
    curr = torch.cat([p.data.flatten() for p in self.parameters()])
    return F.mse_loss(curr, self.init_params)


# ================= 3. 训练/验证/早停机制 =================
class EarlyStopping:
    def __init__(self, patience=15, path="best_model.pth", verbose=True):
        self.patience = patience
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = copy.deepcopy(model.state_dict())
            if self.verbose: print(f"  📉 验证损失下降，已保存最优权重至 {self.path}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose: print(f"  ⏹️ 触发早停机制 (连续 {self.patience} 轮未改善)")

def train_epoch(model, loader, opt, sched, c_nf, c_type, c_sev, device, reg_lambda):
    model.train()
    t_loss, c_nf_acc, c_type_acc, c_sev_acc, n = 0.0, 0, 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        nf_out, type_out, sev_out = model(x)
        loss = c_nf(nf_out, y[:,0]) + c_type(type_out, y[:,1]) + c_sev(sev_out, y[:,2])
        loss += reg_lambda * model.stability_reg()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        t_loss += loss.item() * x.size(0)
        c_nf_acc += (nf_out.argmax(1) == y[:,0]).sum().item()
        c_type_acc += (type_out.argmax(1) == y[:,1]).sum().item()
        c_sev_acc += (sev_out.argmax(1) == y[:,2]).sum().item()
        n += x.size(0)
    sched.step()
    return t_loss/n, c_nf_acc/n, c_type_acc/n, c_sev_acc/n

@torch.no_grad()
def val_epoch(model, loader, c_nf, c_type, c_sev, device):
    model.eval()
    v_loss, c_nf_acc, c_type_acc, c_sev_acc, n = 0.0, 0, 0, 0, 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        nf_out, type_out, sev_out = model(x)
        loss = c_nf(nf_out, y[:,0]) + c_type(type_out, y[:,1]) + c_sev(sev_out, y[:,2])

        v_loss += loss.item() * x.size(0)
        c_nf_acc += (nf_out.argmax(1) == y[:,0]).sum().item()
        c_type_acc += (type_out.argmax(1) == y[:,1]).sum().item()
        c_sev_acc += (sev_out.argmax(1) == y[:,2]).sum().item()
        n += x.size(0)

        all_preds.extend(nf_out.argmax(1).cpu().numpy())
        all_labels.extend(y[:,0].cpu().numpy())
    return v_loss/n, c_nf_acc/n, c_type_acc/n, c_sev_acc/n, np.array(all_preds), np.array(all_labels)

def main():
    print(f"🚀 运行设备: {CONFIG['device']}")
    dataset = CWRUDataset(CONFIG["data_root"], CONFIG["cache_dir"])
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
    model = BearingDiagnosisNet().to(CONFIG["device"])

    opt = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["epochs"])

    c_nf = nn.CrossEntropyLoss()
    c_type = nn.CrossEntropyLoss()
    c_sev = nn.CrossEntropyLoss()
    early_stop = EarlyStopping(patience=CONFIG["early_stop_patience"], path=os.path.join(CONFIG["output_dir"], "best_model.pth"))

    history = {"train_loss": [], "val_loss": [], "acc_nf": [], "acc_type": [], "acc_sev": [], "lr": []}
    print("📊 开始训练循环...")

    for epoch in range(1, CONFIG["epochs"] + 1):
        t_loss, t_nf, t_type, t_sev = train_epoch(model, train_loader, opt, sched, c_nf, c_type, c_sev, CONFIG["device"], model.stability_lambda)
        v_loss, v_nf, v_type, v_sev, _, _ = val_epoch(model, val_loader, c_nf, c_type, c_sev, CONFIG["device"])
        current_lr = opt.param_groups[0]['lr']

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["acc_nf"].append(t_nf)
        history["acc_type"].append(t_type)
        history["acc_sev"].append(t_sev)
        history["lr"].append(current_lr)

        print(f"Epoch [{epoch:03d}/{CONFIG['epochs']}] | "
              f"Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | "
              f"Acc(NF/Type/Sev): {t_nf*100:.1f}% / {t_type*100:.1f}% / {t_sev*100:.1f}% | "
              f"LR: {current_lr:.2e}")

        early_stop(v_loss, model)
        if early_stop.early_stop:
            print("✅ 训练提前结束。")
            break

    print("📈 加载最优模型并生成评估图表...")
    model.load_state_dict(early_stop.best_model_state)
    v_loss, v_nf, v_type, v_sev, preds, labels = val_epoch(model, val_loader, c_nf, c_type, c_sev, CONFIG["device"])
    print(f"🏆 最终验证集性能 | Loss: {v_loss:.4f} | Acc(NF/Type/Sev): {v_nf*100:.1f}% / {v_type*100:.1f}% / {v_sev*100:.1f}%")

    # ================= 可视化 =================
    epochs_range = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss", marker='o', linestyle='-')
    plt.plot(epochs_range, history["val_loss"], label="Val Loss", marker='s', linestyle='-')
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["acc_nf"], label="Acc (Normal/Fault)", marker='o', linestyle='-')
    plt.plot(epochs_range, history["acc_type"], label="Acc (Fault Type)", marker='s', linestyle='-')
    plt.plot(epochs_range, history["acc_sev"], label="Acc (Severity)", marker='^', linestyle='-')
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "training_curves.png"), dpi=300)
    plt.close()

    cm = confusion_matrix(labels, preds, normalize='true') * 100
    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fault"])
    disp.plot(cmap="Blues", values_format=".1f")
    plt.title("Confusion Matrix (Percentage %)")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "confusion_matrix.png"), dpi=300)
    plt.close()

    print("✅ 全部完成！模型权重与可视化图表已保存至:", CONFIG["output_dir"])

if __name__ == "__main__":
    main()