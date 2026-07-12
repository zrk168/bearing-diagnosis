import os
import copy
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import scipy.io as sio
import pywt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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


# ================= 1. 数据集：滑窗 -> CWT -> 时序划分 =================
class CWRUDataset(Dataset):
    """
    标签定义：
    labels[:, 0] -> 故障类型
        0: Normal
        1: Inner Race
        2: Ball
        3: Outer Race

    labels[:, 1] -> 故障程度
        -1: Normal（不参与故障程度分类）
         0: Light   / 0.007 in
         1: Medium  / 0.014 in
         2: Heavy   / 0.021 in

    split_tags:
        0 -> train
        1 -> val
        2 -> test

    划分方式：每个 .mat 文件内部，按时间顺序无重叠滑窗后，
    再按窗口顺序切分为 0~70% train / 70~90% val / 90~100% test，
    避免同一窗口在不同集合间重叠，防止数据泄露。
    """

    def __init__(self, root, cache_dir,
                 sample_len=1024,
                 cwt_scales=64,
                 cwt_time_len=64):

        self.root = root
        self.cache_dir = cache_dir
        self.sample_len = sample_len
        self.cwt_scales = cwt_scales
        self.cwt_time_len = cwt_time_len

        # 文件编号 -> (故障类型, 故障程度)
        self.file_map = {
            "100": (0, -1),   # Normal

            "105": (1, 0),    # Inner Race - Light
            "118": (1, 1),    # Inner Race - Medium
            "130": (1, 2),    # Inner Race - Heavy

            "169": (2, 0),    # Ball - Light
            "185": (2, 1),    # Ball - Medium
            "197": (2, 2),    # Ball - Heavy

            "222": (3, 0),    # Outer Race - Light
            "234": (3, 1),    # Outer Race - Medium
            "246": (3, 2),    # Outer Race - Heavy
        }

        self.samples = []
        self.labels = []
        self.source_ids = []
        self.split_tags = []

        self._load_or_cache()

    def _load_or_cache(self):
        cache_file = os.path.join(
            self.cache_dir,
            "dataset_cwt_temporal_split_v2.pt"
        )

        if os.path.exists(cache_file):
            print("📦 读取缓存数据集...")
            data = torch.load(cache_file, weights_only=False)

            self.samples = data["x"]
            self.labels = data["y"]
            self.source_ids = data["source_ids"]
            self.split_tags = data["split_tags"]

            print(
                f"✅ 缓存读取完成 | 样本数: {len(self.samples)} | "
                f"Train: {(self.split_tags == 0).sum().item()} | "
                f"Val: {(self.split_tags == 1).sum().item()} | "
                f"Test: {(self.split_tags == 2).sum().item()}"
            )
            return

        print("⏳ 加载 .mat 文件，执行无重叠滑窗与 CWT...")
        mats = sorted([
            f for f in os.listdir(self.root)
            if f.endswith(".mat")
        ])

        if not mats:
            raise FileNotFoundError(f"❌ 未在目录中找到 .mat 文件：{self.root}")

        print(f"📂 共发现 {len(mats)} 个 .mat 文件。")

        processed_count = 0

        for mat_name in mats:
            base = os.path.splitext(mat_name)[0]

            match_key = None
            for k in self.file_map.keys():
                if k in base:
                    match_key = k
                    break

            if match_key is None:
                print(f"⚠️ 跳过未匹配标签文件: {mat_name}")
                continue

            try:
                mat_path = os.path.join(self.root, mat_name)
                mat = sio.loadmat(mat_path)

                valid_keys = [k for k in mat.keys() if not k.startswith("__")]

                sig_key = None
                for k in valid_keys:
                    if "time" in k.lower():
                        sig_key = k
                        break

                if sig_key is None and valid_keys:
                    sig_key = valid_keys[0]

                if sig_key is None:
                    print(f"⚠️ 文件无有效信号: {mat_name}")
                    continue

                sig = mat[sig_key].flatten()

            except Exception as e:
                print(f"⚠️ 文件读取失败: {mat_name} | 原因: {e}")
                continue

            # ---------------- 无重叠滑窗 ----------------
            n_seg = len(sig) // self.sample_len

            if n_seg <= 0:
                print(f"⚠️ 信号长度不足一个窗口，跳过: {mat_name}")
                continue

            file_id = processed_count
            type_label, sev_label = self.file_map[match_key]

            print(
                f"处理: {mat_name} | "
                f"窗口数: {n_seg} | "
                f"Type={type_label}, Severity={sev_label}"
            )

            for i in range(n_seg):
                start = i * self.sample_len
                end = start + self.sample_len

                seg = sig[start:end]

                # ---------------- CWT ----------------
                scales = np.arange(1, self.cwt_scales + 1)
                coeffs, _ = pywt.cwt(seg, scales, "morl")

                img = np.log1p(np.abs(coeffs))
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)

                img_tensor = torch.tensor(
                    img, dtype=torch.float32
                ).unsqueeze(0)

                img_tensor = F.interpolate(
                    img_tensor.unsqueeze(0),
                    size=(self.cwt_scales, self.cwt_time_len),
                    mode="bilinear",
                    align_corners=False
                ).squeeze(0)

                self.samples.append(img_tensor)

                self.labels.append(
                    torch.tensor([type_label, sev_label], dtype=torch.long)
                )

                self.source_ids.append(file_id)

            processed_count += 1

        if not self.samples:
            raise RuntimeError("❌ 未成功加载任何样本，请检查路径和文件名。")

        self.samples = torch.stack(self.samples)
        self.labels = torch.stack(self.labels)
        self.source_ids = torch.tensor(self.source_ids, dtype=torch.long)

        # ================= 按文件内部时间顺序划分 70/20/10 =================
        self.split_tags = torch.full(
            (len(self.samples),), -1, dtype=torch.long
        )

        unique_file_ids = torch.unique(self.source_ids)

        for file_id in unique_file_ids:
            file_indices = torch.where(self.source_ids == file_id)[0]
            n = len(file_indices)

            train_end = int(n * 0.70)
            val_end = int(n * 0.90)

            train_idx = file_indices[:train_end]
            val_idx = file_indices[train_end:val_end]
            test_idx = file_indices[val_end:]

            self.split_tags[train_idx] = 0
            self.split_tags[val_idx] = 1
            self.split_tags[test_idx] = 2

        assert torch.all(self.split_tags >= 0), "存在未被划分的样本。"

        torch.save({
            "x": self.samples,
            "y": self.labels,
            "source_ids": self.source_ids,
            "split_tags": self.split_tags
        }, cache_file)

        print("\n✅ 数据处理及缓存完成。")
        print(f"总样本数: {len(self.samples)}")
        print(f"训练集: {(self.split_tags == 0).sum().item()}")
        print(f"验证集: {(self.split_tags == 1).sum().item()}")
        print(f"测试集: {(self.split_tags == 2).sum().item()}")
        print(f"缓存路径: {cache_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


# ================= 2. 核心模块 =================
class BiLSTMBridge(nn.Module):
    """沿时频图的时间轴（W维度）做双向LSTM，捕捉每个频率通道内的时序依赖。

    输入/输出形状均为 [B, C, H, W]，
    通过残差连接保持梯度通畅。
    """
    def __init__(self, channels, hidden_size=64, num_layers=1):
        super().__init__()
        self.channels = channels
        self.hidden_size = hidden_size

        # BiLSTM：输入维度=channels，双向所以输出是hidden_size*2
        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # 把 hidden_size*2 投影回 channels，保持维度不变
        self.proj = nn.Linear(hidden_size * 2, channels)

        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, C, H, W = x.shape

        # [B, C, H, W] -> [B, H, W, C]
        # 把每一行频率（H中的每一行）当做一条长度为W的时间序列
        x_perm = x.permute(0, 2, 3, 1)

        # 合并 B 和 H：每条频率序列独立送进LSTM
        # [B, H, W, C] -> [B*H, W, C]
        x_seq = x_perm.reshape(B * H, W, C)

        # BiLSTM: [B*H, W, hidden_size*2]
        lstm_out, _ = self.lstm(x_seq)

        # 投影回 channels：[B*H, W, C]
        lstm_out = self.proj(lstm_out)
        lstm_out = self.dropout(lstm_out)

        # reshape回原始空间维度：[B, H, W, C]
        lstm_out = lstm_out.reshape(B, H, W, C)

        # LayerNorm +残差连接
        out = self.norm(lstm_out + x_perm)

        # [B, H, W, C] -> [B, C, H, W]
        out = out.permute(0, 3, 1, 2)

        return out

class DynamicSnakeConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.k = kernel_size
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.offset_conv = nn.Conv2d(in_ch, 2, kernel_size, padding=kernel_size // 2)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

    def forward(self, x):
        B, C, H, W = x.shape

        offset = self.offset_conv(x)
        offset = torch.cumsum(offset, dim=2)
        offset = torch.tanh(offset) * 0.5

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1).to(x.device)

        grid = base_grid + offset.permute(0, 2, 3, 1)
        grid = grid.clamp(-1, 1)

        x_deformed = F.grid_sample(
            x, grid, mode='bilinear', padding_mode='zeros', align_corners=True
        )
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

        self.lstm_bridge = BiLSTMBridge(channels=128, hidden_size=64)

        self.dat = DeformableAttentionBlock(128, num_heads=4)

        self.head_type = nn.Linear(128, 4)
        self.head_sev = nn.Linear(128, 3)

        self.stability_lambda = 0.05
        self.init_params = None

    def forward(self, x):
        x = self.stem(x)
        x = self.dsc(x)
        x = self.lstm_bridge(x)
        x = self.dat(x)

        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)

        type_out = self.head_type(x)
        sev_out = self.head_sev(x)

        return type_out, sev_out

    def stability_reg(self):
        if self.init_params is None:
            self.init_params = torch.cat(
                [p.data.flatten() for p in self.parameters()]
            ).detach()

        curr = torch.cat(
            [p.data.flatten() for p in self.parameters()]
        )

        return F.mse_loss(curr, self.init_params)



# ================= 3. 早停机制 =================
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
            if self.verbose:
                print(f"  📉 验证损失下降，已保存最优权重至 {self.path}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  ⏹️ 触发早停机制 (连续 {self.patience} 轮未改善)")


# ================= 4. 训练 / 验证 =================
def train_epoch(model, loader, opt, sched, c_type, c_sev, device, reg_lambda):
    model.train()

    total_loss = 0.0
    type_correct = 0
    sev_correct = 0

    total_num = 0
    sev_num = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        type_label = y[:, 0]
        sev_label = y[:, 1]

        opt.zero_grad()

        type_out, sev_out = model(x)

        loss_type = c_type(type_out, type_label)

        valid_sev_mask = sev_label >= 0
        if valid_sev_mask.any():
            loss_sev = c_sev(sev_out[valid_sev_mask], sev_label[valid_sev_mask])
        else:
            loss_sev = torch.tensor(0.0, device=device)

        loss = loss_type + loss_sev
        loss = loss + reg_lambda * model.stability_reg()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_num += batch_size

        type_pred = type_out.argmax(dim=1)
        sev_pred = sev_out.argmax(dim=1)

        type_correct += (type_pred == type_label).sum().item()

        if valid_sev_mask.any():
            sev_correct += (sev_pred[valid_sev_mask] == sev_label[valid_sev_mask]).sum().item()
            sev_num += valid_sev_mask.sum().item()

    sched.step()

    type_acc = type_correct / total_num
    sev_acc = sev_correct / sev_num if sev_num > 0 else 0.0

    return total_loss / total_num, type_acc, sev_acc


@torch.no_grad()
def val_epoch(model, loader, c_type, c_sev, device):
    model.eval()

    total_loss = 0.0
    type_correct = 0
    sev_correct = 0

    total_num = 0
    sev_num = 0

    all_type_preds = []
    all_type_labels = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        type_label = y[:, 0]
        sev_label = y[:, 1]

        type_out, sev_out = model(x)

        loss_type = c_type(type_out, type_label)

        valid_sev_mask = sev_label >= 0
        if valid_sev_mask.any():
            loss_sev = c_sev(sev_out[valid_sev_mask], sev_label[valid_sev_mask])
        else:
            loss_sev = torch.tensor(0.0, device=device)

        loss = loss_type + loss_sev

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_num += batch_size

        type_pred = type_out.argmax(dim=1)
        sev_pred = sev_out.argmax(dim=1)

        type_correct += (type_pred == type_label).sum().item()

        if valid_sev_mask.any():
            sev_correct += (sev_pred[valid_sev_mask] == sev_label[valid_sev_mask]).sum().item()
            sev_num += valid_sev_mask.sum().item()

        all_type_preds.extend(type_pred.cpu().numpy())
        all_type_labels.extend(type_label.cpu().numpy())

    type_acc = type_correct / total_num
    sev_acc = sev_correct / sev_num if sev_num > 0 else 0.0

    return (
        total_loss / total_num,
        type_acc,
        sev_acc,
        np.array(all_type_preds),
        np.array(all_type_labels)
    )


# ================= 5. 主流程 =================
def main():
    print(f"🚀 运行设备: {CONFIG['device']}")

    dataset = CWRUDataset(
        CONFIG["data_root"],
        CONFIG["cache_dir"],
        sample_len=CONFIG["sample_len"],
        cwt_scales=CONFIG["cwt_scales"],
        cwt_time_len=CONFIG["cwt_time_len"]
    )

    train_indices = torch.where(dataset.split_tags == 0)[0].tolist()
    val_indices = torch.where(dataset.split_tags == 1)[0].tolist()
    test_indices = torch.where(dataset.split_tags == 2)[0].tolist()

    train_ds = torch.utils.data.Subset(dataset, train_indices)
    val_ds = torch.utils.data.Subset(dataset, val_indices)
    test_ds = torch.utils.data.Subset(dataset, test_indices)

    print("\n📊 数据划分结果：")
    print(f"训练集: {len(train_ds)}")
    print(f"验证集: {len(val_ds)}")
    print(f"测试集: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=CONFIG["num_workers"], pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=CONFIG["num_workers"], pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=CONFIG["num_workers"], pin_memory=True
    )

    model = BearingDiagnosisNet().to(CONFIG["device"])

    opt = optim.Adam(
        model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
    )
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["epochs"])

    c_type = nn.CrossEntropyLoss()
    c_sev = nn.CrossEntropyLoss()

    early_stop = EarlyStopping(
        patience=CONFIG["early_stop_patience"],
        path=os.path.join(CONFIG["output_dir"], "best_model.pth")
    )

    history = {
        "train_loss": [], "val_loss": [], "test_loss": [],
        "train_type_acc": [], "val_type_acc": [], "test_type_acc": [],
        "train_sev_acc": [], "val_sev_acc": [], "test_sev_acc": [],
        "lr": []
    }

    print("\n📊 开始训练循环...")

    for epoch in range(1, CONFIG["epochs"] + 1):

        train_loss, train_type_acc, train_sev_acc = train_epoch(
            model, train_loader, opt, sched, c_type, c_sev,
            CONFIG["device"], model.stability_lambda
        )

        val_loss, val_type_acc, val_sev_acc, _, _ = val_epoch(
            model, val_loader, c_type, c_sev, CONFIG["device"]
        )

        # 仅用于记录曲线，不参与训练、调度和早停
        test_loss, test_type_acc, test_sev_acc, _, _ = val_epoch(
            model, test_loader, c_type, c_sev, CONFIG["device"]
        )

        current_lr = opt.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["test_loss"].append(test_loss)

        history["train_type_acc"].append(train_type_acc)
        history["val_type_acc"].append(val_type_acc)
        history["test_type_acc"].append(test_type_acc)

        history["train_sev_acc"].append(train_sev_acc)
        history["val_sev_acc"].append(val_sev_acc)
        history["test_sev_acc"].append(test_sev_acc)

        history["lr"].append(current_lr)

        print(
            f"Epoch [{epoch:03d}/{CONFIG['epochs']}] | "
            f"Loss(T/V/Te): {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f} | "
            f"Type Acc(T/V/Te): {train_type_acc*100:.2f}%/{val_type_acc*100:.2f}%/{test_type_acc*100:.2f}% | "
            f"Sev Acc(T/V/Te): {train_sev_acc*100:.2f}%/{val_sev_acc*100:.2f}%/{test_sev_acc*100:.2f}%"
        )

        early_stop(val_loss, model)
        if early_stop.early_stop:
            print("✅ 训练提前结束。")
            break

    print("\n📈 加载验证集最优模型...")
    model.load_state_dict(early_stop.best_model_state)

    final_val_loss, final_val_type_acc, final_val_sev_acc, _, _ = val_epoch(
        model, val_loader, c_type, c_sev, CONFIG["device"]
    )

    final_test_loss, final_test_type_acc, final_test_sev_acc, test_preds, test_labels = val_epoch(
        model, test_loader, c_type, c_sev, CONFIG["device"]
    )

    print("\n🏆 最终结果：")
    print(
        f"验证集 | Loss: {final_val_loss:.4f} | "
        f"类型准确率: {final_val_type_acc*100:.2f}% | "
        f"程度准确率: {final_val_sev_acc*100:.2f}%"
    )
    print(
        f"测试集 | Loss: {final_test_loss:.4f} | "
        f"类型准确率: {final_test_type_acc*100:.2f}% | "
        f"程度准确率: {final_test_sev_acc*100:.2f}%"
    )

    # ================= 准确率曲线 =================
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    epochs_range = range(1, len(history["train_type_acc"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(epochs_range, history["train_type_acc"], label="训练集", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs_range, history["val_type_acc"], label="验证集", color="#ff7f0e", linewidth=2, linestyle="--")
    axes[0].plot(epochs_range, history["test_type_acc"], label="测试集", color="#d62728", linewidth=2, linestyle=":")
    axes[0].set_title("故障类型分类准确率")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs_range, history["train_sev_acc"], label="训练集", color="#1f77b4", linewidth=2)
    axes[1].plot(epochs_range, history["val_sev_acc"], label="验证集", color="#ff7f0e", linewidth=2, linestyle="--")
    axes[1].plot(epochs_range, history["test_sev_acc"], label="测试集", color="#d62728", linewidth=2, linestyle=":")
    axes[1].set_title("故障程度分类准确率")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("故障类型与故障程度准确率曲线", fontsize=15)
    plt.tight_layout()

    acc_path = os.path.join(CONFIG["output_dir"], "type_severity_accuracy_curves.png")
    plt.savefig(acc_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n✅ 准确率曲线已保存: {acc_path}")

    # ================= 故障类型混淆矩阵 =================
    cm = confusion_matrix(test_labels, test_preds, normalize="true") * 100

    plt.figure(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Inner Race", "Ball", "Outer Race"]
    )
    disp.plot(cmap="Blues", values_format=".1f")
    plt.title("Test Confusion Matrix - Fault Type (%)")
    plt.tight_layout()

    cm_path = os.path.join(CONFIG["output_dir"], "test_confusion_matrix_fault_type.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ 故障类型混淆矩阵已保存: {cm_path}")
    print(f"✅ 全部完成，输出目录: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
