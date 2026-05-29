import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# ===================== 你的实验数据（20轮） =====================
epochs = list(range(1, 21))

# 训练集
train_loss = [0.8509,0.5645,0.4679,0.4020,0.3532,0.3116,0.2883,0.2601,0.2439,0.2315,
              0.2191,0.2091,0.2014,0.1941,0.1884,0.1751,0.1690,0.1667,0.1582,0.1551]
train_acc  = [0.6605,0.7877,0.8247,0.8549,0.8736,0.8908,0.8993,0.9100,0.9180,0.9221,
              0.9246,0.9299,0.9310,0.9341,0.9342,0.9393,0.9434,0.9437,0.9465,0.9462]

# 测试集
test_loss = [0.6074,0.4844,0.4001,0.3355,0.3080,0.2790,0.2434,0.2264,0.2205,0.2189,
             0.1963,0.1836,0.1712,0.1835,0.1754,0.1593,0.1539,0.1532,0.1471,0.1398]
test_acc  = [0.7663,0.8139,0.8507,0.8803,0.8944,0.9022,0.9150,0.9255,0.9229,0.9231,
             0.9336,0.9352,0.9435,0.9373,0.9352,0.9461,0.9443,0.9448,0.9482,0.9482]

# ===================== 收敛点（根据数据确定） =====================
conv_epoch = 16
conv_train_loss = 0.1751
conv_train_acc  = 0.9393
conv_test_loss  = 0.1593
conv_test_acc   = 0.9461

# ===================== 百分比混淆矩阵（4分类：0 1 2 3） =====================
cm = np.array([
    [95, 3, 1, 1],
    [2, 94, 2, 2],
    [1, 2, 95, 2],
    [0, 1, 1, 98]
])

# ===================== 绘图设置 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ======================================================================
# 图 1：训练集 双轴曲线（左准确率，右损失）
# ======================================================================
plt.figure(figsize=(10, 5))
ax1 = plt.gca()
ax1.set_xlabel('迭代次数 (Epoch)', fontsize=12)
ax1.set_ylabel('训练准确率', color='blue', fontsize=12)
line1 = ax1.plot(epochs, train_acc, 'b-o', linewidth=2, label='训练准确率')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.set_ylabel('训练损失', color='red', fontsize=12)
line2 = ax2.plot(epochs, train_loss, 'r-o', linewidth=2, label='训练损失')
ax2.tick_params(axis='y', labelcolor='red')

# 标注收敛点
ax1.axvline(x=conv_epoch, color='g', linestyle='--', linewidth=2)
ax1.text(conv_epoch+0.8, 0.75,
         f'收敛轮数：{conv_epoch}\nAcc：{conv_train_acc:.4f}\nLoss：{conv_train_loss:.4f}',
         color='green', fontweight='bold', fontsize=11)

plt.title('训练集准确率与损失曲线', fontsize=14)
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ======================================================================
# 图 2：测试集 双轴曲线（左准确率，右损失）
# ======================================================================
plt.figure(figsize=(10, 5))
ax1 = plt.gca()
ax1.set_xlabel('迭代次数 (Epoch)', fontsize=12)
ax1.set_ylabel('测试准确率', color='blue', fontsize=12)
line1 = ax1.plot(epochs, test_acc, 'b-o', linewidth=2, label='测试准确率')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.set_ylabel('测试损失', color='red', fontsize=12)
line2 = ax2.plot(epochs, test_loss, 'r-o', linewidth=2, label='测试损失')
ax2.tick_params(axis='y', labelcolor='red')

# 标注收敛点
ax1.axvline(x=conv_epoch, color='g', linestyle='--', linewidth=2)
ax1.text(conv_epoch+0.8, 0.8,
         f'收敛轮数：{conv_epoch}\nAcc：{conv_test_acc:.4f}\nLoss：{conv_test_loss:.4f}',
         color='green', fontweight='bold', fontsize=11)

plt.title('测试集准确率与损失曲线', fontsize=14)
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ======================================================================
# 图 3：百分比混淆矩阵（0 1 2 3）
# ======================================================================
plt.figure(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1', '2', '3'])
disp.plot(cmap='Blues', values_format='d')  # 百分制数字
plt.title('CNN-Transformer 混淆矩阵（百分比）', fontsize=14)
plt.xlabel('预测标签', fontsize=12)
plt.ylabel('真实标签', fontsize=12)
plt.tight_layout()
plt.show()