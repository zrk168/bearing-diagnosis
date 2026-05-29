import matplotlib.pyplot as plt

# ====================== 你提供的训练集数据 ======================
epochs = list(range(1, 61))

train_acc = [
0.5179, 0.5631, 0.5902, 0.6079, 0.6144,
0.6498, 0.6846, 0.8185, 0.9125, 0.9800,
0.9915, 0.9706, 0.9808, 0.9827, 0.9990,
0.9875, 0.9992, 0.9881, 0.9804, 0.9981,
0.9921, 0.9940, 0.9940, 0.9904, 0.9956,
0.9985, 0.9840, 0.9996, 0.9996, 0.9883,
0.9992, 0.9962, 0.9927, 0.9971, 0.9992,
0.9942, 0.9910, 0.9973, 0.9977, 0.9985,
0.9848, 0.9992, 0.9919, 0.9994, 0.9992,
0.9962, 0.9996, 0.9983, 0.9988, 0.9842,
0.9994, 0.9996, 0.9981, 0.9979, 0.9898,
0.9998, 0.9915, 0.9977, 0.9994, 0.9996
]

train_loss = [
0.9367, 0.8164, 0.7751, 0.7525, 0.7400,
0.7052, 0.6289, 0.4095, 0.2307, 0.0879,
0.0392, 0.0932, 0.0616, 0.0537, 0.0092,
0.0326, 0.0086, 0.0392, 0.0644, 0.0099,
0.0201, 0.0178, 0.0322, 0.0347, 0.0146,
0.0075, 0.0550, 0.0045, 0.0031, 0.0396,
0.0036, 0.0095, 0.0209, 0.0079, 0.0032,
0.0204, 0.0253, 0.0062, 0.0060, 0.0054,
0.0447, 0.0043, 0.0328, 0.0040, 0.0038,
0.0104, 0.0029, 0.0054, 0.0040, 0.0771,
0.0056, 0.0018, 0.0057, 0.0060, 0.0507,
0.0024, 0.0292, 0.0094, 0.0027, 0.0027
]

# ====================== 绘图 ======================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax1 = plt.subplots(figsize=(11, 5))

# 左Y轴：准确率
ax1.set_xlabel('迭代次数 (Epoch)', fontsize=12)
ax1.set_ylabel('训练集准确率', color='blue', fontsize=12)
line1 = ax1.plot(epochs, train_acc, color='blue', linewidth=2, label='训练集准确率')
ax1.tick_params(axis='y', labelcolor='blue')

# 右Y轴：损失
ax2 = ax1.twinx()
ax2.set_ylabel('训练集损失', color='red', fontsize=12)
line2 = ax2.plot(epochs, train_loss, color='red', linewidth=2, label='训练集损失')
ax2.tick_params(axis='y', labelcolor='red')

# 标注收敛点（第15轮后平稳）
ax1.axvline(x=15, color='green', linestyle='--', linewidth=2)
ax1.text(16, 0.6, '第15轮后曲线趋于平稳', color='green', fontsize=12, fontweight='bold')

# 合并图例
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')

plt.title('训练集准确率与损失变化曲线', fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()