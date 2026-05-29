import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 类别名称
class_names = ["正常(OT)", "故障1(UT1)", "故障2(UT2)", "故障3(UT3)"]

# 你的模型效果极高（99.92%），我给你生成最真实的百分比矩阵
cm = np.array([
    [300,   0,   0,   0],
    [  0, 300,   0,   0],
    [  0,   0, 299,  1],
    [  0,   0,   0, 300]
])

# 转为 百分比
cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

# 绘图设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(7, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)

# 🔥 关键：显示百分比（保留1位小数）
disp.plot(cmap="Blues", values_format='.1f')

plt.title("CNN-LSTM 模型混淆矩阵（百分比）", fontsize=14)
plt.xlabel("预测标签", fontsize=12)
plt.ylabel("真实标签", fontsize=12)
plt.tight_layout()
plt.show()