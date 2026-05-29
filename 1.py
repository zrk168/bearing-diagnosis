import pandas as pd, glob
f = glob.glob(r'E:\柱塞泵\pump_sim_dataset\**\*.csv', recursive=True)[0]
import pandas as pd, glob

files = glob.glob(r'E:\柱塞泵\pump_sim_dataset\**\*.csv', recursive=True)
# 过滤掉 label 文件，找第一个信号文件
signal_files = [f for f in files if 'label' not in f and 'meta' not in f]
print(f"找到 {len(signal_files)} 个信号文件")
print(signal_files[0])

df = pd.read_csv(signal_files[0])
print(df.shape)
print(df.dtypes)
