import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 讀取CSV文件
csv_file = 'training_loss_2view.csv'
data = pd.read_csv(csv_file)

# 繪製圖表
plt.figure(figsize=(10, 6))
plt.plot(data['iteration'], data['loss'], marker='o', linestyle='-', color='b')
plt.title('Training Loss per Iteration')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
xticks = list(range(0, 1001, 100)) + list(range(1000, 10001, 1000))
yticks = list(np.arange(0, data['loss'].max(), 0.02))
print(yticks)
plt.xticks(xticks, rotation=90)
plt.yticks(yticks)
plt.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)
plt.savefig('training_loss_plot.png')  # 儲存圖表為PNG檔案
plt.show()  # 顯示圖表
