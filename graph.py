import matplotlib.pyplot as plt
import numpy as np

# Data from your research
metrics = ['FPS', 'Latency (ms)', 'CPU Usage (%)', 'Model Size (MB)']
tflite_vals = [361.01, 2.77, 28.4, 8.9]
openvino_vals = [447.43, 2.23, 24.1, 4.6]

x = np.arange(len(metrics))  # Label locations
width = 0.35  # Bar width

fig, ax = plt.subplots(figsize=(10, 6))

# Create bars
rects1 = ax.bar(x - width/2, tflite_vals, width, label='TensorFlow Lite', color='#4285F4') # Standard Blue
rects2 = ax.bar(x + width/2, openvino_vals, width, label='Intel OpenVINO', color='#F4B400') # Standard Orange/Gold

# Labels and Title
ax.set_ylabel('Value')
ax.set_title('Performance Comparison: TensorFlow Lite vs Intel OpenVINO')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Function to add value labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig('figure4_performance_comparison.png', dpi=300)
plt.show()