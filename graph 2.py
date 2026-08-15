import matplotlib.pyplot as plt
import numpy as np

# Data
engines = ['TensorFlow Lite', 'Intel OpenVINO']
preprocessing = np.array([0.30, 0.30])
inference = np.array([2.27, 1.73])
postprocessing = np.array([0.20, 0.20])

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))
width = 0.5

# Create Stacked Bars
p1 = ax.bar(engines, preprocessing, width, label='Preprocessing', color='lightgray')
p2 = ax.bar(engines, inference, width, bottom=preprocessing, label='Model Inference', color=['#4285F4', '#F4B400']) # Blue & Orange
p3 = ax.bar(engines, postprocessing, width, bottom=preprocessing+inference, label='Post-processing', color='darkgray')

# Labels
ax.set_ylabel('Time (milliseconds)')
ax.set_title('Latency Breakdown: Where is the Time Spent?')
ax.legend()

# Add value labels
ax.bar_label(p1, label_type='center')
ax.bar_label(p2, label_type='center', color='white', weight='bold')
ax.bar_label(p3, label_type='center')

# Add Total Time on top
totals = preprocessing + inference + postprocessing
for i, total in enumerate(totals):
    ax.text(i, total + 0.05, f'Total: {total:.2f} ms', ha='center', weight='bold')

plt.savefig('figure6_latency_breakdown.png', dpi=300)
plt.show()