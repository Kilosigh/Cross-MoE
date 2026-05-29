import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ==========================================
# 1. 全局学术风格设置 (IEEE/ACM Standard)
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'], 
    'font.size': 10,               
    'axes.labelsize': 11,          
    'axes.titlesize': 12,          
    'xtick.labelsize': 10,         
    'ytick.labelsize': 10,
    'legend.fontsize': 10,         
    'figure.dpi': 300,             
    'mathtext.fontset': 'stix'     
})

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))



D_vals = [192, 384, 768]
BLOCK_D_vals = [16, 32, 64]
tflops_data = np.array([[5.62, 20.67, 77.57], [5.97, 20.78, 74.64], [6.0, 20.8, 77.65]])

batch_sizes = [1, 2, 4, 8, 16, 32]
total_time = [1.03, 0.97, 0.98, 1.03, 1.64, 3.03]
speedup = [2.28, 2.4, 2.75, 3.49, 3.61, 3.46]



# ==========================================
# 图 1: 热力图 (Feature D vs BLOCK_D)
# ==========================================
ax1 = axes[0]


sns.heatmap(tflops_data, annot=True, fmt=".1f", cmap="YlGnBu", 
            xticklabels=D_vals, yticklabels=BLOCK_D_vals, 
            cbar_kws={'label': 'Compute Throughput (TFLOPS)'}, ax=ax1)

ax1.set_xlabel('Feature Dimension ($D$)')
ax1.set_ylabel('Triton Block Size ($\\mathit{BLOCK\\_D}$)')
ax1.set_title('(a) Throughput Heatmap (TFLOPS)')
ax1.invert_yaxis()  # 标准化坐标系，使 BLOCK_D 从小到大排列

# ==========================================
# 图 2: 双 Y 轴图 (Batch Size 性能影响)
# ==========================================
ax2 = axes[1]
# 根据实际测试输出，最高测试到 32
x_pos = np.arange(len(batch_sizes))


ax2_twin = ax2.twinx()

color_bar = '#4C72B0' 
color_line = '#C44E52'

bars = ax2.bar(x_pos, total_time, color=color_bar, alpha=0.75, width=0.5, label='Triton Time (ms)')

line = ax2_twin.plot(x_pos, speedup, color=color_line, marker='s', markersize=6, 
                     linewidth=2, linestyle='-', label='Speedup')

ax2.set_xticks(x_pos)
ax2.set_xticklabels(batch_sizes)
ax2.set_xlabel('Batch Size')
ax2.set_title('(b) Performance vs. Batch Size')

ax2.set_ylabel('Triton Time (ms)', color=color_bar)
ax2_twin.set_ylabel('Speedup Ratio', color=color_line)
ax2.tick_params(axis='y', labelcolor=color_bar)
ax2_twin.tick_params(axis='y', labelcolor=color_line)

# 合并图例并设置在图表左上方
lines_1, labels_1 = ax2.get_legend_handles_labels()
lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
ax2_twin.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', framealpha=0.9)

# ==========================================
# 3. 布局调整与保存
# ==========================================
plt.tight_layout()
# plt.savefig('hardware_finetuning_analysis_real.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.savefig('hardware_finetuning_analysis_real.png', dpi=300, bbox_inches='tight')
plt.show()