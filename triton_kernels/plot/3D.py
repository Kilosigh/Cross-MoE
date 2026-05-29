import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

# ==========================================
# 1. 全局样式设置 (学术论文风格)
# ==========================================
sns.set_theme(style="whitegrid", context="paper")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

# ==========================================
# 2. 数据填入
# ==========================================
# Sequence Length (N): 64 到 2048
N_vals = np.array([64, 128, 256, 512, 1024, 2048])
# Split-K (S): 1 到 16
S_vals = np.array([1, 2, 4, 8, 16])

# 图 1: 3D 曲面网格数据 (T_mesh)
T_mesh = np.array([
    [8.1562, 7.9706, 9.0888, 8.0627, 9.5832, 12.7800],  # Split-K = 1
    [8.8460, 7.8641, 10.3986, 7.8083, 9.3568, 11.5569], # Split-K = 2
    [8.1164, 8.1757, 8.3514, 8.5098, 11.6632, 11.8322], # Split-K = 4
    [8.2145, 8.5581, 11.6085, 7.2923, 12.4479, 14.6854], # Split-K = 8
    [10.5897, 8.1157, 8.4462, 7.4477, 9.5355, 11.8315],  # Split-K = 16
])

# 图 2: 2D 折线图数据
time_pytorch = np.array([236.9333, 440.4637, 927.8577, 1743.0733, 4030.5562, 8134.5537])
time_triton = np.array([10.5897, 8.1157, 8.4462, 7.4477, 9.5355, 11.8315])

N_mesh, S_mesh = np.meshgrid(N_vals, S_vals)

# 采用 log2 空间作为绘图坐标，防止 3D 网格变形
N_log_mesh = np.log2(N_mesh)
S_log_mesh = np.log2(S_mesh)

# ==========================================
# 3. 图像绘制
# ==========================================
fig = plt.figure(figsize=(13, 5.5))

# ------------------------------------------
# 图 1 (左): 3D Surface Plot
# ------------------------------------------
ax1 = fig.add_subplot(1, 2, 1, projection='3d')

surf = ax1.plot_surface(N_log_mesh, S_log_mesh, T_mesh, 
                        cmap='viridis', 
                        edgecolor='none', 
                        alpha=0.9, 
                        antialiased=True)

ax1.set_xlabel('Sequence Length ($N$)', labelpad=12)
ax1.set_ylabel('Split-K ($num\_splits$)', labelpad=12)
ax1.set_zlabel('Time (ms)', labelpad=12)
ax1.set_title('(a) Execution Time vs. $N$ and Split-K', pad=20, fontweight='bold')

# 重置 X 和 Y 刻度为真实值
ax1.set_xticks(np.log2(N_vals))
ax1.set_xticklabels(N_vals)
ax1.set_yticks(np.log2(S_vals))
ax1.set_yticklabels(S_vals)

# 视角设定 (稍微拉高一点仰角以适应紧凑布局)
ax1.view_init(elev=32, azim=-125)
# 修复：移除了导致警告的 ax1.dist = 11

# 调整 Colorbar 间距
cbar = fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=12, pad=0.15)
cbar.set_label('Time (ms)', rotation=270, labelpad=15)

# ------------------------------------------
# 图 2 (右): 2D 折线图
# ------------------------------------------
ax2 = fig.add_subplot(1, 2, 2)

# PyTorch 曲线
ax2.plot(N_vals, time_pytorch, 
         label='PyTorch Native', 
         color='#d62728',         
         linewidth=2.5, 
         linestyle='--',
         marker='s',              
         markersize=6)

# Triton 曲线
ax2.plot(N_vals, time_triton, 
         label='Triton Custom ($num\_splits=16$)', 
         color='#1f77b4',         
         linewidth=2.5, 
         linestyle='-', 
         marker='o',              
         markersize=6)            

# X 轴：对数坐标
ax2.set_xscale('log', base=2)
ax2.set_xticks(N_vals)
ax2.xaxis.set_major_formatter(ScalarFormatter())

# Y 轴：新增对数坐标 (Log Scale) 以展示 8000ms 和 10ms 差距
ax2.set_yscale('log', base=10)
# 若强行要求 Y 轴是线性的，请注释掉上面这一行。

ax2.set_xlabel('Sequence Length ($N$)')
ax2.set_ylabel('Execution Time (ms) - Log Scale')
ax2.set_title('(b) Performance Comparison at Optimal Split-K', pad=15, fontweight='bold')

# 图例和网格设置
ax2.legend(loc='upper left', frameon=True, shadow=False, edgecolor='black')
# 为对数坐标增加次级网格线 (minor grid) 使其更专业
ax2.grid(True, which='major', linestyle='-', linewidth=1.0, alpha=0.7)
ax2.grid(True, which='minor', linestyle=':', linewidth=0.8, alpha=0.4)

# ==========================================
# 4. 整体布局调整与输出
# ==========================================
# 修复：增加 pad 和 wspace 确保元素有足够的留白，不挤压
plt.tight_layout(pad=2.0, w_pad=1.5)

# 保存时 bbox_inches='tight' 会接管最终裁切，现在已经不冲突了
plt.savefig("performance_comparison_updated.png", format='png', dpi=300)
# print("绘图完成！已保存为 performance_comparison_updated.png")