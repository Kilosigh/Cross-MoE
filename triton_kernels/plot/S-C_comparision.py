import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置学术风格绘图参数
sns.set_theme(style="whitegrid", font="serif")  # 设置主题为带有网格的白底，字体为衬线字体
plt.rcParams['font.family'] = 'serif'  # 设置字体系列
plt.rcParams['mathtext.fontset'] = 'stix'  # 设置数学公式字体

# 1. 使用用户提供的数据
np.random.seed(42)  # 设置随机种子，保证结果可复现

# --- 图 1：Self-Attention 扩展数据 ---
n_self = np.array([64, 128, 256, 512, 1024, 2048, 4096])
time_torch_self = np.array([200.9006, 460.2583, 907.3759, 1891.3536, 3876.7872, 7568.4189, 7568.4189])
time_triton_self = np.array([8.5822, 8.6062, 8.8341, 11.5509, 9.4592, 16.1822, 16.1822])

# --- 图 2：Cross-Attention 扩展数据 ---
# 根据 seq_lengths = [2 ** i for i in range(0, 6)] 计算 KV 序列长度
# [1024, 2048, 4096, 8192, 16384, 32768]
seq_lengths = [2 ** i * 4 for i in range(0, 6)]
n_cross = np.array([x * 1024 for x in seq_lengths])
time_torch_cross = np.array([23.2236, 24.2429, 23.6038, 26.3883, 37.9856, 66.1398])
time_triton_cross = np.array([8.4571, 8.9528, 8.8418, 10.4115, 14.3708, 20.1278])


# 2. 绘制双图并排
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=False) # 创建 1 行 2 列的子图， sharey=False 表示不共享 Y 轴

# 颜色设置（深浅不一的学术配色）
# 蓝色系用于 PyTorch Native
color_torch = sns.color_palette("muted")[0]  # 使用 muted palette 的第一个蓝色
# 橙色系用于 Triton Custom
color_triton = sns.color_palette("muted")[1] # 使用 muted palette 的第二个橙色

# 设置标记点样式
marker_torch = 'o' # 圆点
marker_triton = '^' # 三角形


# =========================================================
# --- 图 1 (左)：Self-Attention 扩展 ---
# =========================================================
ax = axs[0]
ax.plot(n_self, time_torch_self, marker=marker_torch, linestyle='-', color=color_torch, label='PyTorch Native', linewidth=2, markersize=8)
ax.plot(n_self, time_triton_self, marker=marker_triton, linestyle='--', color=color_triton, label='Triton Custom', linewidth=2, markersize=8)

# 添加阴影表示超出显存截断的概念性展示 (由于数据未NaN，我们从最后一个点开始绘制阴影)
# 设置 OOM 区域起始点为 2048
oom_start_x = n_self[-1]

# 扩展 X 轴范围以便展示 OOM 阴影
extended_x_lim = 6000
ax.set_xlim(n_self[0]*0.9, extended_x_lim)

# 生成 OOM 区域的 X 轴范围
oom_x_range = np.linspace(oom_start_x, extended_x_lim, 100)
# 设置阴影颜色和透明度
ax.fill_between(oom_x_range, 0, ax.get_ylim()[1], color='red', alpha=0.1, hatch='//', edgecolor='red')
ax.text(5000, ax.get_ylim()[1] * 0.8, 'Conceptual OOM Region\n(Out of Memory)', color='red', fontsize=12, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))


# 设置图表属性
ax.set_title('Self-Attention Scaling', fontsize=16)
ax.set_xlabel('Sequence Length $N$', fontsize=14)
ax.set_ylabel('Execution Time (ms)', fontsize=14)
ax.set_xscale('log', base=2) # 设置 X 轴为对数刻度（以 2 为底）
ax.set_xticks(n_self) # 设置 X 轴刻度线位置
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter()) # 格式化 X 轴刻度标签为数字
ax.grid(True, which='both', linestyle='--', linewidth=0.5) # 显示主要和次要网格线
ax.legend(fontsize=12, loc='upper left') # 显示图例


# =========================================================
# --- 图 2 (右)：Cross-Attention 扩展 (Query Len=6, 长 KV) ---
# =========================================================
ax = axs[1]
ax.plot(n_cross, time_torch_cross, marker=marker_torch, linestyle='-', color=color_torch, label='PyTorch Native', linewidth=2, markersize=8)
ax.plot(n_cross, time_triton_cross, marker=marker_triton, linestyle='--', color=color_triton, label='Triton Custom', linewidth=2, markersize=8)

# 添加阴影表示超出显存截断的概念性展示 (按照要求，从 32768 开始)
# 设置 OOM 区域起始点为 32768
oom_start_x_cross = n_cross[-1]

# 扩展 X 轴范围以便展示 OOM 阴影
extended_x_lim_cross = 180000
ax.set_xlim(n_cross[0]*0.9, extended_x_lim_cross)

# 生成 OOM 区域的 X 轴范围
oom_x_range_cross = np.linspace(oom_start_x_cross, extended_x_lim_cross, 100)
# 设置阴影颜色和透明度
ax.fill_between(oom_x_range_cross, 0, ax.get_ylim()[1], color='red', alpha=0.1, hatch='//', edgecolor='red')
ax.text(155000, ax.get_ylim()[1] * 0.8, 'Conceptual OOM Region\n(Out of Memory)', color='red', fontsize=12, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# 设置图表属性
# 将标题改为 Query Len=6
ax.set_title('Cross-Attention Scaling (Query Len=6)', fontsize=16)
ax.set_xlabel('KV Sequence Length $N$', fontsize=14)
ax.set_ylabel('Execution Time (ms)', fontsize=14)
ax.set_xscale('log', base=2) # 设置 X 轴为对数刻度（以 2 为底）
ax.set_xticks(n_cross) # 设置 X 轴刻度线位置
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter()) # 格式化 X 轴刻度标签为数字
ax.grid(True, which='both', linestyle='--', linewidth=0.5) # 显示主要和次要网格线
ax.legend(fontsize=12, loc='upper left') # 显示图例

# 调整子图布局
plt.tight_layout()

# 保存图片，设置 DPI 为 300
plt.savefig('attention_scaling.png', dpi=300)

# # 如果需要直接显示图形，可以取消注释下方这一行
# plt.show()