import numpy as np
import matplotlib.pyplot as plt

# ======================
# 参数配置区（方便自定义）
# ======================
plt.rcParams.update({'font.size': 10})  # 全局字体大小

# 数据集参数
datasets = ['ETTh1', 'Weather']
num_samples = 4        # 样本数量（S1-S4）
num_fragments = 4      # 时间片段数量（D1-D4）
num_time_points = 96

# 颜色配置
line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 折线图颜色（蓝/橙/绿/红）
cmap_weights = 'Blues'  # 权重热图颜色映射

# 图形尺寸
figsize = (10, 8)  # 建议宽高比4:3

# ======================
# 数据生成（示例随机数据）
# ======================
# 生成时间序列数据（形状：[数据集数, 样本数, 时间片段数]）
# line_data = np.random.randn(2, num_samples, num_fragments).cumsum(axis=2)  # 添加累计噪声
line_data = np.random.randn(2, num_samples, num_time_points)
print(line_data.shape)

# 生成权重矩阵（形状：[数据集数, 样本数, 类别数]）
heatmap_data = np.random.rand(2, num_samples, num_fragments)
heatmap_data /= heatmap_data.sum(axis=2, keepdims=True)  # 归一化为概率分布

# ======================
# 绘图逻辑
# ======================
fig, axs = plt.subplots(2, 2, figsize=figsize, 
                       gridspec_kw={'width_ratios': [3, 2]})  # 调整左右列宽度比

# 绘制左侧折线图
for row_idx, dataset in enumerate(datasets):
    ax = axs[row_idx, 0]
    
    # 绘制四条样本曲线
    for sample_idx in range(num_samples):
        x = np.arange(num_time_points)
        y = line_data[row_idx, sample_idx]
        ax.plot(x, y, 
               color=line_colors[sample_idx],
               linewidth=1.5,
               marker='o',  # 添加数据点标记
               markersize=4,
               label=f'S{sample_idx+1}')
    
    # 设置子图样式
    ax.set_title(f"Four samples from {dataset}", pad=12)
    ax.set_xlabel('Time Fragments')
    ax.set_ylabel('Value')
    ax.grid(True, linestyle='--', alpha=0.6)
    # ax.legend(loc='upper right', framealpha=0.9)
    ax.legend(
    loc='upper center',   # 定位锚点在顶部中心
    bbox_to_anchor=(0.5, -0.15),  # 坐标偏移（水平居中，向下偏移15%）
    ncol=4,              # 图例分4列排列
    frameon=False,       # 去掉图例边框
    fontsize=9           # 调小字号
)

# 绘制右侧热力图
for row_idx, dataset in enumerate(datasets):
    ax = axs[row_idx, 1]
    
    # 显示矩阵数据
    im = ax.imshow(heatmap_data[row_idx], 
                  cmap=cmap_weights,
                  aspect='auto',
                  vmin=0, vmax=1)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, 
                        fraction=0.046, pad=0.04)
    cbar.set_label('Weight', rotation=270, labelpad=15)
    
    # 设置坐标标签
    ax.set_title(f"Distribution weights of {dataset}", pad=12)
    ax.set_xticks(np.arange(num_fragments))
    ax.set_yticks(np.arange(num_samples))
    ax.set_xticklabels([f'D{i+1}' for i in range(num_fragments)])
    ax.set_yticklabels([f'S{i+1}' for i in range(num_samples)])
    
    # 添加数值标签（可选）
    # for i in range(num_samples):
    #     for j in range(num_fragments):
    #         text = ax.text(j, i, f"{heatmap_data[row_idx, i, j]:.2f}",
    #                        ha="center", va="center", color="w")

# ======================
# 全局调整
# ======================
plt.tight_layout(pad=3.0)  # 增加子图间距
plt.subplots_adjust(wspace=0.3, hspace=0.35)  # 调整行列间距

# 保存图片（可选）
plt.savefig('analysis_plot.png', dpi=300, bbox_inches='tight')

plt.show()