import numpy as np
import matplotlib.pyplot as plt

# 设置学术图表常用的字体和大小
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300 
})

# 1. 构造模拟数据
num_splits = np.array([1, 2, 4, 8, 16, 32, 64])

comp_latency = 120 / num_splits + 5 
reduction_overhead = 1.5 * num_splits 
total_latency = comp_latency + reduction_overhead

# ================= 新增：模拟误差/标准差数据 =================
# 假设系统延迟存在波动，我们为 Total Latency 模拟一个误差（例如波动范围为总延迟的 10%）
latency_std = total_latency * 0.10  # 这里的 0.10 可以替换为你实际测试算出的标准差数组

# 计算误差带的上下界
upper_bound = total_latency + latency_std
lower_bound = total_latency - latency_std
# ==============================================================

# 2. 开始绘图
fig, ax = plt.subplots(figsize=(8, 5))

# 绘制三条线
ax.plot(num_splits, comp_latency, marker='s', linestyle='--', color='#1f77b4', 
        label='Computation Latency (Long-tail Hidden)')
ax.plot(num_splits, reduction_overhead, marker='^', linestyle='-.', color='#ff7f0e', 
        label='Global Reduction Overhead (Phase 2)')
ax.plot(num_splits, total_latency, marker='o', linestyle='-', color='#2ca02c', linewidth=2.5,
        label='Total Latency (Mean)')

# ================= 新增：绘制误差带 =================
# 使用 fill_between 将下界和上界之间的区域填充为半透明阴影
ax.fill_between(num_splits, lower_bound, upper_bound, 
                color='#2ca02c', alpha=0.2,  # alpha 控制透明度，0.2表示比较透明
                label='Total Latency $\pm 1\sigma$ (Variance)')
# ====================================================

# 3. 寻找并标注最优驻点
optimal_idx = np.argmin(total_latency)
optimal_x = num_splits[optimal_idx]
optimal_y = total_latency[optimal_idx]

ax.annotate('Optimal Point\n(Sweet Spot)', 
            xy=(optimal_x, optimal_y), 
            xytext=(optimal_x + 5, optimal_y + 20),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
            fontsize=11, weight='bold')

ax.axvline(x=optimal_x, color='gray', linestyle=':', alpha=0.7)

# 4. 图表细节设置
ax.set_xscale('log', base=2) 
ax.set_xticks(num_splits)
ax.set_xticklabels(num_splits)

ax.set_xlabel(r'Split-K Concurrency ($\mathtt{num\_splits}$)')
ax.set_title(r'Performance Impact of $\mathtt{num\_splits}$ Tuning', pad=15)
ax.set_ylabel('Latency (ms)')

ax.grid(True, which='both', linestyle='--', alpha=0.5)

# 设置图例 (由于多了一个误差带的图例，适当调整位置)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

plt.tight_layout()
plt.savefig('num_splits_tuning_with_error_band.png', format='png', bbox_inches='tight') 