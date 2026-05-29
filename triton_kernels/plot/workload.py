import matplotlib.pyplot as plt
import numpy as np

# ================= 1. 准备数据 =================
# 模拟序列长度 N (呈二次方/指数级增长)
# 这里以 1K, 4K, 16K, 64K, 256K 为例
x_labels = ['1K', '4K', '16K', '64K', '256K']
x_positions = np.arange(len(x_labels))

# 模拟 Native PyTorch 的耗时/内存碎片化开销 (指数级上升)
# 假设单位为毫秒 (ms) 或 显存碎片率 (%)
pytorch_overhead = [10, 45, 250, 1500, 8500] 

# 模拟 本算子 (Our Operator) 的开销 (平缓的线性 O(N) 增长)
our_overhead = [12, 25, 50, 100, 200]

# ================= 2. 全局样式设置 =================
# 推荐使用学术论文常用的字体和字号
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] # 英文常用 Times New Roman
# 若需显示中文字体，可取消下面这行的注释并修改为系统支持的中文字体（如 SimHei, SimSun）
# plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
plt.rcParams['font.size'] = 12

# ================= 3. 创建图表 =================
fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300) # dpi=300 保证论文打印清晰度

# 绘制 Native PyTorch 折线
# 使用 'o' (圆圈) 作为标记，红色虚线
ax.plot(x_positions, pytorch_overhead, 
        marker='o', markersize=8, linestyle='--', color='#d62728', linewidth=2, 
        label='Native PyTorch')

# 绘制 本算子 折线
# 使用 's' (方块) 作为标记，蓝色实线
ax.plot(x_positions, our_overhead, 
        marker='s', markersize=8, linestyle='-', color='#1f77b4', linewidth=2, 
        label='Our Operator (Ours)')

# ================= 4. 设置轴标签与刻度 =================
ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels)
ax.set_xlabel('Sequence Length ($N$)', fontsize=14, fontweight='bold')
ax.set_ylabel('Time Cost / Memory Overhead', fontsize=14, fontweight='bold') # 请根据实际情况修改 Y 轴名称

# ================= 5. 添加网格与图例 =================
ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.7)
ax.legend(loc='upper left', fontsize=12, framealpha=0.9, edgecolor='black')

# ================= 6. 细节优化 =================
# 隐藏右边和上边的边框，使图表更加干净整洁
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 可选：如果 PyTorch 的数值实在太大，可以考虑将 Y 轴设置为对数坐标以看清底层细节
# ax.set_yscale('log')

plt.title('Performance Scaling with Sequence Length $N$', fontsize=15, pad=15)
plt.tight_layout() # 自动调整子图参数，使之填充整个图像区域

# ================= 7. 保存与显示 =================
# plt.savefig('workload_scaling.pdf', format='pdf', b   box_inches='tight') # 保存为矢量图 PDF，最适合 LaTeX
plt.savefig('workload_scaling.png', format='png', bbox_inches='tight', dpi=300)
plt.show()