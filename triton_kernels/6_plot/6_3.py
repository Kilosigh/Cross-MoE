import matplotlib.pyplot as plt
import numpy as np

def plot_real_benchmark():
    # 设置全局字体，确保中文正常显示 (根据系统可替换为 'SimHei', 'Songti SC' 等)
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  
    plt.rcParams['axes.unicode_minus'] = False 

    # 提取的实测并发数
    concurrency = [1, 5, 50, 100, 300, 500]

    # 原生引擎 (NATIVE) 测试数据 (使用最新版本)
    native_qps = [85.25, 88.82, 95.26, 105.21, 116.45, 113.30]
    native_latency = [11.70, 56.23, 521.15, 937.49, 2474.87, 4133.56]

    # Triton 引擎 - Batch-Size = 64 (上一轮数据)
    triton_bs64_qps = [9.96, 38.28, 141.36, 179.07, 205.39, 205.27]
    triton_bs64_latency = [100.38, 130.59, 353.42, 555.97, 1431.93, 2356.11]

    # Triton 引擎 - Batch-Size = 4 (本轮最新数据)
    triton_bs4_qps = [38.46, 101.51, 183.64, 180.62, 185.13, 183.63]
    triton_bs4_latency = [25.96, 49.22, 271.08, 548.49, 1578.16, 2624.14]

    # 创建 1x2 的双子图画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ==========================
    # 子图 1: 平均响应时延 (Latency)
    # ==========================
    ax1.plot(concurrency, native_latency, marker='o', markersize=8, linewidth=2, 
             linestyle='-', color='#d95f02', label='PyTorch 原生引擎')
    
    # 增加两条不同 Batch-Size 的 Triton 曲线
    ax1.plot(concurrency, triton_bs64_latency, marker='s', markersize=8, linewidth=2, 
             linestyle='-', color='#1b9e77', label='Triton 优化引擎 (B=64)')
    ax1.plot(concurrency, triton_bs4_latency, marker='^', markersize=8, linewidth=2, 
             linestyle='-', color='#7570b3', label='Triton 优化引擎 (B=4)')
    
    ax1.set_title('端到端平均响应延迟随并发量变化', fontsize=14, pad=15)
    ax1.set_xlabel('并发请求数量', fontsize=12)
    ax1.set_ylabel('平均延迟 (Latency / ms) [对数刻度]', fontsize=12)
    
    # 启用对数刻度
    ax1.set_yscale('log')
    ax1.set_yticks([10, 100, 1000, 5000, 10000])
    ax1.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    
    ax1.grid(True, which="both", linestyle='--', alpha=0.5)
    ax1.legend(fontsize=11)

    # ==========================
    # 子图 2: 系统吞吐量 (QPS)
    # ==========================
    ax2.plot(concurrency, native_qps, marker='o', markersize=8, linewidth=2, 
             linestyle='-', color='#d95f02', label='PyTorch 原生引擎')
             
    # 增加两条不同 Batch-Size 的 Triton 曲线
    ax2.plot(concurrency, triton_bs64_qps, marker='s', markersize=8, linewidth=2, 
             linestyle='-', color='#1b9e77', label='Triton 优化引擎 (B=64)')
    ax2.plot(concurrency, triton_bs4_qps, marker='^', markersize=8, linewidth=2, 
             linestyle='-', color='#7570b3', label='Triton 优化引擎 (B=4)')
    
    ax2.set_title('系统吞吐量随并发量变化', fontsize=14, pad=15)
    ax2.set_xlabel('并发请求数量', fontsize=12)
    ax2.set_ylabel('吞吐量 (QPS / 次/秒)', fontsize=12)
    
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=11)

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig('system_benchmark_real.pdf', format='pdf', dpi=300, bbox_inches='tight')
    print("实测性能评测图已生成: system_benchmark_real.pdf")

if __name__ == "__main__":
    plot_real_benchmark()