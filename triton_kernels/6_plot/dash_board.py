import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import numpy as np

def plot_real_dashboard(json_path="real_metrics.json"):
    # 读取真实落盘数据
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 过滤掉开头 QPS 为 0 的预热空白期
    data = [d for d in data if d['qps'] > 0]
    
    time_steps = np.arange(len(data))
    qps = [d['qps'] for d in data]
    avg_latency = [d['avg_latency'] for d in data]
    p95_latency = [d['p95_latency'] for d in data]
    gpu_mem = [d['gpu_mem_pct'] for d in data]

    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  
    plt.rcParams['axes.unicode_minus'] = False 

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], hspace=0.35, wspace=0.15)

    # ==========================
    # 面板 1: 真实 GPU 显存
    # ==========================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_steps, gpu_mem, color='#984ea3', linewidth=2, label='显存占用率 (%)')
    ax1.fill_between(time_steps, gpu_mem, color='#984ea3', alpha=0.1)
    ax1.axhline(y=90, color='red', linestyle='--', linewidth=1.5, label='OOM 预警水位 (90%)')
    ax1.set_title('GPU 显存动态占用监控 (实测)', fontsize=13, pad=10)
    ax1.set_ylabel('显存占用率 (%)', fontsize=11)
    ax1.set_ylim(0, 100)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower left', fontsize=10)

    # ==========================
    # 面板 2: 真实 QPS
    # ==========================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time_steps, qps, color='#4daf4a', linewidth=2, label='实时 QPS')
    ax2.fill_between(time_steps, qps, color='#4daf4a', alpha=0.1)
    ax2.set_title('并发请求吞吐量监控 (实测)', fontsize=13, pad=10)
    ax2.set_ylabel('吞吐量 (Requests/sec)', fontsize=11)
    ax2.set_ylim(0, max(qps) * 1.2 if max(qps) > 0 else 100)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower left', fontsize=10)

    # ==========================
    # 面板 3: 真实耗时与 P95 长尾
    # ==========================
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(time_steps, avg_latency, color='#377eb8', linewidth=2, label='平均耗时 (Avg Latency)')
    ax3.plot(time_steps, p95_latency, color='#e41a1c', linewidth=1.5, linestyle='-.', label='P95 长尾耗时 (95th Percentile)')
    ax3.fill_between(time_steps, avg_latency, p95_latency, color='#e41a1c', alpha=0.08, label='时延长尾区间')
    
    ax3.set_title('端到端推理耗时与长尾效应监控 (实测)', fontsize=13, pad=10)
    ax3.set_xlabel('监控时长 (秒)', fontsize=11)
    ax3.set_ylabel('请求耗时 (ms)', fontsize=11)
    ax3.set_xlim(0, max(time_steps) if len(time_steps) > 0 else 10)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc='upper left', fontsize=10)

    plt.savefig('real_monitoring_dashboard.pdf', format='pdf', dpi=300, bbox_inches='tight')
    print("✅ 真实监控面板矢量图已生成: real_monitoring_dashboard.pdf")

if __name__ == "__main__":
    plot_real_dashboard()