import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE  # 添加t-SNE库
import seaborn as sns  # 用于美化图表
import os
from visualization.tSNE import TSNEVisualizer
from visualization.attn_heat_map import AttentionHeatmapVisualizer
from typing import Dict, Optional
import time

import json
import torch
from typing import Dict
from pathlib import Path

class AttentionStatistics:
    """集成在MoE-ATTN网络模块中的FLOPs统计类"""
    
    def __init__(self, output_dir: str = "./moe_attn_stats", enabled: bool = True):
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.output_dir_batch = self.output_dir / "batch_wise_info/"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir_batch.mkdir(parents=True, exist_ok=True)
        
        # 重置统计
        self.reset_stats()
    
    def reset_stats(self):
        """重置所有统计"""
        self.stats = {
            'total_batches': 0,
            'total_qk_flops': 0,
            'total_native_flops': 0,
            'total_connections': 0,
            'total_theoretical_connections': 0,
            'batch_details': []  # 存储每个batch的详细信息
        }
    
    def update_qk_stats(self,
                       batch_id: int,
                       batch_size: int,
                       seq_len_q: int,
                       seq_len_k: int,
                       d_model: int,
                       assignments: torch.Tensor,
                       M: int):
        """更新QK计算统计并保存当前batch信息"""
        if not self.enabled:
            return
        
        query_assignments = assignments[:, :seq_len_q]
        key_assignments = assignments[:, seq_len_q:seq_len_q + seq_len_k]
        
        # 计算MoE QK FLOPs
        moe_flops = 0
        connections = 0
        active_clusters = 0
        
        for b in range(batch_size):
            for m in range(M):
                q_count = (query_assignments[b] == m).sum().item()
                k_count = (key_assignments[b] == m).sum().item()
                
                if q_count > 0 and k_count > 0:
                    # QK^T的FLOPs: q_count * k_count * (2 * d_model - 1)
                    moe_flops += q_count * k_count * (2 * d_model - 1)
                    connections += q_count * k_count
                    active_clusters += 1
        
        # 计算Native QK FLOPs
        native_flops = batch_size * seq_len_q * seq_len_k * (2 * d_model - 1)
        theoretical_connections = batch_size * seq_len_q * seq_len_k
        
        # 当前batch的统计
        batch_stats = {
            'batch_id': batch_id,
            'batch_size': batch_size,
            'seq_len_q': seq_len_q,
            'seq_len_k': seq_len_k,
            'd_model': d_model,
            'num_clusters': M,
            'moe_qk_flops': moe_flops,
            'native_qk_flops': native_flops,
            'actual_connections': connections,
            'theoretical_connections': theoretical_connections,
            'sparsity': 1 - (connections / theoretical_connections) if theoretical_connections > 0 else 1.0,
            'flops_reduction': 1 - (moe_flops / native_flops) if native_flops > 0 else 0,
            'cluster_utilization': active_clusters / (batch_size * M) if M > 0 else 0
        }
        
        # 更新累计统计
        self.stats['total_batches'] += 1
        self.stats['total_qk_flops'] += moe_flops
        self.stats['total_native_flops'] += native_flops
        self.stats['total_connections'] += connections
        self.stats['total_theoretical_connections'] += theoretical_connections
        self.stats['batch_details'].append(batch_stats)
        
        # 立即保存当前batch统计到文件
        self._save_batch_stats(batch_stats)
    
    def _save_batch_stats(self, batch_stats: Dict):
        """保存单个batch的统计到文件"""
        batch_file = self.output_dir_batch / f"batch_{batch_stats['batch_id']:06d}.json"
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(batch_stats, f, indent=2, ensure_ascii=False)
    
    def save_final_summary(self, experiment_name: str = "moe_attn_experiment"):
        """保存最终汇总统计（在训练/测试结束时手动调用）"""
        if not self.enabled or self.stats['total_batches'] == 0:
            return
        
        summary = self.get_summary()
        summary['experiment_name'] = experiment_name
        
        # 保存汇总文件
        summary_file = self.output_dir / f"{experiment_name}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 保存详细的batch信息（可选，用于后续分析）
        details_file = self.output_dir / f"{experiment_name}_batch_details.json"
        with open(details_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats['batch_details'], f, indent=2, ensure_ascii=False)
        
        print(f"MoE Attention statistics saved to:")
        print(f"  - Summary: {summary_file}")
        print(f"  - Batch details: {details_file}")
        
        return summary
    
    def get_summary(self) -> Dict:
        """获取统计摘要"""
        if not self.enabled or self.stats['total_batches'] == 0:
            return {}
        
        total_batches = self.stats['total_batches']
        
        # 计算各种比率
        qk_flops_reduction = 1 - (self.stats['total_qk_flops'] / 
                                self.stats['total_native_flops'])
        
        avg_sparsity = 1 - (self.stats['total_connections'] / 
                          self.stats['total_theoretical_connections'])
        
        connection_ratio = (self.stats['total_connections'] / 
                          self.stats['total_theoretical_connections'])
        
        # 计算平均簇利用率（从所有batch中计算）
        utilizations = [batch['cluster_utilization'] for batch in self.stats['batch_details']]
        avg_cluster_utilization = sum(utilizations) / len(utilizations) if utilizations else 0
        
        summary = {
            'total_batches': total_batches,
            'qk_flops_reduction': qk_flops_reduction,
            'average_sparsity': avg_sparsity,
            'connection_ratio': connection_ratio,
            'average_cluster_utilization': avg_cluster_utilization,
            'total_qk_gflops_saved': (self.stats['total_native_flops'] - 
                                    self.stats['total_qk_flops']) / 1e9,
            'total_moe_qk_gflops': self.stats['total_qk_flops'] / 1e9,
            'total_native_qk_gflops': self.stats['total_native_flops'] / 1e9,
        }
        
        return summary
    
    def print_summary(self):
        """打印统计摘要"""
        summary = self.get_summary()
        if not summary:
            print("No statistics available.")
            return
        
        print("\n" + "="*60)
        print("MoE ATTENTION FLOPs STATISTICS")
        print("="*60)
        print(f"Total Batches Processed: {summary['total_batches']}")
        print(f"QK FLOPs Reduction: {summary['qk_flops_reduction']:.1%}")
        print(f"Average Sparsity: {summary['average_sparsity']:.1%}")
        print(f"Connection Ratio: {summary['connection_ratio']:.3f}")
        print(f"Average Cluster Utilization: {summary['average_cluster_utilization']:.1%}")
        print(f"Total QK GFLOPs Saved: {summary['total_qk_gflops_saved']:.2f}")
        print(f"Total MoE QK GFLOPs: {summary['total_moe_qk_gflops']:.2f}")
        print(f"Total Native QK GFLOPs: {summary['total_native_qk_gflops']:.2f}")
        print("="*60 + "\n")

class ClusterTokenStatistics:
    """簇内Token统计类"""
    
    def __init__(self, num_clusters: int):
        self.num_clusters = num_clusters
        self.reset()
    
    def reset(self):
        """重置统计数据"""
        self.batch_stats = []  # 存储每个batch的统计
        self.total_batches = 0
        
    def update(self, assignments: torch.Tensor, seq_len_q: int, seq_len_k: int):
        """
        更新单个batch的统计
        
        Args:
            assignments: [batch_size, seq_len_q + seq_len_k] 簇分配
            seq_len_q: 查询序列长度
            seq_len_k: 键序列长度
        """
        batch_size = assignments.shape[0]
        
        # 分离Q和K的分配
        query_assignments = assignments[:, :seq_len_q]  # [batch_size, seq_len_q]
        key_assignments = assignments[:, seq_len_q:seq_len_q + seq_len_k]  # [batch_size, seq_len_k]
        
        # 统计每个batch中每个簇的Q和K token数量
        batch_cluster_stats = []
        
        for b in range(batch_size):
            cluster_info = {}
            for m in range(self.num_clusters):
                q_count = (query_assignments[b] == m).sum().item()
                k_count = (key_assignments[b] == m).sum().item()
                cluster_info[m] = {
                    'q_tokens': q_count,
                    'k_tokens': k_count,
                    'total_tokens': q_count + k_count,
                    'q_ratio': q_count / seq_len_q if seq_len_q > 0 else 0,
                    'k_ratio': k_count / seq_len_k if seq_len_k > 0 else 0
                }
            batch_cluster_stats.append(cluster_info)
        
        self.batch_stats.append({
            'batch_size': batch_size,
            'seq_len_q': seq_len_q,
            'seq_len_k': seq_len_k,
            'cluster_stats': batch_cluster_stats,
            'timestamp': time.time()
        })
        self.total_batches += 1
        
        return batch_cluster_stats
    
    def get_current_batch_summary(self) -> Dict:
        """获取当前batch的统计摘要"""
        if not self.batch_stats:
            return {}
        
        latest = self.batch_stats[-1]
        summary = {
            'batch_index': self.total_batches - 1,
            'batch_size': latest['batch_size'],
            'seq_len_q': latest['seq_len_q'],
            'seq_len_k': latest['seq_len_k'],
            'clusters': {}
        }
        
        # 计算整个batch的平均值
        for m in range(self.num_clusters):
            q_tokens_list = [batch[m]['q_tokens'] for batch in latest['cluster_stats']]
            k_tokens_list = [batch[m]['k_tokens'] for batch in latest['cluster_stats']]
            
            summary['clusters'][f'cluster_{m}'] = {
                'avg_q_tokens': np.mean(q_tokens_list),
                'avg_k_tokens': np.mean(k_tokens_list),
                'total_q_tokens': sum(q_tokens_list),
                'total_k_tokens': sum(k_tokens_list),
                'min_q_tokens': min(q_tokens_list),
                'max_q_tokens': max(q_tokens_list),
                'min_k_tokens': min(k_tokens_list),
                'max_k_tokens': max(k_tokens_list),
                'active_samples': sum(1 for q, k in zip(q_tokens_list, k_tokens_list) if q > 0 or k > 0)
            }
        
        return summary
    
    def get_global_summary(self) -> Dict:
        """获取所有batch的全局统计摘要"""
        if not self.batch_stats:
            return {}
        
        global_summary = {
            'total_batches': self.total_batches,
            'clusters': {}
        }
        
        # 收集所有batch的数据
        for m in range(self.num_clusters):
            all_q_tokens = []
            all_k_tokens = []
            
            for batch_data in self.batch_stats:
                for sample_stats in batch_data['cluster_stats']:
                    all_q_tokens.append(sample_stats[m]['q_tokens'])
                    all_k_tokens.append(sample_stats[m]['k_tokens'])
            
            global_summary['clusters'][f'cluster_{m}'] = {
                'total_q_tokens': sum(all_q_tokens),
                'total_k_tokens': sum(all_k_tokens),
                'avg_q_tokens_per_sample': np.mean(all_q_tokens) if all_q_tokens else 0,
                'avg_k_tokens_per_sample': np.mean(all_k_tokens) if all_k_tokens else 0,
                'std_q_tokens': np.std(all_q_tokens) if all_q_tokens else 0,
                'std_k_tokens': np.std(all_k_tokens) if all_k_tokens else 0,
                'utilization_rate': sum(1 for q, k in zip(all_q_tokens, all_k_tokens) if q > 0 or k > 0) / len(all_q_tokens) if all_q_tokens else 0
            }
        
        return global_summary
    
    def print_batch_distribution(self, batch_idx: int = -1):
        """打印指定batch的token分布"""
        if not self.batch_stats:
            print("No statistics available.")
            return
        
        batch_data = self.batch_stats[batch_idx]
        print("\n" + "="*80)
        print(f"BATCH {self.total_batches + batch_idx if batch_idx < 0 else batch_idx} TOKEN DISTRIBUTION")
        print("="*80)
        print(f"Batch Size: {batch_data['batch_size']}, Seq Len Q: {batch_data['seq_len_q']}, Seq Len K: {batch_data['seq_len_k']}")
        print("-"*80)
        
        # 打印每个样本的详细分布
        for sample_idx, sample_stats in enumerate(batch_data['cluster_stats']):
            print(f"\nSample {sample_idx}:")
            print(f"{'Cluster':<10} {'Q Tokens':<12} {'K Tokens':<12} {'Total':<10} {'Q Ratio':<10} {'K Ratio':<10}")
            print("-"*70)
            
            for m in range(self.num_clusters):
                stats = sample_stats[m]
                if stats['total_tokens'] > 0:  # 只打印非空簇
                    print(f"Cluster {m:<3} {stats['q_tokens']:<12} {stats['k_tokens']:<12} "
                          f"{stats['total_tokens']:<10} {stats['q_ratio']:<10.2%} {stats['k_ratio']:<10.2%}")
        
        # 打印batch摘要
        summary = self.get_current_batch_summary()
        print("\n" + "-"*80)
        print("BATCH SUMMARY:")
        print(f"{'Cluster':<10} {'Avg Q':<10} {'Avg K':<10} {'Total Q':<10} {'Total K':<10} {'Active':<10}")
        print("-"*70)
        
        for m in range(self.num_clusters):
            cluster_stats = summary['clusters'][f'cluster_{m}']
            print(f"Cluster {m:<3} {cluster_stats['avg_q_tokens']:<10.1f} {cluster_stats['avg_k_tokens']:<10.1f} "
                  f"{cluster_stats['total_q_tokens']:<10} {cluster_stats['total_k_tokens']:<10} "
                  f"{cluster_stats['active_samples']:<10}")
        
        print("="*80 + "\n")
    
    def plot_distribution(self, save_path: Optional[str] = None):
        """绘制token分布图"""
        if not self.batch_stats:
            print("No statistics available for plotting.")
            return
        
        latest = self.batch_stats[-1]
        
        # 准备数据
        cluster_ids = list(range(self.num_clusters))
        avg_q_tokens = []
        avg_k_tokens = []
        
        for m in cluster_ids:
            q_tokens = [batch[m]['q_tokens'] for batch in latest['cluster_stats']]
            k_tokens = [batch[m]['k_tokens'] for batch in latest['cluster_stats']]
            avg_q_tokens.append(np.mean(q_tokens))
            avg_k_tokens.append(np.mean(k_tokens))
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 柱状图
        x = np.arange(len(cluster_ids))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, avg_q_tokens, width, label='Q Tokens', alpha=0.8)
        bars2 = ax1.bar(x + width/2, avg_k_tokens, width, label='K Tokens', alpha=0.8)
        
        ax1.set_xlabel('Cluster ID')
        ax1.set_ylabel('Average Token Count')
        ax1.set_title('Average Token Distribution per Cluster')
        ax1.set_xticks(x)
        ax1.set_xticklabels(cluster_ids)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 饼图 - 显示总token分布
        total_tokens = [q + k for q, k in zip(avg_q_tokens, avg_k_tokens)]
        non_zero_clusters = [(i, t) for i, t in enumerate(total_tokens) if t > 0]
        
        if non_zero_clusters:
            labels = [f'Cluster {i}' for i, _ in non_zero_clusters]
            sizes = [t for _, t in non_zero_clusters]
            
            ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Total Token Distribution Across Clusters')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Distribution plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


class MoEClusteredAttention(nn.Module):
    def __init__(self, configs, d_model, num_clusters, update_weight, 
                 init_data=None, expert_hidden_dim=None, 
                 kmeans_n_init=10, kmeans_max_iter=300,
                 use_trainable_center=False, enable_token_stats=True):
        """
        Args:
            use_trainable_center (bool): 
                True - 使用可训练的簇核心（通过梯度更新）
                False - 使用EMA更新（冻结梯度）
                enable_token_stats (bool): 是否启用token统计功能
        """
        super().__init__()
        self.configs = configs
        self.plot_attn = configs.plot_attn
        self.plot_tsne = configs.plot_tsne  # 保存绘图标志
        self.tsne_path = None

        self.d_model = d_model
        self.M = num_clusters
        self.lambda_ = update_weight
        self.use_trainable_center = use_trainable_center

        self.stats = AttentionStatistics(output_dir=f"./attn_results/MoE_attn/overhead/{configs.model}/num_tx_experts_{configs.num_tx_experts}/")
            
        # 根据配置选择簇核心初始化方式
        if use_trainable_center:
            # 作为可训练参数
            self.miu = nn.Parameter(torch.empty(num_clusters, d_model))
        else:
            # 作为缓冲区（不可训练）
            self.register_buffer('miu', torch.empty(num_clusters, d_model))
        
        self.enable_token_stats = enable_token_stats
        if enable_token_stats:
            self.token_stats = ClusterTokenStatistics(num_clusters)
            print(f"  - Token统计: 已启用")
            
        # 设置专家网络隐藏层维度
        expert_hidden_dim = expert_hidden_dim or 4 * d_model
        
        # 初始化查询专家网络
        self.experts_Q = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(0.1)
                # nn.Linear(expert_hidden_dim, d_model)
            ) for _ in range(num_clusters)
        ])
        
        # 初始化键专家网络
        self.experts_K = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(0.1)
                # nn.Linear(expert_hidden_dim, d_model)
            ) for _ in range(num_clusters)
        ])
        
        if not self.configs.is_training:
            return 
        # 使用k-means初始化聚类中心
        print(f"use_k_means_init:{configs.use_k_means_init}")
        if init_data is not None:
            self._init_with_kmeans(init_data, n_init=kmeans_n_init, max_iter=kmeans_max_iter)
        else:
            nn.init.normal_(self.miu, mean=0.0, std=0.02)

        # 打印配置信息
        print(f"初始化MoE-Enhanced Clustered Attention:")
        print(f"  - 簇核心更新方式: {'可训练参数' if use_trainable_center else 'EMA更新'}")
        print(f"  - 簇数量: {num_clusters}")
        print(f"  - 更新权重: {update_weight}")

    def _init_with_kmeans(self, init_data, n_init=30, max_iter=500):  # 增加默认值
        """改进的k-means初始化，解决中心点聚集问题"""
        print("使用改进的k-means算法初始化聚类中心")
        
        if isinstance(init_data, torch.Tensor):
            data_np = init_data.cpu().numpy()
        else:
            data_np = np.array(init_data)
        
        self.data_np = data_np

        # 1. 数据标准化（解决尺度问题）
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_np)
        
        # 2. 确定最佳簇数量（如果未指定）
        if self.M is None or self.M <= 0:
            print("自动确定最佳簇数量...")
            from sklearn.metrics import silhouette_score
            best_score = -1
            best_k = 1
            k_range = range(2, min(15, len(data_scaled)//10 + 1))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = kmeans.fit_predict(data_scaled)
                
                # 跳过无效聚类
                if len(np.unique(labels)) < 2:
                    continue
                    
                score = silhouette_score(data_scaled, labels)
                print(f"k={k} - 轮廓分数: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    
            self.M = best_k
            print(f"自动确定最佳簇数量: {best_k} (轮廓分数: {best_score:.4f})")
        
        # 3. 使用k-means++初始化（避免中心点聚集）
        print(f"使用k-means++初始化{self.M}个聚类中心 (n_init={n_init})")
        kmeans = KMeans(
            n_clusters=self.M,
            init='k-means++',  # 使用智能初始化
            n_init=n_init,     # 增加初始化次数
            max_iter=max_iter,
            random_state=42
        )
        
        # 4. 多次尝试避免局部最优
        best_inertia = float('inf')
        best_centers = None
        
        for attempt in range(3):  # 最多尝试3次
            kmeans.fit(data_scaled)
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                best_centers = kmeans.cluster_centers_
                print(f"尝试 {attempt+1} - 损失值: {kmeans.inertia_:.2f}")
        
        if best_centers is None:
            best_centers = kmeans.cluster_centers_
        
        # 5. 检查中心点距离
        from sklearn.metrics.pairwise import euclidean_distances
        center_distances = euclidean_distances(best_centers)
        np.fill_diagonal(center_distances, np.inf)  # 忽略对角线
        
        min_distance = np.min(center_distances)
        print(f"最小簇中心距离: {min_distance:.4f}")
        
        if min_distance < 0.1:  # 阈值可调整
            print("警告: 检测到簇中心过于接近，可能表示簇数量过多或数据分布问题")
        
        # 6. 反标准化中心点
        cluster_centers = scaler.inverse_transform(best_centers)
        cluster_labels = kmeans.labels_
        
        # 7. 转换为PyTorch张量
        cluster_centers_tensor = torch.tensor(cluster_centers, dtype=torch.float32)
        
        # 更新模块参数
        if isinstance(self.miu, nn.Parameter):
            self.miu.data.copy_(cluster_centers_tensor)
        else:
            self.miu.copy_(cluster_centers_tensor)
        
        print(f"成功初始化{self.M}个聚类中心")
        
        # 8. 生成改进的t-SNE图
        # if self.plot_tsne:
        #     self._generate_tsne_plot(data_np, cluster_labels, \
        #                                                     cluster_centers, min_distance)       
        #     exit()

        if self.plot_tsne:
            folder_path = './tsne_results/init_stage/' + f"num_centers={self.configs.num_tx_experts}/"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            visualizer = TSNEVisualizer(output_dir=folder_path, M=self.M)
            file_name =  f"{self.configs.model_id}_" + f"{self.configs.num_tx_experts}"

            visualizer.generate_tsne_plot(
                data=data_np,
                centers=cluster_centers,
                labels=cluster_labels,
                min_distance=0.5,
                title="Initial Clustering",
                filename=file_name
            )

            # exit()

    def extract_cluster_centers(self):
        """
        从模型中提取簇核心(miu)并转换为适合 t-SNE 可视化的格式
        
        参数:
        model: 包含簇核心的模型
        
        返回:
        cluster_centers: (num_clusters, d_model) 的 NumPy 数组
        """
        # # 检查簇核心是参数还是缓冲区
        # if hasattr(model, 'miu'):
        #     miu_tensor = model.miu
        # else:
        #     raise AttributeError("模型中没有 'miu' 属性")
        
        miu_tensor = self.miu

        # 确保我们处理的是张量
        if not isinstance(miu_tensor, torch.Tensor):
            raise TypeError("'miu' 应该是 torch.Tensor 类型")
        
        # 处理梯度计算和计算图分离
        if miu_tensor.requires_grad:
            # 如果是可训练参数，需要分离计算图并复制数据
            miu_tensor = miu_tensor.detach()
        
        # 移动到 CPU 并转换为 NumPy 数组
        cluster_centers = miu_tensor.cpu().numpy()
        
        # 验证形状
        if len(cluster_centers.shape) != 2:
            raise ValueError(f"簇核心形状应为 (num_clusters, d_model)，实际为 {cluster_centers.shape}")
        
        return cluster_centers

    def plot_t_SNE(self):
        assert(self.plot_tsne)
        folder_path = './tsne_results/'
        if self.configs.use_trainable_center:
            folder_path += f"trainable/"
        else:
            folder_path += f"untrainable/"
        folder_path  += f"num_centers={self.configs.num_tx_experts}/"

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        visualizer = TSNEVisualizer(output_dir=folder_path, M=self.M)
        file_name =  f"{self.configs.model_id}_" + f"{self.configs.num_tx_experts}_" + f"{self.configs.use_trainable_center}"

        cluster_centers = self.extract_cluster_centers()

        visualizer.generate_tsne_plot(
            data=self.data_np,
            centers=cluster_centers,
            min_distance=0.5,
            title="Initial Clustering",
            filename=file_name
        )

    def f_plot_attn(self, attention_weights, idx):
        folder_path = f"./attn_results/MoE_attn/plot_attn/{self.configs.model}/num_centers:{self.configs.num_tx_experts}/"

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        folder_path +=  f"{self.configs.model_id}_b_out_idx:{idx}"

        attn_visualizer = AttentionHeatmapVisualizer(self.configs)

        k_len = attention_weights.shape[-1]
        plot_attn = attention_weights.reshape(-1, k_len)
        plot_full_path = folder_path
        fig1 = attn_visualizer.plot_attention(
                plot_attn,
                title=f"Attention Heatmap ({plot_attn.shape[0]}x{plot_attn.shape[1]})",
                colormap='cool',
                save_path=plot_full_path,
                show_values=True,
                grid=True,
            )
        
        plt.close(fig1)
        
    def generate_clustered_attention(self, Q_prime, K_prime, V, assignments):
        """
        生成基于聚类的稀疏注意力
        
        Args:
            Q_prime: 变换后的查询 [batch_size, seq_len_q, d_model]
            K_prime: 变换后的键 [batch_size, seq_len_k, d_model]
            V: 值矩阵 [batch_size, seq_len_k, d_model]
            assignments: 专家分配 [batch_size, seq_len_q + seq_len_k]
            M: 簇的数量
        
        Returns:
            attention_output: 注意力输出 [batch_size, seq_len_q, d_model]
            attention_weights: 注意力权重 [batch_size, seq_len_q, seq_len_k]
        """
        batch_size, seq_len_q, d_model = Q_prime.shape
        _, seq_len_k, _ = K_prime.shape
        device = Q_prime.device
        M = self.M
        
        # 初始化输出和注意力权重存储
        attention_output = torch.zeros_like(Q_prime)
        attention_weights = torch.zeros(batch_size, seq_len_q, seq_len_k, device=device)
        
        # 获取查询和键的簇分配
        query_assignments = assignments[:, :seq_len_q]  # [batch_size, seq_len_q]
        key_assignments = assignments[:, seq_len_q:]    # [batch_size, seq_len_k]
        
        # 方法1: 批量计算(更高效)
        for m in range(M):
            # 找出属于簇m的查询和键的mask
            query_mask = (query_assignments == m)  # [batch_size, seq_len_q]
            key_mask = (key_assignments == m)      # [batch_size, seq_len_k]
            
            for b in range(batch_size):
                q_indices = query_mask[b].nonzero(as_tuple=True)[0]
                k_indices = key_mask[b].nonzero(as_tuple=True)[0]
                
                if len(q_indices) == 0 or len(k_indices) == 0:
                    continue
                
                # 提取簇内的Q, K, V
                Q_cluster = Q_prime[b, q_indices]  # [num_q, d_model]
                K_cluster = K_prime[b, k_indices]  # [num_k, d_model]
                V_cluster = V[b, k_indices]        # [num_k, d_model]
                
                # 计算簇内注意力分数
                attn_scores = torch.matmul(Q_cluster, K_cluster.T) / (d_model ** 0.5)  # [num_q, num_k]
                attn_probs = F.softmax(attn_scores, dim=-1)  # [num_q, num_k]
                
                # 计算注意力输出
                cluster_output = torch.matmul(attn_probs, V_cluster)  # [num_q, d_model]
                
                # 将结果写回对应位置
                attention_output[b, q_indices] = cluster_output
                
                # 将注意力权重填入完整的权重矩阵
                # 使用高级索引将簇内的注意力权重填入对应位置
                attention_weights[b, q_indices[:, None], k_indices[None, :]] = attn_probs


        attention_weights = attention_weights.detach().cpu().numpy()
                
        return attention_output, attention_weights

    def forward(self, Q, K, V, idx):
        batch_size, seq_len_q, d = Q.shape
        _, seq_len_k, _ = K.shape
        device = Q.device
        # print("QK")
        # print(Q.shape)
        # print(K.shape)
        
        # 根据配置选择簇核心处理方式
        if self.use_trainable_center:
            # 使用可训练参数（带梯度）
            miu_for_routing = self.miu
        else:
            # 使用分离的miu进行路由计算（不产生梯度）
            miu_for_routing = self.miu.detach()
        
        # 1. 合并Q和K用于路由计算
        x = torch.cat([Q, K], dim=1)  # [batch_size, seq_len_q + seq_len_k, d]
        
        # 2. 计算路由得分
        miu_expanded = miu_for_routing.unsqueeze(0).expand(batch_size, -1, -1)
        scores = torch.matmul(x, miu_expanded.transpose(1, 2)) / (d ** 0.5)  # [batch_size, 2*seq_len, M]
        
        # 3. 确定专家分配
        assignments = torch.argmax(scores, dim=-1)  # [batch_size, 2*seq_len]

        if self.configs.is_testing:
            self.stats.update_qk_stats(
                        batch_id=idx,
                        batch_size=Q.size(0),
                        seq_len_q=Q.size(1),
                        seq_len_k=K.size(1),
                        d_model=self.d_model,
                        assignments=assignments,
                        M=self.M
                    )
        
        # 4. 应用专家变换
        x_transformed = torch.zeros_like(x)
        
        # 处理查询向量 (前seq_len个)
        # print(assignments.shape)
        for m in range(self.M):
            mask = (assignments == m) & (torch.arange(seq_len_q + seq_len_k, device=device) < seq_len_q)
            for b in torch.where(mask.any(dim=1))[0]:
                indices = mask[b].nonzero(as_tuple=True)[0]
                x_transformed[b, indices] = self.experts_Q[m](x[b, indices])
        
        # 处理键向量 (后seq_len个)
        for m in range(self.M):
            mask = (assignments == m) & (torch.arange(seq_len_q + seq_len_k, device=device) >= seq_len_q)
            for b in torch.where(mask.any(dim=1))[0]:
                indices = mask[b].nonzero(as_tuple=True)[0]
                x_transformed[b, indices] = self.experts_K[m](x[b, indices])
        
        # 分离变换后的Q'和K'
        Q_prime = x_transformed[:, :seq_len_q]
        K_prime = x_transformed[:, seq_len_q:]
        V = K_prime
        
        O, attention_weights = self.generate_clustered_attention(Q_prime, K_prime, V, assignments)

        if self.plot_attn and self.configs.is_testing:
            self.f_plot_attn(attention_weights, idx)

        
        # 6. 簇核心更新策略
        if self.training and not self.use_trainable_center:
            # 只在训练模式且使用EMA更新时执行
            # 收集整个批次中每个簇的查询向量
            cluster_queries = [[] for _ in range(self.M)]
            query_assignments = assignments[:, :seq_len_q]
            
            for b in range(batch_size):
                for i in range(seq_len_q):
                    m = query_assignments[b, i].item()
                    cluster_queries[m].append(Q[b, i])
            
            # 更新聚类中心
            new_miu = self.miu.clone()  # 创建副本用于更新
            
            for m in range(self.M):
                if cluster_queries[m]:
                    queries_tensor = torch.stack(cluster_queries[m])
                    new_centroid = queries_tensor.mean(dim=0)
                    new_miu[m] = (1 - self.lambda_) * new_miu[m] + self.lambda_ * new_centroid
            
            # 无梯度更新
            self.miu.copy_(new_miu)
            
        
        return O+Q, 0

    def finalize_statistics(self, experiment_name: str = None):
        """在训练/测试结束时调用此方法保存统计"""
        return self.stats.save_final_summary(experiment_name)