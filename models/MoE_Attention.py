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
from triton_kernels.split_k_fw_plus_bwd import FinalCrossMoEMultiHeadAttentionFunc
from utils.tools import nan_debugging_report
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
                       Sq: int,
                       Sk: int,
                       d_model: int,
                       assignments: torch.Tensor,
                       M: int):
        """更新QK计算统计并保存当前batch信息"""
        if not self.enabled:
            return
        
        query_assignments = assignments[:, :Sq]
        key_assignments = assignments[:, Sq:Sq + Sk]
        
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
        native_flops = batch_size * Sq * Sk * (2 * d_model - 1)
        theoretical_connections = batch_size * Sq * Sk
        
        # 当前batch的统计
        batch_stats = {
            'batch_id': batch_id,
            'batch_size': batch_size,
            'Sq': Sq,
            'Sk': Sk,
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
        
    def update(self, assignments: torch.Tensor, Sq: int, Sk: int):
        """
        更新单个batch的统计
        
        Args:
            assignments: [batch_size, Sq + Sk] 簇分配
            Sq: 查询序列长度
            Sk: 键序列长度
        """
        batch_size = assignments.shape[0]
        
        # 分离Q和K的分配
        query_assignments = assignments[:, :Sq]  # [batch_size, Sq]
        key_assignments = assignments[:, Sq:Sq + Sk]  # [batch_size, Sk]
        
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
                    'q_ratio': q_count / Sq if Sq > 0 else 0,
                    'k_ratio': k_count / Sk if Sk > 0 else 0
                }
            batch_cluster_stats.append(cluster_info)
        
        self.batch_stats.append({
            'batch_size': batch_size,
            'Sq': Sq,
            'Sk': Sk,
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
            'Sq': latest['Sq'],
            'Sk': latest['Sk'],
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
        print(f"Batch Size: {batch_data['batch_size']}, Seq Len Q: {batch_data['Sq']}, Seq Len K: {batch_data['Sk']}")
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
                 init_data=None, num_heads=1, expert_hidden_dim=None, 
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

        self.use_triton = getattr(configs, 'use_triton', True)
        self.num_splits = getattr(configs, 'triton_num_splits', 2)

        self.d_model = d_model
        self.head_dim = d_model
        self.M = num_clusters
        self.lambda_ = update_weight
        self.use_trainable_center = use_trainable_center
        self.num_heads = num_heads
        self.use_learnable_text_emb = configs.use_learnable_text_emb

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)

        self.shared_router = getattr(configs, 'shared_router', True)  # 默认共享router
        self.shared_experts = getattr(configs, 'shared_experts', False)  # 默认不共享专家组

        self.stats = AttentionStatistics(output_dir=f"./attn_results/MoE_attn/overhead/{configs.model}/num_tx_experts_{configs.num_tx_experts}/")

        self.enable_token_stats = enable_token_stats
        if self.enable_token_stats:
            self.token_stats = ClusterTokenStatistics(num_clusters)
            print(f"  - Token统计: 已启用")
            
        # 设置专家网络隐藏层维度
        # expert_hidden_dim = expert_hidden_dim or 4 * d_model

        expert_hidden_dim = d_model

        self.H = 6
        d_model_head = d_model // self.H 
        # ----------------------------
        # Router (miu) 参数
        # ----------------------------
        if self.shared_router:
            # Q 和 K 共享同一个 miu
            self.miu = nn.Parameter(torch.empty(num_heads, num_clusters, d_model))
        else:
            # Q 和 K 各自拥有 miu
            self.miu_Q = nn.Parameter(torch.empty(num_heads, num_clusters, d_model))
            self.miu_K = nn.Parameter(torch.empty(num_heads, num_clusters, d_model))

        # ----------------------------
        # Experts 参数
        # ----------------------------
        if self.shared_experts:
            # Q 和 K 共享专家
            self.experts_weight = nn.Parameter(torch.empty(self.H, num_clusters, d_model_head, d_model_head))
            self.experts_bias = nn.Parameter(torch.empty(self.H, num_clusters, d_model_head))
        else:
            # Q 和 K 各自拥有专家
            self.experts_Q_weight = nn.Parameter(torch.empty(self.H, num_clusters, d_model_head, d_model_head))
            self.experts_Q_bias = nn.Parameter(torch.empty(self.H, num_clusters, d_model_head))
            self.experts_K_weight = nn.Parameter(torch.empty(self.H, num_clusters, d_model_head, d_model_head))
            self.experts_K_bias = nn.Parameter(torch.empty(self.H, num_clusters, d_model_head))

        if self.use_triton:
            self.V_weight = nn.Parameter(torch.randn(d_model, expert_hidden_dim) * (d_model ** -0.5))
            
        
        if not self.configs.is_training:
            return 
        # 使用k-means初始化聚类中心
        print(f"use_k_means_init:{configs.use_k_means_init}")


        # 激活和 dropout
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)

        # 输出投影
        self.output_projection = nn.Linear(num_heads * expert_hidden_dim, d_model)

        if init_data is not None:
            self._init_with_kmeans(init_data, n_init=kmeans_n_init, max_iter=kmeans_max_iter)

        else:
            # 初始化参数
            if self.shared_router:
                nn.init.normal_(self.miu, mean=0.0, std=0.02)
            else:
                nn.init.normal_(self.miu_Q, mean=0.0, std=0.02)
                nn.init.normal_(self.miu_K, mean=0.0, std=0.02)

        if self.shared_experts:
            nn.init.normal_(self.experts_weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.experts_bias)
        else:
            nn.init.normal_(self.experts_Q_weight, mean=0.0, std=0.02)
            nn.init.normal_(self.experts_K_weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.experts_Q_bias)
            nn.init.zeros_(self.experts_K_bias)

        # if torch.isnan(self.miu_Q).any() or torch.isnan(self.miu_K).any():
        #     print("❌ 严重错误: k-means 初始化后参数包含 NaN!")

        # 打印配置信息
        print(f"初始化MoE-Enhanced Clustered Attention:")
        print(f"  - 簇核心更新方式: {'可训练参数' if use_trainable_center else 'EMA更新'}")
        print(f"  - 簇数量: {num_clusters}")
        print(f"  - 更新权重: {update_weight}")

    def _get_router_parameters(self):
        """获取router参数，处理共享/非共享情况"""
        if self.shared_router:
            if self.use_trainable_center:
                miu_for_routing = self.miu
            else:
                miu_for_routing = self.miu.detach()
            return miu_for_routing, miu_for_routing
        else:
            if self.use_trainable_center:
                miu_Q = self.miu_Q
                miu_K = self.miu_K
            else:
                miu_Q = self.miu_Q.detach()
                miu_K = self.miu_K.detach()
            return miu_Q, miu_K

    def _compute_router_scores(self, x, miu, batch_size, seq_len):
        """计算路由得分"""
        miu_expanded = miu.unsqueeze(0).expand(batch_size, -1, -1, -1)
        scores = torch.matmul(x, miu_expanded.transpose(-2, -1)) / (self.d_model ** 0.5)
        return scores


    def apply_experts_batch(self, x, assignments, is_query=True):
        B, S, D = x.shape
        H = self.num_heads
        M = self.M
        Dh = self.head_dim
        device = x.device

        if self.shared_experts:
            W_full = self.experts_weight   # (H, M, D, Dh)
            b_full = self.experts_bias     # (H, M, Dh)
        else:
            if is_query:
                W_full = self.experts_Q_weight
                b_full = self.experts_Q_bias
            else:
                W_full = self.experts_K_weight
                b_full = self.experts_K_bias

        # 使用与第一个版本相同的张量形状
        x_exp = x.unsqueeze(1).expand(B, H, S, D)  # (B, H, S, D)
        x_flat = x_exp.reshape(B * H * S, D)       # (B*H*S, D)
        if not is_query:
            pass
            # print(assignments.shape)
            # print(B,H,S)
        assignments_flat = assignments.reshape(B * H * S)  # (B*H*S,)

        output_flat = torch.zeros(B * H * S, Dh, dtype=x.dtype, device=device)

        # 遍历每个专家
        for m in range(M):
            # 找出分配到专家m的所有token
            mask = (assignments_flat == m)
            if not mask.any():
                continue
                
            indices = torch.nonzero(mask, as_tuple=True)[0]
            x_selected = x_flat[indices]  # (N, D)
            
            # 获取对应的头索引
            head_indices = indices // (B * S)  # 计算每个token属于哪个头
            
            # 为每个选择的token应用对应的专家
            results = []
            for idx, h in zip(indices, head_indices):
                W_hm = W_full[h, m]  # (D, Dh)
                b_hm = b_full[h, m]  # (Dh,)
                result = F.linear(x_flat[idx:idx+1], W_hm.T, b_hm)
                results.append(result)
            
            if results:
                transformed = torch.cat(results, dim=0)
                transformed = self.activation(transformed)
                transformed = self.dropout(transformed)
                output_flat[indices] = transformed
        

        # 重塑回原始形状
        output = output_flat.reshape(B, H, S, Dh)
        return output
    
    def assign_clusters(self, Q, K):
        B, Sq, D = Q.shape
        _, Sk, _ = K.shape
        H = self.num_heads
        M = self.M
        device = Q.device

        Q_exp = Q.unsqueeze(1).expand(B, H, Sq, D)
        K_exp = K.unsqueeze(1).expand(-1, H, Sk, D)
        # exit()
        
        # if self.use_learnable_text_emb:
        #     K_exp = K_exp[0:B, :, :, :]

        if self.shared_router:
            if self.configs.is_debugging:
                print("miu stats:", self.miu.min().item(), self.miu.max().item(), self.miu.isnan().sum().item())
            x_combined = torch.cat([Q_exp, K_exp], dim=2)  # (B, H, Sq+Sk, D)
            miu = self.miu
            miu_exp = miu.unsqueeze(0).expand(B, H, M, D)
            logits = torch.matmul(x_combined, miu_exp.transpose(-2, -1)) / (D ** 0.5 + 1e-6)
            assignments = torch.argmax(logits, dim=-1)  # (B, H, Sq+Sk)
            return assignments, logits
        else:
            if self.configs.is_debugging:
                print("miu_Q stats:", self.miu_Q.min().item(), self.miu_Q.max().item(), self.miu_Q.isnan().sum().item())
                print("miu_K stats:", self.miu_K.min().item(), self.miu_K.max().item(), self.miu_K.isnan().sum().item())
            miu_Q_exp = self.miu_Q.unsqueeze(0).expand(B, H, M, D)
            miu_K_exp = self.miu_K.unsqueeze(0).expand(B, H, M, D)

            logits_Q = torch.matmul(Q_exp, miu_Q_exp.transpose(-2, -1)) / (D ** 0.5 + 1e-6)
            logits_K = torch.matmul(K_exp, miu_K_exp.transpose(-2, -1)) / (D ** 0.5 + 1e-6)

            assign_Q = torch.argmax(logits_Q, dim=-1)
            assign_K = torch.argmax(logits_K, dim=-1)

            assignments = torch.cat([assign_Q, assign_K], dim=2)
            logits = torch.cat([logits_Q, logits_K], dim=2)
            return assignments, logits
        
    def compute_balance_loss(self, router_logits, assignments):
        """
        Compute auxiliary load balancing loss per head.
        
        Args:
            router_logits: (B, H, S, M) - raw logits from router (before softmax)
            assignments:   (B, H, S)   - hard cluster assignments (long tensor)
        
        Returns:
            balance_loss: scalar tensor (mean over heads)
        """
        B, H, S, M = router_logits.shape
        device = router_logits.device
        if self.configs.is_debugging and torch.isnan(router_logits).any() or torch.isinf(router_logits).any():
            print("Warning: router_logits contains NaN or Inf!")

        # 1. Compute f_i: fraction of tokens assigned to each expert (per head)
        # assignments: (B, H, S)
        # Expand to one-hot: (B, H, S, M)
        assignment_one_hot = F.one_hot(assignments, num_classes=M).float()  # (B, H, S, M)
        f_i = assignment_one_hot.mean(dim=(0, 2))  # (H, M) — average over batch and seq

        # 2. Compute p_i: mean router probability for each expert (per head)
        router_probs = F.softmax(router_logits, dim=-1)  # (B, H, S, M)
        p_i = router_probs.mean(dim=(0, 2))  # (H, M)

        # 3. Balance loss per head: M * sum_{i=0}^{M-1} f_i * p_i
        balance_loss_per_head = M * (f_i * p_i).sum(dim=-1)  # (H,)

        # 4. Average over heads
        balance_loss = balance_loss_per_head.mean()

        return balance_loss

    def _init_with_kmeans(self, init_data, n_init=30, max_iter=500, cache_path=None):  # 增加默认值
        """改进的k-means初始化，解决中心点聚集问题，支持缓存"""
        print("使用改进的k-means算法初始化聚类中心")
        
        if isinstance(init_data, torch.Tensor):
            data_np = init_data.cpu().numpy()
        else:
            data_np = np.array(init_data)
        
        self.data_np = data_np

        # 检查缓存文件是否存在
        if cache_path is None:
            cache_path = f"./checkpoints/k_means_cache/num_cluster_{self.M}/d_model_{self.d_model}/{os.path.basename(self.configs.root_path)}"
        # 确保缓存目录存在
        cache_dir = os.path.dirname(cache_path)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = cache_path
        if os.path.exists(cache_file):
            print(f"找到缓存文件: {cache_file}")
            try:
                # 加载缓存的聚类中心
                cache_data = torch.load(cache_file)
                cluster_centers = cache_data['cluster_centers']
                cluster_labels = cache_data['cluster_labels']
                self.M = cache_data['M']
                
                print(f"从缓存加载 {self.M} 个聚类中心")
                
                # 转换为PyTorch张量
                cluster_centers_tensor = torch.tensor(cluster_centers, dtype=torch.float32)
                cluster_centers_tensor = cluster_centers_tensor.unsqueeze(0).repeat(self.num_heads, 1, 1)

                # 设置聚类中心
                if self.shared_router:
                    if isinstance(self.miu, nn.Parameter):
                        self.miu.data.copy_(cluster_centers_tensor)
                    else:
                        self.miu.copy_(cluster_centers_tensor)
                else:
                    if isinstance(self.miu_Q, nn.Parameter):
                        self.miu_Q.data.copy_(cluster_centers_tensor)
                        self.miu_K.data.copy_(cluster_centers_tensor.clone())
                    else:
                        self.miu_Q.copy_(cluster_centers_tensor)
                        self.miu_K.copy_(cluster_centers_tensor.clone())
                
                # 生成t-SNE图（如果需要）
                if self.plot_tsne:
                    folder_path = './tsne_results/init_stage/' + f"num_centers={self.configs.num_tx_experts}/"
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)

                    visualizer = TSNEVisualizer(output_dir=folder_path, M=self.M)
                    file_name = f"{self.configs.model_id}_" + f"{self.configs.num_tx_experts}"

                    visualizer.generate_tsne_plot(
                        data=data_np,
                        centers=cluster_centers,
                        labels=cluster_labels,
                        min_distance=0.5,
                        title="Cached Clustering",
                        filename=file_name
                    )
                
                print("成功从缓存加载聚类中心")
                return  # 直接返回，跳过后续计算
                
            except Exception as e:
                print(f"加载缓存失败: {e}，将重新计算聚类中心")
                cache_file = None  # 标记为需要重新计算

        # 如果没有缓存或加载失败，执行原始的计算逻辑
        print("未找到缓存文件或缓存加载失败，开始计算聚类中心...")
        
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
        
        # 7. 缓存结果（如果指定了缓存路径）
        if cache_file is not None:
            try:
                cache_data = {
                    'cluster_centers': cluster_centers,
                    'cluster_labels': cluster_labels,
                    'M': self.M,
                    'timestamp': time.time()
                }
                torch.save(cache_data, cache_file)
                print(f"聚类中心已缓存到: {cache_file}")
            except Exception as e:
                print(f"缓存保存失败: {e}")
        
        # 8. 转换为PyTorch张量
        cluster_centers_tensor = torch.tensor(cluster_centers, dtype=torch.float32)

        if self.shared_router:
            if isinstance(self.miu, nn.Parameter):
                self.miu.data.copy_(cluster_centers_tensor)
            else:
                self.miu.copy_(cluster_centers_tensor)
        else:
            # 非共享router时，Q和K使用相同的初始化簇核心
            if isinstance(self.miu_Q, nn.Parameter):
                self.miu_Q.data.copy_(cluster_centers_tensor)
                self.miu_K.data.copy_(cluster_centers_tensor.clone())
            else:
                self.miu_Q.copy_(cluster_centers_tensor)
                self.miu_K.copy_(cluster_centers_tensor.clone())
        
        print(f"成功初始化{self.M}个聚类中心")
        
        # 9. 生成改进的t-SNE图
        if self.plot_tsne:
            folder_path = './tsne_results/init_stage/' + f"num_centers={self.configs.num_tx_experts}/"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            visualizer = TSNEVisualizer(output_dir=folder_path, M=self.M)
            file_name = f"{self.configs.model_id}_" + f"{self.configs.num_tx_experts}"

            visualizer.generate_tsne_plot(
                data=data_np,
                centers=cluster_centers,
                labels=cluster_labels,
                min_distance=0.5,
                title="Initial Clustering",
                filename=file_name
            )

    def _update_cluster_centers(self, Q, assignments, batch_size, Sq):
        """更新簇核心"""
        if self.shared_router:
            # 共享router：使用Q的分配更新miu
            cluster_queries = [[] for _ in range(self.M)]
            query_assignments = assignments[:, :Sq]
            
            for b in range(batch_size):
                for i in range(Sq):
                    m = query_assignments[b, i].item()
                    cluster_queries[m].append(Q[b, i])
            
            new_miu = self.miu.clone()
            for m in range(self.M):
                if cluster_queries[m]:
                    queries_tensor = torch.stack(cluster_queries[m])
                    new_centroid = queries_tensor.mean(dim=0)
                    new_miu[m] = (1 - self.lambda_) * new_miu[m] + self.lambda_ * new_centroid
            
            self.miu.copy_(new_miu)
        else:
            # 不共享router：分别更新miu_Q和miu_K
            # 更新Q的簇核心
            cluster_queries_q = [[] for _ in range(self.M)]
            query_assignments = assignments[:, :Sq]
            
            for b in range(batch_size):
                for i in range(Sq):
                    m = query_assignments[b, i].item()
                    cluster_queries_q[m].append(Q[b, i])
            
            new_miu_Q = self.miu_Q.clone()
            for m in range(self.M):
                if cluster_queries_q[m]:
                    queries_tensor = torch.stack(cluster_queries_q[m])
                    new_centroid = queries_tensor.mean(dim=0)
                    new_miu_Q[m] = (1 - self.lambda_) * new_miu_Q[m] + self.lambda_ * new_centroid
            
            self.miu_Q.copy_(new_miu_Q)

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
            Q_prime: 变换后的查询 [batch_size, Sq, d_model]
            K_prime: 变换后的键 [batch_size, Sk, d_model]
            V: 值矩阵 [batch_size, Sk, d_model]
            assignments: 专家分配 [batch_size, Sq + Sk]
            M: 簇的数量
        
        Returns:
            attention_output: 注意力输出 [batch_size, Sq, d_model]
            attention_weights: 注意力权重 [batch_size, Sq, Sk]
        """
        batch_size, Sq, d_model = Q_prime.shape
        _, Sk, _ = K_prime.shape
        device = Q_prime.device
        M = self.M
        
        # 初始化输出和注意力权重存储
        attention_output = torch.zeros_like(Q_prime)
        attention_weights = torch.zeros(batch_size, Sq, Sk, device=device)
        
        # 获取查询和键的簇分配
        query_assignments = assignments[:, :Sq]  # [batch_size, Sq]
        key_assignments = assignments[:, Sq:]    # [batch_size, Sk]
        
        # 方法1: 批量计算(更高效)
        for m in range(M):
            # 找出属于簇m的查询和键的mask
            query_mask = (query_assignments == m)  # [batch_size, Sq]
            key_mask = (key_assignments == m)      # [batch_size, Sk]
            
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

    # ---------------------------- 聚类注意力部分 ----------------------------
    def generate_clustered_attention_fast(self, Q_prime, K_prime, V, assignments):
        B,H,Sq,Dh = Q_prime.shape
        _,_,Sk,_ = K_prime.shape

        query_assignments = assignments[:,:,:Sq]
        key_assignments = assignments[:,:,Sq:]

        attn_scores = torch.full((B, H, Sq, Sk), -1e9, device=Q_prime.device, dtype=Q_prime.dtype)
       
        mask = (query_assignments.unsqueeze(-1) == key_assignments.unsqueeze(-2))
        # mask: (B, H, Sq, Sk) - 布尔掩码，标识哪些查询-键对属于同一簇
        if mask.any():
        # 对同类位置计算注意力分数
            same_cluster_idx = mask.nonzero(as_tuple=True)  # 获取同类位置索引

            # 高级索引，同批次，同一个头，的q和k计算
            ## 这里理不清了，ai说是头独立和簇独立的。
            q_sel = Q_prime[same_cluster_idx[0], same_cluster_idx[1], same_cluster_idx[2]]  # [N, Dh]
            k_sel = K_prime[same_cluster_idx[0], same_cluster_idx[1], same_cluster_idx[3]]  # [N, Dh]
            scores = torch.sum(q_sel * k_sel, dim=-1) / (Dh ** 0.5 )  # [N]
            attn_scores[same_cluster_idx] = scores  # 更新同类位置分数

        # softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        attention_output = torch.matmul(attn_probs, V)
       
        attn_probs = attn_probs.detach().cpu().numpy()
        return attention_output, attn_probs

    def generate_clustered_attention_batch_fast_dense(self, Q_prime, K_prime, V_prime, assignments):
        B, H, Sq, Dh = Q_prime.shape
        _, _, Sk, _ = K_prime.shape

        query_assignments = assignments[:, :, :Sq]      # (B, H, Sq)
        key_assignments   = assignments[:, :, Sq:]      # (B, H, Sk)

        # 1. 全量 QK 计算（高度并行）
        attn_scores = torch.matmul(Q_prime, K_prime.transpose(-2, -1)) / (Dh ** 0.5)

        # 2. 构建“不同簇”掩码（注意：是 !=）
        mask = (query_assignments.unsqueeze(-1) != key_assignments.unsqueeze(-2))  # (B, H, Sq, Sk)

        # 3. 关键：用 -inf 掩盖不同簇 → softmax 后概率为 0
        # attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn_scores = attn_scores.masked_fill(mask, -1e9)

        # 4. softmax（自动在同簇 key 上归一化）
        attn_probs = F.softmax(attn_scores, dim=-1)
        if self.configs.is_debugging:
            nan_debugging_report(attn_probs, "attn_probs")
        O = torch.matmul(attn_probs, V_prime)
        attn_probs = attn_probs.detach().cpu().numpy()
        # 5. 输出
        return O, attn_probs


    def forward(self, Q, K, V, idx=0):
        
        B, Sq, D = Q.shape
        _, Sk, _ = K.shape
        H = self.num_heads
        M = self.M
        device = Q.device

        # print("QK")
        # print(Q.shape)
        # print(K.shape)
        # miu_Q, miu_K = self._get_router_parameters()
        if self.configs.is_debugging:
            nan_debugging_report(Q, "init_Q")
            print("miu_Q stats0:", self.miu_Q.min().item(), self.miu_Q.max().item(), self.miu_Q.isnan().sum().item())
            print("miu_K stats0:", self.miu_K.min().item(), self.miu_K.max().item(), self.miu_K.isnan().sum().item())

        orig_dtype = Q.dtype
        if self.use_triton:
            # 🔥 T4 硬件保命符：强行洗成 float16 并刷为连续内存 (Contiguous)
            # 只有严格的 FP16 才能激活 T4 的 Tensor Core MMA 指令，绕过 LLVM 崩溃
            compute_dtype = torch.float16
            
            Q_triton = Q.to(compute_dtype).contiguous()
            K_triton = K.to(compute_dtype).contiguous()

            # 动态获取权重，并全部对齐到 float16
            rq = (self.miu.squeeze(0) if self.shared_router else self.miu_Q.squeeze(0)).to(compute_dtype).contiguous()
            rk = (self.miu.squeeze(0) if self.shared_router else self.miu_K.squeeze(0)).to(compute_dtype).contiguous()

            eq = (self.experts_weight if self.shared_experts else self.experts_Q_weight).transpose(0, 1).to(compute_dtype).contiguous() 
            ek = (self.experts_weight if self.shared_experts else self.experts_K_weight).transpose(0, 1).to(compute_dtype).contiguous() 

            w_v_weight = self.V_weight.to(compute_dtype).contiguous()

            # 调用 Triton Autograd
            O = FinalCrossMoEMultiHeadAttentionFunc.apply(
                Q_triton, K_triton, rq, rk, eq, ek, w_v_weight, self.H, self.num_splits
            )

            # 🔥 算完之后，立刻洗回原精度，不干扰大模型的后续网络
            O = O.to(orig_dtype)
            
            O = self.output_projection(O)
            assignments, router_logits = self.assign_clusters(Q, K)
            balance_loss = self.compute_balance_loss(router_logits, assignments)
            return O + Q, balance_loss


        assignments, router_logits = self.assign_clusters(Q, K)

        if self.configs.is_debugging:
            print("miu_Q stats1:", self.miu_Q.min().item(), self.miu_Q.max().item(), self.miu_Q.isnan().sum().item())
            print("miu_K stats1:", self.miu_K.min().item(), self.miu_K.max().item(), self.miu_K.isnan().sum().item())

        if self.configs.is_debugging and torch.isnan(router_logits).any():
            print("❌ router_logits 包含 NaN!")
            print("router_logits stats:", router_logits.min().item(), router_logits.max().item(), torch.isnan(router_logits).sum().item())

        if self.configs.is_testing:
            self.stats.update_qk_stats(
                        batch_id=idx,
                        batch_size=Q.size(0),
                        Sq=Q.size(1),
                        Sk=K.size(1),
                        d_model=self.d_model,
                        assignments=assignments,
                        M=self.M
                    )
            
        if self.configs.is_debugging:
            print("miu_Q stats2:", self.miu_Q.min().item(), self.miu_Q.max().item(), self.miu_Q.isnan().sum().item())
            print("miu_K stats2:", self.miu_K.min().item(), self.miu_K.max().item(), self.miu_K.isnan().sum().item())

        # Step 2: 专家变换（支持共享/非共享 experts）
        K_prime = self.apply_experts_batch(K, assignments[:, :, Sq:], is_query=False)
        Q_prime = self.apply_experts_batch(Q, assignments[:, :, :Sq], is_query=True)
        V_prime = K_prime

        if self.configs.is_debugging:
            print("miu_Q stats3:", self.miu_Q.min().item(), self.miu_Q.max().item(), self.miu_Q.isnan().sum().item())
            print("miu_K stats3:", self.miu_K.min().item(), self.miu_K.max().item(), self.miu_K.isnan().sum().item())
        
        O, attention_weights = self.generate_clustered_attention_fast(Q_prime, K_prime, V_prime, assignments)

        if self.configs.is_debugging:
            nan_debugging_report(O, "clustered_attention_O")

        combined = O.transpose(1, 2).contiguous().view(B, Sq, self.num_heads * self.head_dim)
        O = self.output_projection(combined)

        if self.configs.is_debugging:
            print("miu_Q stats4:", self.miu_Q.min().item(), self.miu_Q.max().item(), self.miu_Q.isnan().sum().item())
            print("miu_K stats4:", self.miu_K.min().item(), self.miu_K.max().item(), self.miu_K.isnan().sum().item())

        balance_loss = self.compute_balance_loss(router_logits, assignments)

        if self.configs.is_debugging:
            print("miu_Q stats5:", self.miu_Q.min().item(), self.miu_Q.max().item(), self.miu_Q.isnan().sum().item())
            print("miu_K stats5:", self.miu_K.min().item(), self.miu_K.max().item(), self.miu_K.isnan().sum().item())

        if self.plot_attn and self.configs.is_testing:
            self.f_plot_attn(attention_weights, idx)

        
        # 6. 簇核心更新策略
        if self.training and not self.use_trainable_center:
            print("!!!")
            self._update_cluster_centers(Q, assignments, B, Sq)

        if self.configs.is_debugging:
            print("miu_Q stats6:", self.miu_Q.min().item(), self.miu_Q.max().item(), self.miu_Q.isnan().sum().item())
            print("miu_K stats6:", self.miu_K.min().item(), self.miu_K.max().item(), self.miu_K.isnan().sum().item())

        if torch.isnan(balance_loss).any() or torch.isinf(balance_loss).any():
            print("Warning: balance_loss is NaN or Inf!")
            print(balance_loss)

        if self.configs.is_debugging:
            print("-------------------------")
            nan_debugging_report(O, "final_O")
            
        return O+Q, balance_loss

    def finalize_statistics(self, experiment_name: str = None):
        """在训练/测试结束时调用此方法保存统计"""
        return self.stats.save_final_summary(experiment_name)