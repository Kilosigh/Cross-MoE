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
from typing import Dict, Optional
import time

class AttentionStatistics:
    """轻量级注意力统计类，可集成到MoEClusteredAttention中"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.reset_stats()
    
    def reset_stats(self):
        """重置统计数据"""
        self.stats = {
            'total_batches': 0,
            'total_qk_flops': 0,
            'total_native_flops': 0,
            'total_connections': 0,
            'total_theoretical_connections': 0,
            'avg_sparsity': 0,
            'cluster_utilization': [],
            'time_measurements': {
                'routing': [],
                'expert_forward': [],
                'attention': []
            }
        }
    
    def update_qk_stats(self, 
                       batch_size: int,
                       seq_len_q: int,
                       seq_len_k: int,
                       d_model: int,
                       assignments: torch.Tensor,
                       M: int):
        """更新QK计算统计"""
        if not self.enabled:
            return
        
        query_assignments = assignments[:, :seq_len_q]
        key_assignments = assignments[:, seq_len_q:seq_len_q + seq_len_k]
        
        # 计算MoE QK FLOPs
        moe_flops = 0
        connections = 0
        cluster_sizes = []
        
        for b in range(batch_size):
            for m in range(M):
                q_count = (query_assignments[b] == m).sum().item()
                k_count = (key_assignments[b] == m).sum().item()
                
                if q_count > 0 and k_count > 0:
                    # QK^T的FLOPs: q_count * k_count * (2 * d_model - 1)
                    moe_flops += q_count * k_count * (2 * d_model - 1)
                    connections += q_count * k_count
                    cluster_sizes.append((q_count, k_count))
        
        # 计算Native QK FLOPs
        native_flops = batch_size * seq_len_q * seq_len_k * (2 * d_model - 1)
        theoretical_connections = batch_size * seq_len_q * seq_len_k
        
        # 更新统计
        self.stats['total_batches'] += 1
        self.stats['total_qk_flops'] += moe_flops
        self.stats['total_native_flops'] += native_flops
        self.stats['total_connections'] += connections
        self.stats['total_theoretical_connections'] += theoretical_connections
        
        # 更新平均稀疏度
        current_sparsity = 1 - (connections / theoretical_connections)
        self.stats['avg_sparsity'] = (
            (self.stats['avg_sparsity'] * (self.stats['total_batches'] - 1) + current_sparsity) 
            / self.stats['total_batches']
        )
        
        # 记录簇利用率
        utilization = len(cluster_sizes) / M
        self.stats['cluster_utilization'].append(utilization)
    
    def add_time_measurement(self, category: str, time_ms: float):
        """添加时间测量"""
        if not self.enabled:
            return
        if category in self.stats['time_measurements']:
            self.stats['time_measurements'][category].append(time_ms)
    
    def get_summary(self) -> Dict:
        """获取统计摘要"""
        if not self.enabled or self.stats['total_batches'] == 0:
            return {}
        
        summary = {
            'total_batches': self.stats['total_batches'],
            'qk_flops_reduction': 1 - (self.stats['total_qk_flops'] / self.stats['total_native_flops']),
            'average_sparsity': self.stats['avg_sparsity'],
            'connection_ratio': self.stats['total_connections'] / self.stats['total_theoretical_connections'],
            'avg_cluster_utilization': sum(self.stats['cluster_utilization']) / len(self.stats['cluster_utilization'])
                if self.stats['cluster_utilization'] else 0,
            'total_qk_gflops_saved': (self.stats['total_native_flops'] - self.stats['total_qk_flops']) / 1e9,
        }
        
        # 添加时间统计
        for category, times in self.stats['time_measurements'].items():
            if times:
                summary[f'avg_{category}_time_ms'] = sum(times) / len(times)
        
        return summary
    
    def print_summary(self):
        """打印统计摘要"""
        summary = self.get_summary()
        if not summary:
            print("No statistics available.")
            return
        
        print("\n" + "="*60)
        print("MoE ATTENTION QK COMPUTATION STATISTICS")
        print("="*60)
        print(f"Total Batches Processed: {summary['total_batches']}")
        print(f"QK FLOPs Reduction: {summary['qk_flops_reduction']:.1%}")
        print(f"Average Sparsity: {summary['average_sparsity']:.1%}")
        print(f"Connection Ratio: {summary['connection_ratio']:.3f}")
        print(f"Average Cluster Utilization: {summary['avg_cluster_utilization']:.1%}")
        print(f"Total QK GFLOPs Saved: {summary['total_qk_gflops_saved']:.2f}")
        
        # 打印时间统计
        if 'avg_routing_time_ms' in summary:
            print(f"\nTiming Statistics:")
            for key, value in summary.items():
                if 'time_ms' in key:
                    category = key.replace('avg_', '').replace('_time_ms', '')
                    print(f"  Average {category} time: {value:.2f} ms")
        print("="*60 + "\n")

class MoEClusteredAttention(nn.Module):
    def __init__(self, configs, d_model, num_clusters, update_weight, 
                 init_data=None, expert_hidden_dim=None, 
                 kmeans_n_init=10, kmeans_max_iter=300,
                 use_trainable_center=False):
        """
        Args:
            use_trainable_center (bool): 
                True - 使用可训练的簇核心（通过梯度更新）
                False - 使用EMA更新（冻结梯度）
        """
        super().__init__()
        self.configs = configs
        self.d_model = d_model
        self.M = num_clusters
        self.lambda_ = update_weight
        self.use_trainable_center = use_trainable_center
        self.plot_tsne = configs.plot_tsne  # 保存绘图标志

        self.tsne_path = None
            
        
        # 根据配置选择簇核心初始化方式
        if use_trainable_center:
            # 作为可训练参数
            self.miu = nn.Parameter(torch.empty(num_clusters, d_model))
        else:
            # 作为缓冲区（不可训练）
            self.register_buffer('miu', torch.empty(num_clusters, d_model))
        
        # 使用k-means初始化聚类中心
        print(f"use_k_means_init:{configs.use_k_means_init}")
        if init_data is not None:
            self._init_with_kmeans(init_data, n_init=kmeans_n_init, max_iter=kmeans_max_iter)
        else:
            nn.init.normal_(self.miu, mean=0.0, std=0.02)
        
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

    def forward(self, Q, K, V):
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
            mask = (assignments == m) & (torch.arange(seq_len_q + seq_len_k, device=device) >= seq_len_k)
            for b in torch.where(mask.any(dim=1))[0]:
                indices = mask[b].nonzero(as_tuple=True)[0]
                x_transformed[b, indices] = self.experts_K[m](x[b, indices])
        
        # 分离变换后的Q'和K'
        Q_prime = x_transformed[:, :seq_len_q]
        K_prime = x_transformed[:, seq_len_k:]
        V = K_prime
        
        # 5. 聚类注意力计算
        O = torch.zeros_like(Q)
        
        # 预计算簇内键值
        cluster_keys = {}
        cluster_values = {}
        for m in range(self.M):
            # 找出属于当前簇的所有键/值
            mask = assignments[:, seq_len_k:] == m
            if not mask.any():
                continue
                
            # 收集所有批次的簇内键值
            keys = []
            values = []
            for b in range(batch_size):
                if mask[b].any():
                    keys.append(K_prime[b, mask[b]])
                    values.append(V[b, mask[b]])
            cluster_keys[m] = keys
            cluster_values[m] = values
        
        # 为每个查询计算注意力
        for b in range(batch_size):
            for i in range(seq_len_q):
                m_i = assignments[b, i].item()
                
                # 检查簇是否存在且包含当前批次的数据
                if m_i not in cluster_keys or not cluster_keys[m_i] or b >= len(cluster_keys[m_i]):
                    # Fallback: 使用全局平均
                    O[b, i] = V[b].mean(dim=0)
                    continue
                
                # 获取当前批次的簇内键值
                K_m = cluster_keys[m_i][b]  # [num_keys, d]
                V_m = cluster_values[m_i][b]  # [num_keys, d]
                
                # 计算注意力
                attn_scores = torch.matmul(Q_prime[b, i], K_m.T) / (d ** 0.5)
                attn_weights = F.softmax(attn_scores, dim=-1)
                O[b, i] = torch.matmul(attn_weights, V_m)
        
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
        
        return O+Q

    def get_stats_summary(self):
        """获取统计摘要"""
        return self.stats.get_summary()
    
    def print_stats(self):
        """打印统计信息"""
        self.stats.print_summary()
    
    def reset_stats(self):
        """重置统计"""
        self.stats.reset_stats()