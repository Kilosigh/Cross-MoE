import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import os

class TSNEVisualizer:
    """
    高级 t-SNE 可视化工具，支持中心点更新后的标签重新分配
    
    功能特点:
    1. 自动处理更新后的聚类中心
    2. 动态调整 t-SNE 参数解决重叠问题
    3. 智能采样处理大数据集
    4. 中心点标记优化防止重叠
    5. 丰富的自定义选项
    
    使用方法:
    visualizer = TSNEVisualizer(output_dir="results/", M=10)
    visualizer.generate_tsne_plot(
        data=your_data, 
        centers=updated_centers, 
        labels=None,  # 将自动重新计算
        min_distance=0.1,
        title="Updated Clusters"
    )
    """
    
    def __init__(self, output_dir="visualizations/", M=10, random_state=42):
        """
        初始化 t-SNE 可视化器
        
        参数:
        output_dir: 输出目录路径
        M: 聚类中心数量
        random_state: 随机种子
        """
        self.output_dir = output_dir
        self.M = M
        self.random_state = random_state
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
    @staticmethod
    def assign_labels(data, centers):
        """
        根据中心点重新分配标签（考虑更新后的中心位置）
        
        参数:
        data: (N, d) 数据点矩阵
        centers: (M, d) 中心点矩阵
        
        返回:
        labels: (N,) 重新分配后的标签数组
        """
        # 计算每个数据点到所有中心的距离
        expanded_data = data[:, np.newaxis, :]
        expanded_centers = centers[np.newaxis, :, :]
        distances = np.linalg.norm(expanded_data - expanded_centers, axis=2)
        
        # 为每个点分配最近中心的索引
        return np.argmin(distances, axis=1)
    
    def generate_tsne_plot(self, data, centers, labels=None, min_distance=0.1, 
                          title="Cluster Visualization", filename="tsne_plot.png",
                          sample_size=5000, perplexity=30, learning_rate=200,
                          early_exaggeration=12.0, n_iter=1000, figsize=(16, 14),
                          dpi=300, show_annotations=True):
        """
        生成改进版 t-SNE 可视化图（支持中心点更新）
        
        参数:
        data: (N, d) 原始数据点
        centers: (M, d) 更新后的中心点
        labels: (N,) 原始标签 (可选，未提供则重新计算)
        min_distance: 最小中心距离阈值
        title: 图表标题
        filename: 输出文件名
        sample_size: 最大采样点数
        perplexity: t-SNE 困惑度参数
        learning_rate: t-SNE 学习率
        early_exaggeration: t-SNE 早期放大因子
        n_iter: t-SNE 迭代次数
        figsize: 图表尺寸
        dpi: 输出图像分辨率
        show_annotations: 是否显示中心点距离标注
        """
        print(f"生成改进版t-SNE可视化: {title}...")
        
        # 1. 重新计算标签（如果需要）
        if labels is None:
            labels = self.assign_labels(data, centers)
            print("已根据更新后的中心重新计算标签")
        
        # 2. 数据采样
        if len(data) > sample_size:
            sample_indices = np.random.choice(len(data), sample_size, replace=False)
            sampled_data = data[sample_indices]
            sampled_labels = labels[sample_indices]
        else:
            sampled_data = data
            sampled_labels = labels
        
        # 3. 合并数据（数据点 + 中心点）
        all_points = np.vstack([sampled_data, centers])
        
        # 4. 动态调整 t-SNE 参数
        eff_perplexity = min(perplexity, max(5, len(sampled_data) // 100))
        print(f"使用参数: 困惑度={eff_perplexity}, 学习率={learning_rate}, 迭代次数={n_iter}")
        
        # 5. 执行 t-SNE 降维
        tsne = TSNE(
            n_components=2,
            random_state=self.random_state,
            perplexity=eff_perplexity,
            learning_rate=learning_rate,
            early_exaggeration=early_exaggeration,
            n_iter=n_iter
        )
        tsne_results = tsne.fit_transform(all_points)
        
        # 6. 分离结果
        data_tsne = tsne_results[:len(sampled_data)]
        centers_tsne = tsne_results[len(sampled_data):]
        
        # 7. 创建图表
        plt.figure(figsize=figsize)
        ax = plt.gca()
        
        # 8. 创建颜色映射
        if self.M <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, self.M))
        else:
            colors = plt.cm.gist_ncar(np.linspace(0, 1, self.M))
        cmap = ListedColormap(colors)
        
        # 9. 绘制数据点
        scatter = plt.scatter(
            data_tsne[:, 0], 
            data_tsne[:, 1], 
            c=sampled_labels, 
            cmap=cmap,
            alpha=0.6,
            s=25,
            label="数据点"
        )
        
        # 10. 绘制中心点（带智能标记）
        center_markers = []  # 用于图例
        for i, center in enumerate(centers_tsne):
            # 计算到其他中心的最小距离
            other_centers = np.delete(centers_tsne, i, axis=0)
            if len(other_centers) > 0:
                distances = np.linalg.norm(other_centers - center, axis=1)
                min_dist = np.min(distances) if len(distances) > 0 else 0
            else:
                min_dist = 0
            
            # 根据距离确定标记样式
            if min_dist < min_distance:
                marker = 'D'  # 菱形
                size = 300
                edge_color = 'red'
            else:
                marker = 'X' if min_dist < min_distance * 2 else '*'
                size = 250
                edge_color = 'black'
            
            # 绘制中心点
            plt.scatter(
                center[0], 
                center[1], 
                color=colors[i % len(colors)],
                marker=marker,
                s=size,
                edgecolors=edge_color,
                linewidths=1.5,
                zorder=10  # 确保中心点在顶层
            )
            
            # 添加中心点编号
            plt.text(
                center[0], 
                center[1], 
                str(i),
                fontsize=12,
                fontweight='bold',
                ha='center',
                va='center',
                color='white' if np.mean(colors[i]) < 0.5 else 'black'
            )
            
            # 添加距离标注
            if show_annotations and min_dist < min_distance * 3:
                plt.annotate(f"d={min_dist:.2f}", 
                            (center[0], center[1]),
                            xytext=(10, 10), 
                            textcoords='offset points',
                            fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.2", 
                                    fc='yellow', 
                                    alpha=0.7))
            
            # 保存标记样式用于图例
            center_markers.append(marker)
        
        # 11. 添加图表信息
        plt.title(f"{title}\n(簇数: {self.M}, 最小中心距离: {min(min_distance, min_dist):.4f})")
        plt.xlabel("t-SNE维度 1")
        plt.ylabel("t-SNE维度 2")
        
        # 12. 添加颜色条
        cbar = plt.colorbar(scatter, ticks=range(self.M))
        cbar.set_label('簇编号')
        
        # 13. 添加图例
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                  markersize=8, label='数据点'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
                  markersize=12, label='中心点 (正常)'),
            Line2D([0], [0], marker='X', color='w', markerfacecolor='red', 
                  markersize=12, label='中心点 (较近)'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='red', 
                  markersize=12, label='中心点 (过近)')
        ]
        plt.legend(handles=legend_elements, loc='best')
        
        # 14. 网格和样式
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        
        # 15. 保存图表
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"t-SNE 可视化已保存至: {output_path}")
        plt.close()
        
        return output_path