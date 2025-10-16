import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional, Tuple, Union, List
import seaborn as sns


class AttentionHeatmapVisualizer:
    """注意力热图可视化器"""
    
    def __init__(self, configs):
        """初始化可视化器"""
        # 预定义一些颜色方案
        self.color_schemes = {
            'default': 'viridis',
            'warm': 'YlOrRd',
            'cool': 'Blues',
            'diverging': 'RdBu_r',
            'purple': 'Purples',
            'green': 'Greens',
            'seismic': 'seismic',
            'plasma': 'plasma',
            'inferno': 'inferno',
            'magma': 'magma'
        }
        self.configs = configs
        # 设置最大图像尺寸限制（英寸），避免超出matplotlib限制
        self.max_figure_size = 50  # 最大50英寸，在100 DPI下约5000像素
    
    def _calculate_safe_figsize(self, aspect_ratio: float, figsize_base: float) -> Tuple[float, float]:
        """
        计算安全的图像尺寸，避免超出matplotlib限制
        
        参数:
        ----------
        aspect_ratio : float
            图像宽高比 (width/height)
        figsize_base : float
            基础图像大小
            
        返回:
        -------
        Tuple[float, float]
            安全的图像尺寸 (width, height)
        """
        if aspect_ratio >= 1:
            width = figsize_base * aspect_ratio
            height = figsize_base
        else:
            width = figsize_base
            height = figsize_base / aspect_ratio
        
        # 限制最大尺寸
        if width > self.max_figure_size:
            scale_factor = self.max_figure_size / width
            width = self.max_figure_size
            height = height * scale_factor
        
        if height > self.max_figure_size:
            scale_factor = self.max_figure_size / height
            height = self.max_figure_size
            width = width * scale_factor
        
        return (width, height)
    
    def plot_attention(
        self,
        attention_weights: np.ndarray,
        title: str = "Attention Heatmap",
        colormap: Union[str, mcolors.Colormap] = 'default',
        figsize_base: float = 8.0,
        aspect_ratio: Optional[float] = None,
        save_path: Optional[str] = None,
        dpi: int = 300,
        show_values: bool = False,
        value_threshold: float = 0.5,
        cbar_label: str = "Attention Weight",
        x_labels: Optional[List[str]] = None,
        y_labels: Optional[List[str]] = None,
        x_label: str = "Keys (K)",
        y_label: str = "Queries (Q)",
        font_size: int = 10,
        title_size: int = 14,
        label_rotation: Tuple[int, int] = (45, 0),
        grid: bool = False,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None
    ) -> plt.Figure:
        """
        绘制注意力热图
        
        参数:
        ----------
        attention_weights : np.ndarray
            注意力权重矩阵，shape为(q_len, k_len)
        title : str
            图表标题
        colormap : str or Colormap
            颜色映射，可以是预定义的名称或matplotlib colormap
        figsize_base : float
            基础图像大小
        aspect_ratio : float, optional
            图像宽高比，如果为None则自动计算
        save_path : str, optional
            保存路径，如果提供则保存图像
        dpi : int
            保存图像的DPI
        show_values : bool
            是否在热图上显示数值
        value_threshold : float
            显示数值时的阈值，用于调整文字颜色
        cbar_label : str
            颜色条标签
        x_labels : List[str], optional
            x轴标签列表
        y_labels : List[str], optional
            y轴标签列表
        x_label : str
            x轴名称
        y_label : str
            y轴名称
        font_size : int
            字体大小
        title_size : int
            标题字体大小
        label_rotation : Tuple[int, int]
            x和y轴标签旋转角度
        grid : bool
            是否显示网格
        vmin : float, optional
            颜色映射最小值
        vmax : float, optional
            颜色映射最大值
        
        返回:
        -------
        fig : matplotlib.figure.Figure
            生成的图像对象
        """
        # 获取矩阵维度
        q_len, k_len = attention_weights.shape
        
        # 自动计算图像比例
        if aspect_ratio is None:
            aspect_ratio = k_len / q_len
        
        # 计算安全的图像大小
        figsize = self._calculate_safe_figsize(aspect_ratio, figsize_base)
        
        # 处理颜色映射
        if isinstance(colormap, str):
            if colormap in self.color_schemes:
                colormap = self.color_schemes[colormap]
        
        # 创建图像，降低DPI避免内存问题
        display_dpi = min(100, dpi)  # 显示时使用较低DPI
        fig, ax = plt.subplots(figsize=figsize, dpi=display_dpi)
        
        # 绘制热图
        im = ax.imshow(
            attention_weights,
            cmap=colormap,
            aspect='auto',
            interpolation='nearest',
            vmin=vmin,
            vmax=vmax
        )
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size-2)
        
        # 设置标题
        ax.set_title(title, fontsize=title_size, pad=20)
        
        # 设置轴标签
        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel(y_label, fontsize=font_size)
        
        # 设置刻度
        if x_labels is not None:
            ax.set_xticks(np.arange(len(x_labels)))
            ax.set_xticklabels(x_labels, fontsize=font_size-2, rotation=label_rotation[0], ha='right')
        else:
            ax.set_xticks(np.arange(0, k_len, max(1, k_len//10)))
            ax.set_xticklabels(np.arange(0, k_len, max(1, k_len//10)), fontsize=font_size-2, rotation=label_rotation[0])
        
        if y_labels is not None:
            ax.set_yticks(np.arange(len(y_labels)))
            ax.set_yticklabels(y_labels, fontsize=font_size-2, rotation=label_rotation[1])
        else:
            ax.set_yticks(np.arange(0, q_len, max(1, q_len//10)))
            ax.set_yticklabels(np.arange(0, q_len, max(1, q_len//10)), fontsize=font_size-2, rotation=label_rotation[1])
        
        # 显示数值（可选）- 只在小矩阵时显示
        if show_values and q_len <= 20 and k_len <= 20:
            for i in range(q_len):
                for j in range(k_len):
                    value = attention_weights[i, j]
                    text_color = 'white' if value > value_threshold else 'black'
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                           color=text_color, fontsize=font_size-4)
        
        # 添加网格（可选）
        if grid:
            ax.set_xticks(np.arange(k_len + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(q_len + 1) - 0.5, minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # 调整布局，添加异常处理
        try:
            plt.tight_layout()
        except ValueError as e:
            print(f"警告: tight_layout失败 ({e})，跳过布局调整")
            # 手动调整边距
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
        
        # 保存图像（可选）
        if save_path:
            try:
                fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
                print(f"图像已保存至: {save_path}")
            except Exception as e:
                print(f"保存图像失败: {e}")
                # 尝试以较低分辨率保存
                try:
                    fig.savefig(save_path, dpi=min(150, dpi), bbox_inches='tight')
                    print(f"图像已以较低分辨率保存至: {save_path}")
                except Exception as e2:
                    print(f"低分辨率保存也失败: {e2}")
        
        return fig
    
    def plot_multi_head_attention(
        self,
        attention_weights: np.ndarray,
        head_names: Optional[List[str]] = None,
        title: str = "Multi-Head Attention",
        colormap: Union[str, mcolors.Colormap] = 'default',
        save_path: Optional[str] = None,
        dpi: int = 300,
        **kwargs
    ) -> plt.Figure:
        """
        绘制多头注意力热图
        
        参数:
        ----------
        attention_weights : np.ndarray
            多头注意力权重，shape为(num_heads, q_len, k_len)
        head_names : List[str], optional
            每个注意力头的名称
        其他参数同plot_attention
        """
        
        num_heads, q_len, k_len = attention_weights.shape
        
        # 计算子图布局
        cols = min(4, num_heads)
        rows = (num_heads + cols - 1) // cols
        
        # 计算安全的图像大小
        aspect_ratio = k_len / q_len
        base_size = 4
        subplot_width, subplot_height = self._calculate_safe_figsize(aspect_ratio, base_size)
        
        # 限制总体图像大小
        fig_width = min(subplot_width * cols, self.max_figure_size)
        fig_height = min(subplot_height * rows, self.max_figure_size)
        
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), dpi=100)
        axes = axes.flatten() if num_heads > 1 else [axes]
        
        # 处理颜色映射
        if isinstance(colormap, str):
            if colormap in self.color_schemes:
                colormap = self.color_schemes[colormap]
        
        # 绘制每个注意力头
        for i in range(num_heads):
            ax = axes[i]
            im = ax.imshow(
                attention_weights[i],
                cmap=colormap,
                aspect='auto',
                interpolation='nearest'
            )
            
            # 设置子标题
            if head_names:
                ax.set_title(head_names[i], fontsize=10)
            else:
                ax.set_title(f'Head {i+1}', fontsize=10)
            
            # 简化刻度标签
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 添加颜色条
            try:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            except Exception as e:
                print(f"警告: 无法为第{i+1}个头添加颜色条: {e}")
        
        # 隐藏多余的子图
        for i in range(num_heads, len(axes)):
            axes[i].set_visible(False)
        
        # 设置总标题
        fig.suptitle(title, fontsize=14, y=1.02)
        
        # 调整布局，添加异常处理
        try:
            plt.tight_layout()
        except ValueError as e:
            print(f"警告: 多头注意力图tight_layout失败 ({e})，跳过布局调整")
            plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9)
        
        # 保存图像
        if save_path:
            try:
                fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
                print(f"多头注意力图像已保存至: {save_path}")
            except Exception as e:
                print(f"多头注意力图像保存失败: {e}")
                try:
                    fig.savefig(save_path, dpi=min(150, dpi), bbox_inches='tight')
                    print(f"多头注意力图像已以较低分辨率保存至: {save_path}")
                except Exception as e2:
                    print(f"低分辨率保存也失败: {e2}")
        
        return fig


# 使用示例
def example_usage():
    """示例：如何使用修复后的AttentionHeatmapVisualizer"""
    
    # 创建可视化器实例
    visualizer = AttentionHeatmapVisualizer(None)
    
    # 示例：超大尺寸注意力矩阵测试
    print("测试超大尺寸注意力矩阵...")
    q_len, k_len = 100, 5000  # 极端的宽高比
    attention_large = np.random.rand(q_len, k_len)
    attention_large = attention_large / attention_large.sum(axis=1, keepdims=True)
    
    # 现在应该不会报错了
    fig = visualizer.plot_attention(
        attention_large,
        title=f"Large Attention Matrix ({q_len}x{k_len})",
        colormap='plasma',
        figsize_base=10,
        save_path='attention_large_safe.png'
    )
    plt.show()
    
    print("大尺寸矩阵测试完成！")


if __name__ == "__main__":
    example_usage()