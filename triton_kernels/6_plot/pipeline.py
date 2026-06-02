from graphviz import Digraph

def draw_pure_data_flow():
    dot = Digraph(name="Pure Data Flow", format="pdf")
    dot.attr(rankdir='TB', splines='ortho', fontname='SimHei')
    
    # 定义不同类型数据的节点样式
    # 文本模态数据 (蓝色系)
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='#e8f4f8', color='#2b8cbe', fontname='SimHei')
    dot.node('LiveText', '实时输入文本\n(模拟外部事件)')
    dot.node('LiveEmb', '实时文本特征向量\n(Shape: N_now x d)')
    dot.node('HistEmb', '历史文本特征向量\n预计算落盘 (Shape: N_his x d)')
    dot.node('TextFeatures', '完整文本特征张量\n(Shape: N x d)')

    # 时序模态数据 (橙色系)
    dot.attr('node', style='rounded,filled', fillcolor='#fdf0e6', color='#d95f02')
    dot.node('HistTS', '历史原始时间序列\n(日频/小时频)')
    dot.node('TSPatches', '时间序列Patch特征\n(Shape: L x d)')

    # 交叉融合与输出数据 (紫色/绿色系)
    dot.attr('node', style='rounded,filled', fillcolor='#f4eef7', color='#7570b3')
    dot.node('HeteroData', '异构多模态输入特征')
    
    dot.attr('node', style='rounded,filled', fillcolor='#eef7ee', color='#2ca25f')
    dot.node('PredTS', '预测未来时间序列\n(Shape: H x 1)')
    dot.node('AttnMatrix', '稠密注意力权重矩阵\n(Shape: L x N)')

    # --- 数据流转连线 (标明操作) ---
    dot.attr('edge', fontname='SimHei', fontsize='10', color='#555555')
    
    # 文本流
    dot.edge('LiveText', 'LiveEmb', label=' LLM 实时特征提取')
    dot.edge('HistEmb', 'TextFeatures', label=' 依时间窗口 (如近7天) 拼接')
    dot.edge('LiveEmb', 'TextFeatures', label=' 特征拼接 (Concat)')

    # 时序流
    dot.edge('HistTS', 'TSPatches', label=' 依时间窗口 (如近60天) 提取 & Patching分块')

    # 融合与双引擎路由
    dot.edge('TextFeatures', 'HeteroData', label=' 注入')
    dot.edge('TSPatches', 'HeteroData', label=' 注入')

    # 引擎A：Triton 常规计算
    dot.edge('HeteroData', 'PredTS', label=' [默认路由]\nTriton 算子计算 (计算与显存访问融合)')
    
    # 引擎B：PyTorch 可视化计算
    dot.edge('HeteroData', 'PredTS', label=' [可视化路由]\nPyTorch 原生计算', style='dashed', color='#7570b3')
    dot.edge('HeteroData', 'AttnMatrix', label=' [可视化路由]\n反向提取与保留中间激活值', style='dashed', color='#7570b3')

    # 渲染生成
    dot.render('pure_data_flow_diagram', view=False)
    print("纯数据视角流转图已生成: pure_data_flow_diagram.pdf")

if __name__ == "__main__":
    draw_pure_data_flow()