from graphviz import Digraph

def draw_architecture():
    # 创建有向图，设置整体属性
    dot = Digraph(name="System Architecture", format="pdf")
    # 设置全图字体为中文 SimHei，边和节点的属性
    dot.attr(rankdir='TB', splines='ortho', fontname='SimHei', compound='true')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='#f9f9f9', fontname='SimHei')
    
    # 组合层 cluster_top
    with dot.subgraph(name='cluster_top') as c_top:
        c_top.attr(label='表现与逻辑层', style='dashed', color='blue')
        
        # 1. 表现层
        with c_top.subgraph(name='cluster_presentation') as c:
            c.attr(label='表现层', style='dashed', color='blue', bgcolor='#e8f4f8')
            c.node('UI', 'Web可视化前端')
            c.node('Dashboard', '数据资产总览与图表渲染')
            c.node('Intervention', '外部因素冲击模拟表单')
            c.node('Vis', '多模态注意力热图呈现')
            
            # 【优化技巧】用隐形连线强制表现层按顺序从左到右排列，打好顶部地基
            with c.subgraph() as s:
                s.attr(rank='same')
                s.edge('UI', 'Dashboard', style='invis')
                s.edge('Dashboard', 'Intervention', style='invis')
                s.edge('Intervention', 'Vis', style='invis')
            
        # 2. 业务逻辑层
        with c_top.subgraph(name='cluster_logic') as c:
            c.attr(label='业务逻辑层', style='dashed', color='green', bgcolor='#eef7ee')
            c.node('API', 'API网关')
            c.node('DataCtrl', '数据对齐与预处理控制器')
            c.node('TaskQueue', '异步推理任务队列')
            
            # 【优化技巧】同样用隐形连线锁定逻辑层的左右顺序
            with c.subgraph() as s:
                s.attr(rank='same')
                s.edge('API', 'DataCtrl', style='invis')
                s.edge('DataCtrl', 'TaskQueue', style='invis')

        # 定义表现层到逻辑层的连接
        dot.edge('UI', 'API', label='HTTP / WebSocket')
        dot.edge('Dashboard', 'DataCtrl')
        dot.edge('Intervention', 'TaskQueue')
        dot.edge('Vis', 'TaskQueue')
        
        dot.edge('API', 'TaskQueue', label='任务分发')
        dot.edge('API', 'DataCtrl', label='状态查询')
    
    # 组合层 cluster_bottom
    with dot.subgraph(name='cluster_bottom') as c_bottom:
        c_bottom.attr(label='推理与数据层', style='dashed', color='purple')

        # 4. 数据访问层
        with c_bottom.subgraph(name='cluster_data') as c:
            c.attr(label='数据访问层', style='dashed', color='red', bgcolor='#fdf0f0')
            c.node('MetaDB', '关系型数据库元数据管理')
            c.node('VectorStore', '列式二进制存储多模态对齐特征')
            # 使用隐形高权重连线，确保红色框内的两个节点垂直居中排布且不跑偏
            c.edge('MetaDB', 'VectorStore', style='invis', weight='10')

        # 3. 模型推理层
        with c_bottom.subgraph(name='cluster_inference') as c:
            c.attr(label='模型推理层', style='dashed', color='purple', bgcolor='#f4eef7')
            c.node('PyTorch', 'PyTorch推理引擎')
            c.node('MoE_Module', 'MoE-Attn多模态融合模块')
            c.node('LLM_Embed', 'LLM文本特征提取器')
            c.node('Triton_Kernel', '底层Triton自定义算子')
            
            # 加强内部垂直拉力
            c.edge('PyTorch', 'MoE_Module', weight='10')
            c.edge('MoE_Module', 'LLM_Embed', label='文本特征提取')
            c.edge('MoE_Module', 'Triton_Kernel', label='JIT编译与调用')
            
            # 让底层算子水平排布
            with c.subgraph() as s:
                s.attr(rank='same')
                s.edge('LLM_Embed', 'Triton_Kernel', style='invis')

    # 定义跨层级连接关系
    # 【核心居中魔法】利用高权重 (weight='10') 把下半层像钉子一样死死对齐在控制层和任务队列的正下方！
    dot.edge('DataCtrl', 'MetaDB', label='元数据聚合与统计渲染', weight='10')
    dot.edge('TaskQueue', 'PyTorch', label='触发推理', weight='10')
    
    # 【防止变形】其余斜向交叉线，设定 constraint='false'，让它们只连线、不产生拉扯节点的引力
    dot.edge('DataCtrl', 'VectorStore', label='数据检索与对齐', weight='1', tailport='sw', headport='nw')
    dot.edge('PyTorch', 'VectorStore', label='加载时序与文本张量', constraint='false')

    # 保存并渲染
    dot.render('system_architecture_diagram_v3', view=False)
    print("架构图已生成: system_architecture_diagram_v3.pdf")

if __name__ == "__main__":
    draw_architecture()