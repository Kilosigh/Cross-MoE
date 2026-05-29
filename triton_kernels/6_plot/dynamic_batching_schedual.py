from graphviz import Digraph

def generate_u_shape_flowchart():
    # 核心配置
    dot = Digraph(comment='Dynamic Batching Flowchart - U Shape', format='png')
    dot.attr(rankdir='TB', fontname='WenQuanYi Micro Hei', nodesep='0.8', ranksep='0.6') 
    
    # 全局样式
    dot.attr('node', shape='rectangle', style='rounded,filled', fillcolor='white', fontname='WenQuanYi Micro Hei', fontsize='12')
    dot.attr('edge', fontname='WenQuanYi Micro Hei', fontsize='10')

    # === 左半边：接收与攒批阶段 (自上而下) ===
    with dot.subgraph(name='cluster_left') as c:
        c.attr(label='API 与攒批阶段 (自上而下)', style='dashed', bgcolor='#f0f8ff', fontname='WenQuanYi Micro Hei')
        c.node('API', 'FastAPI 路由\n(/predict/triton_dynamic)', fillcolor='lightblue')
        c.node('Queue', '线程安全队列\n(Request Queue)', shape='folder', fillcolor='lightgrey')
        c.node('WaitFirst', '阻塞等待第一个请求')
        c.node('InitBatch', '初始化 Batch 并记录 Start Time')
        c.node('CheckCondition', '条件判断:\nBatch Size < Max ?\n且未超时 ?', shape='diamond', fillcolor='lightyellow')
        c.node('WaitNext', '限时等待后续请求\n(Timeout = 剩余时间)')
        c.node('AddBatch', '加入 Batch')

    # === 右半边：执行与返回阶段 (自下而上) ===
    with dot.subgraph(name='cluster_right') as c:
        c.attr(label='执行与唤醒阶段 (自下而上)', style='dashed', bgcolor='#f5fffa', fontname='WenQuanYi Micro Hei')
        c.node('DataTransfer', 'Host to Device 异步拷贝\n(non_blocking=True)')
        c.node('Padding', '无效槽位补零\n(Zero Padding)')
        c.node('CUDAGraph', 'CUDA Graph Replay\n(执行推理)', fillcolor='salmon')
        c.node('Sync', 'CUDA 同步\n(torch.cuda.synchronize)')
        c.node('Callback', '遍历 Batch\n(提取结果)')
        c.node('Future', 'Asyncio Future\n(挂起等待)', fillcolor='lightyellow')

    # === 外部游离节点 ===
    dot.node('Client', '客户端并发请求', shape='cylinder', fillcolor='lightgrey')
    dot.node('Response', '返回推理结果', fillcolor='lightgrey')

    # === 核心技巧 1：强制水平对齐 (打造完美的 U 型结构) ===
    # 这一步把左右两边的节点像梯子横档一样绑定，确保两边对称
    dot.body.append('{rank=same; "API"; "Future"}')
    dot.body.append('{rank=same; "Queue"; "Callback"}')
    dot.body.append('{rank=same; "WaitFirst"; "Sync"}')
    dot.body.append('{rank=same; "InitBatch"; "CUDAGraph"}')
    dot.body.append('{rank=same; "CheckCondition"; "Padding"}')
    dot.body.append('{rank=same; "WaitNext"; "DataTransfer"}')

    # === 连线逻辑 ===

    # 1. 外部进出
    dot.edge('Client', 'API', label=' 发起请求')
    dot.edge('Future', 'Response', label=' 结束')

    # 2. 左侧主线 (自上而下)
    dot.edge('API', 'Queue', label=' 放入请求')
    dot.edge('Queue', 'WaitFirst', label=' 消费者提取')
    dot.edge('WaitFirst', 'InitBatch')
    dot.edge('InitBatch', 'CheckCondition')
    dot.edge('CheckCondition', 'WaitNext', label=' Yes')
    dot.edge('WaitNext', 'AddBatch', label=' 获取到请求')
    dot.edge('AddBatch', 'CheckCondition')

    # 3. 谷底桥连 (从左边底部跨越到右边底部)
    dot.edge('WaitNext', 'DataTransfer', label=' 超时为空')
    # constraint='false' 避免这条跨越线破坏 U 型的底部结构
    dot.edge('CheckCondition', 'DataTransfer', label=' No (满批或超时)', constraint='false')

    # 4. 右侧主线 (自下而上)
    # 核心技巧 2：dir='back' 欺骗排版引擎，实际画出的箭头是朝上的
    dot.edge('Padding', 'DataTransfer', dir='back', label=' 填充')
    dot.edge('CUDAGraph', 'Padding', dir='back')
    dot.edge('Sync', 'CUDAGraph', dir='back')
    dot.edge('Callback', 'Sync', dir='back')
    dot.edge('Future', 'Callback', dir='back', label=' Threadsafe Set Result\n(唤醒 Future)', color='blue')

    # 5. 顶层横向等待线
    # constraint='false' 确保它只是一条虚线补充，不会把左右两边硬拉在一起
    dot.edge('API', 'Future', label=' Await', style='dotted', constraint='false')

    # === 渲染输出 ===
    output_filename = 'dynamic_batching_flow_u_shape'
    dot.render(output_filename, view=False, cleanup=True) 
    print(f"✅ U型流水线架构图已生成：{output_filename}.png")

if __name__ == '__main__':
    generate_u_shape_flowchart()