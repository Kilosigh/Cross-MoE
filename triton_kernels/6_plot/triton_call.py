import graphviz

def draw_triton_call_stack():
    # 创建一个有向图
    dot = graphviz.Digraph(comment='Triton Kernel Call Stack', format='png')
    
    dot.attr(dpi='300')
    dot.attr(rankdir='TB', size='12,16')
    
    font_name = 'Microsoft YaHei'
    dot.attr('node', shape='box', style='rounded,filled', fontname=font_name)
    dot.attr('edge', fontname=font_name, fontsize='10')

    # 定义配色方案
    COLOR_PYTORCH = '#E4F0F8'
    COLOR_WRAPPER = '#FDF4EF'
    COLOR_KERNEL = '#ECFCCB'

    # 【核心修改点】：创建一个辅助函数，将带换行的文本转换为 HTML 表格标签
    # 通过调节 spacing 的数值（比如 4, 6, 8）就可以自由控制节点内的文字行距
    def add_line_spacing(text, spacing=6):
        if '\n' not in text:
            return text
        lines = text.split('\n')
        # 构建 HTML 结构，CELLSPACING 即为我们想要的“行距”
        rows = "".join([f'<TR><TD>{line}</TD></TR>' for line in lines])
        return f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="{spacing}">{rows}</TABLE>>'

    # ==========================
    # 1. PyTorch 层 (Top Level)
    # ==========================
    with dot.subgraph(name='cluster_pytorch') as c:
        c.attr(label='PyTorch 模块与 Autograd 引擎', style='dashed', color='blue', fontname=font_name)
        
        # 使用辅助函数，这里设一个较大的行距 spacing=8
        c.node('nn_module', add_line_spacing('FinalCrossMoEMultiHeadAttention.forward()\n(nn.Module)', spacing=8), fillcolor=COLOR_PYTORCH)
        c.node('autograd_fwd', 'FinalCrossMoEMultiHeadAttentionFunc.forward()', fillcolor=COLOR_PYTORCH)
        c.node('autograd_bwd', 'FinalCrossMoEMultiHeadAttentionFunc.backward()', fillcolor=COLOR_PYTORCH)
        
        c.edge('nn_module', 'autograd_fwd', label=' 前向传播 (Forward)')
        c.edge('nn_module', 'autograd_bwd', label=' 反向传播 (Backward)', style='dashed')

    # ==========================
    # 2. 前向传播调用链路
    # ==========================
    with dot.subgraph(name='cluster_forward') as c:
        c.attr(label='前向计算链路 (Forward Operations)', style='dotted', color='black', fontname=font_name)
        
        # 对带换行的节点统一应用行距函数
        c.node('fwd_wrap_moe', add_line_spacing('triton_moe_project_mha()\n(分别处理 Q 和 KV 的专家投影)', spacing=6), fillcolor=COLOR_WRAPPER)
        c.node('fwd_wrap_attn', 'batched_flash_decoding_mha()', fillcolor=COLOR_WRAPPER)
        
        c.node('fwd_kernel_moe', add_line_spacing('@triton.jit\nbatched_expert_project_kernel', spacing=4), fillcolor=COLOR_KERNEL)
        c.node('fwd_kernel_attn_p1', add_line_spacing('@triton.jit\nbatched_flash_decoding_mha_phase1', spacing=4), fillcolor=COLOR_KERNEL)
        c.node('fwd_kernel_attn_p2', add_line_spacing('@triton.jit\nbatched_flash_decoding_mha_phase2', spacing=4), fillcolor=COLOR_KERNEL)

        c.edge('autograd_fwd', 'fwd_wrap_moe', label=' 1. 路由与专家投影')
        c.edge('autograd_fwd', 'fwd_wrap_attn', label=' 2. Flash Decoding 注意力计算')
        
        c.edge('fwd_wrap_moe', 'fwd_kernel_moe', label=' 线程网格: (M, B*H, max_blocks)')
        c.edge('fwd_wrap_attn', 'fwd_kernel_attn_p1', label=' 线程网格: (max_q_blocks, num_splits, M*B*H)')
        c.edge('fwd_wrap_attn', 'fwd_kernel_attn_p2', label=' 线程网格: (max_q_blocks, M*B*H)')

    # ==========================
    # 3. 反向传播调用链路
    # ==========================
    with dot.subgraph(name='cluster_backward') as c:
        c.attr(label='反向计算链路 (Backward Operations)', style='dotted', color='black', fontname=font_name)
        
        c.node('bwd_wrap_attn', 'triton_attention_backward()', fillcolor=COLOR_WRAPPER)
        c.node('bwd_wrap_moe', add_line_spacing('triton_moe_project_backward()\n(分别计算 dQ 和 dKV 的梯度)', spacing=6), fillcolor=COLOR_WRAPPER)
        
        c.node('bwd_kernel_attn_dq', add_line_spacing('@triton.jit\nbwd_kernel_dq_mha', spacing=4), fillcolor=COLOR_KERNEL)
        c.node('bwd_kernel_attn_dkdv', add_line_spacing('@triton.jit\nbwd_kernel_dk_dv_mha', spacing=4), fillcolor=COLOR_KERNEL)
        c.node('bwd_kernel_moe', add_line_spacing('@triton.jit\nbwd_expert_project_kernel', spacing=4), fillcolor=COLOR_KERNEL)

        c.edge('autograd_bwd', 'bwd_wrap_attn', label=' 1. 注意力反向传播 (Attention Grad)')
        c.edge('autograd_bwd', 'bwd_wrap_moe', label=' 2. 专家层反向传播 (MoE Grad)')
        
        c.edge('bwd_wrap_attn', 'bwd_kernel_attn_dq', label=' 线程网格: (M, B*H, q_blocks)')
        c.edge('bwd_wrap_attn', 'bwd_kernel_attn_dkdv', label=' 线程网格: (M, B*H, k_blocks)')
        c.edge('bwd_wrap_moe', 'bwd_kernel_moe', label=' 线程网格: (M, B*H)')

    # 保存并渲染
    output_path = 'triton_cross_moe_call_stack_line_spacing'
    dot.render(output_path, view=False)
    print(f"节点内行距调整完毕，图示已保存至: {output_path}.png")

if __name__ == '__main__':
    draw_triton_call_stack()