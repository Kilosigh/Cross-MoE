from graphviz import Digraph

def generate_pytorch_vs_triton_diagram():
    # 创建有向图对象
    dot = Digraph(comment='PyTorch vs Triton Call Stack', format='png')
    dot.attr(rankdir='TB', size='12,8', dpi='300', fontname='SimHei')

    # ========== PyTorch 调用栈 (左侧) ==========
    with dot.subgraph(name='cluster_pytorch') as pt:
        pt.attr(label='原生 PyTorch (Eager Mode)\n"多次 Kernel 启动与显存墙瓶颈"', 
                style='dashed', color='blue', fontname='SimHei', fontsize='16')
        
        pt.node('PT_User', 'Python 用户代码\n例如: d = a + b; e = d * c', 
                shape='box', style='filled', fillcolor='#D0E4F5')
        pt.node('PT_API', 'PyTorch Python API', shape='box')
        pt.node('PT_CPP', 'C++ ATen / Dispatcher (算子分发)', shape='box')
        
        # 第一次操作：加法
        pt.node('PT_K1', 'CUDA Launch: Add Kernel', 
                shape='ellipse', style='filled', fillcolor='#E2E2E2')
        pt.node('PT_Mem1', 'GPU VRAM (全局显存):\n读 a, b -> 写 d', 
                shape='cylinder', style='filled', fillcolor='#FFDDC1')
        
        # 第二次操作：乘法
        pt.node('PT_K2', 'CUDA Launch: Mul Kernel', 
                shape='ellipse', style='filled', fillcolor='#E2E2E2')
        pt.node('PT_Mem2', 'GPU VRAM (全局显存):\n读 d, c -> 写 e', 
                shape='cylinder', style='filled', fillcolor='#FFDDC1')

        # 构建边
        pt.edge('PT_User', 'PT_API')
        pt.edge('PT_API', 'PT_CPP')
        pt.edge('PT_CPP', 'PT_K1')
        pt.edge('PT_K1', 'PT_Mem1')
        pt.edge('PT_Mem1', 'PT_K2', label=' 返回控制权 / 显存读取开销', color='red', fontcolor='red')
        pt.edge('PT_K2', 'PT_Mem2')

    # ========== Triton 调用栈 (右侧) ==========
    with dot.subgraph(name='cluster_triton') as tr:
        tr.attr(label='OpenAI Triton\n"编译器 JIT 融合与 SRAM 优化"', 
                style='dashed', color='green', fontname='SimHei', fontsize='16')
        
        tr.node('TR_User', 'Python 用户代码\n(@triton.jit 装饰的融合算子)', 
                shape='box', style='filled', fillcolor='#D0F5D6')
        tr.node('TR_AST', 'Triton 编译器 (Python AST解析)', shape='box')
        tr.node('TR_IR', 'Triton IR / 优化 Pass\n(Triton-IR -> LLVM-IR -> PTX)', shape='box')
        tr.node('TR_Driver', 'CUDA Driver API (加载 PTX 代码)', shape='box')
        
        # 融合计算
        tr.node('TR_K1', 'CUDA Launch: 单个 Fused Kernel', 
                shape='ellipse', style='filled', fillcolor='#E2E2E2')
        tr.node('TR_SRAM', 'GPU SRAM (片上共享内存):\n一次性加载 a, b, c\n直接在 SRAM 中完成 Add 和 Mul 计算', 
                shape='box3d', style='filled', fillcolor='#FFEB3B')
        tr.node('TR_VRAM', 'GPU VRAM (全局显存):\n仅将最终结果 e 写回', 
                shape='cylinder', style='filled', fillcolor='#FFDDC1')

        # 构建边
        tr.edge('TR_User', 'TR_AST')
        tr.edge('TR_AST', 'TR_IR')
        tr.edge('TR_IR', 'TR_Driver')
        tr.edge('TR_Driver', 'TR_K1')
        tr.edge('TR_K1', 'TR_SRAM', label=' 避免中间变量落盘', color='blue')
        tr.edge('TR_SRAM', 'TR_VRAM')

    # 生成并保存图片
    output_filename = 'pytorch_vs_triton_architecture'
    dot.render(output_filename, cleanup=True)
    print(f"图表已成功生成并保存为 {output_filename}.png")

if __name__ == "__main__":
    generate_pytorch_vs_triton_diagram()