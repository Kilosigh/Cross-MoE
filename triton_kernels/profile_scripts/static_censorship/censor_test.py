import torch
import triton
import triton.language as tl
import re

# ==========================================
# 1. 这是一个演示用的 Triton Kernel (包含 dot 操作)
# ==========================================
@triton.jit
def demo_dot_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 计算指针
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # 累加器初始化
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # 演示：读取数据并做点乘 (触发 Tensor Core)
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    accumulator += tl.dot(a, b)

    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator)

# ==========================================
# 2. 自动化 PTX 分析引擎
# ==========================================
def analyze_triton_kernel(kernel_func, grid, *args, **kwargs):
    print(f"\n🚀 正在触发 [{kernel_func.__name__}] 的 JIT 编译...")
    
    # 1. 组合 kernel_func 和 grid，运行一次 Kernel 以触发编译缓存
    kernel_func[grid](*args, **kwargs)

    # 2. 从原始 kernel_func 中提取 PTX
    cache_values = list(kernel_func.cache.values())
    if not cache_values:
        print("❌ 未找到编译缓存，Kernel 可能未成功运行。")
        return

    compiled_data = cache_values[0]
    
    # 动态适配不同的 Triton 缓存结构
    if hasattr(compiled_data, 'asm'):
        # 情况 1: 这是一个标准的 CompiledKernel 对象
        ptx_code = compiled_data.asm['ptx']
    elif isinstance(compiled_data, dict):
        # 情况 2: 它直接是一个字典
        if 'asm' in compiled_data:
            ptx_code = compiled_data['asm']['ptx']
        elif 'ptx' in compiled_data:
            # 情况 3: 字典本身就是存储汇编代码的 mapping
            ptx_code = compiled_data['ptx']
        else:
            print(f"❌ 字典中未找到 'ptx' 键。当前字典的键为: {compiled_data.keys()}")
            return
    else:
        print(f"❌ 未知的数据类型: {type(compiled_data)}")
        return
        
    if not ptx_code:
        print("❌ 提取到的 PTX 代码为空！")
        return
    
    print("\n📊 " + "="*10 + f" [{kernel_func.__name__}] PTX 静态审查报告 " + "="*10)
    
    # --- 检查项 A: Tensor Core (mma.sync) ---
    mma_count = len(re.findall(r'mma\.sync', ptx_code))
    if mma_count > 0:
        print(f"✅ Tensor Core 状态  : 极佳！成功调用 (找到 {mma_count} 条 mma.sync 指令)")
    else:
        print(f"❌ Tensor Core 状态  : 警报！未调用！(你的 tl.dot 退化成了慢速的 FMA 计算)")

    # --- 检查项 B: 向量化访存 (ld.global.v4) ---
    v4_load = len(re.findall(r'ld\.global.*\.v4', ptx_code))
    v4_store = len(re.findall(r'st\.global.*\.v4', ptx_code))
    v2_load = len(re.findall(r'ld\.global.*\.v2', ptx_code))
    if v4_load > 0 or v4_store > 0:
        print(f"✅ 内存带宽利用率    : 极佳！实现 v4 向量化 (Load_v4: {v4_load}, Store_v4: {v4_store})")
    elif v2_load > 0:
        print(f"⚠️ 内存带宽利用率    : 一般。部分向量化 (Load_v2: {v2_load})，可能指针步长未完全对齐。")
    else:
        print(f"❌ 内存带宽利用率    : 极差！全是单点零碎访存，速度会被严重拖慢。")

    # --- 检查项 C: 寄存器溢出 (Register Spilling) ---
    local_load = len(re.findall(r'ld\.local', ptx_code))
    local_store = len(re.findall(r'st\.local', ptx_code))
    if local_load == 0 and local_store == 0:
        print(f"✅ 寄存器压力 (SRAM): 健康！没有发生溢出 (Spilling)。")
    else:
        print(f"⚠️ 寄存器压力 (SRAM): 警告！发生溢出 (Local Load: {local_load}, Local Store: {local_store})")
        print("   👉 建议: 你的 BLOCK_M/N 设得太大了，或者循环里变量太多，尝试调小 BLOCK 尺寸。")

    print("="*60 + "\n")
    
    # 保存 PTX 以备手动查阅
    filename = f"{kernel_func.__name__}_dump.ptx"
    with open(filename, "w") as f:
        f.write(ptx_code)
    print(f"💡 完整的 PTX 代码已提取并保存至当前目录: '{filename}'")

if __name__ == "__main__":
    # 准备测试数据 (注意：fp16 更容易触发 Tensor Core)
    M, N, K = 32, 32, 32
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    c = torch.empty((M, N), device='cuda', dtype=torch.float32)

    # 定义 grid
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    # 执行自动化审查
    analyze_triton_kernel(
        demo_dot_kernel, # 传入 Kernel 和 Grid
        grid,
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=32 # 传入超参
    )