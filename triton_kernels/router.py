import torch
import triton
import triton.language as tl

# =====================================================================
# Triton Fused Gather & Expert Projection Kernel
# =====================================================================
@triton.jit
def fused_gather_expert_kernel(
    X, Experts, Out,
    sorted_indices, cluster_ids,
    stride_xb, stride_xn, stride_xd,
    stride_em, stride_ed_in, stride_ed_out,
    stride_ob, stride_on, stride_od,
    stride_idx_b, stride_idx_n,
    stride_cid_b, stride_cid_n,
    BLOCK_DIN: tl.constexpr, BLOCK_DOUT: tl.constexpr
):
    b_idx = tl.program_id(0)
    out_token_idx = tl.program_id(1)  # 排序后的目标 Token 位置
    out_d_idx = tl.program_id(2)      # 负责的输出维度 Block

    # 1. 查表获取原始 Token 的位置和对应的专家 ID
    src_token_idx = tl.load(sorted_indices + b_idx * stride_idx_b + out_token_idx * stride_idx_n)
    c_id = tl.load(cluster_ids + b_idx * stride_cid_b + src_token_idx * stride_cid_n)

    # 2. 计算内存偏移量
    offs_din = tl.arange(0, BLOCK_DIN)
    offs_dout = out_d_idx * BLOCK_DOUT + tl.arange(0, BLOCK_DOUT)

    # 加载 Token 特征 (转换为 2D 以便 tl.dot 计算)
    x_ptrs = X + b_idx * stride_xb + src_token_idx * stride_xn + offs_din * stride_xd

    # 加载对应的专家权重块
    w_ptrs = Experts + c_id * stride_em + offs_din[:, None] * stride_ed_in + offs_dout[None, :] * stride_ed_out

    # ---------------- 替换后的新代码 ----------------
    # 加载 Token 特征 (保持一维)
    x = tl.load(x_ptrs)           # 形状: [BLOCK_DIN]

    # 加载对应的专家权重块
    w = tl.load(w_ptrs)           # 形状: [BLOCK_DIN, BLOCK_DOUT]

    # 3. 广播乘法 + Reduce Sum 代替 tl.dot
    # x[:, None] 将维度扩充为 [BLOCK_DIN, 1]，使其能够与 [BLOCK_DIN, BLOCK_DOUT] 的 w 逐元素相乘
    # 然后沿 axis=0 (即 BLOCK_DIN 维度) 求和，得到 [BLOCK_DOUT] 的输出
    out = tl.sum(x[:, None] * w, axis=0)

    # 4. 直接写入到按簇排布的连续内存中
    out_ptrs = Out + b_idx * stride_ob + out_token_idx * stride_on + offs_dout * stride_od
    tl.store(out_ptrs, out)
    # ------------------------------------------------

def triton_moe_router_and_project(X, Router, Experts):
    B, N, D_in = X.shape
    M, _, D_out = Experts.shape
    
    # --- 阶段 1: PyTorch 负责轻量级的路由与元数据生成 ---
    # 1. 计算路由得分并分配 Cluster
    logits = torch.matmul(X, Router.transpose(0, 1)) # [B, N, M]
    cluster_ids = torch.argmax(logits, dim=-1).to(torch.int32) # [B, N]
    
    # 2. 生成排序索引与偏移量
    sorted_indices = torch.argsort(cluster_ids, dim=-1).to(torch.int32)
    k_cu_seqlens = torch.zeros((B, M + 1), dtype=torch.int32, device=X.device)
    for b in range(B):
        k_cu_seqlens[b, 1:] = torch.cumsum(torch.bincount(cluster_ids[b], minlength=M), dim=0)

    # --- 阶段 2: Triton 负责沉重的内存收集与专家映射 ---
    out = torch.empty((B, N, D_out), dtype=X.dtype, device=X.device)
    
    # T4 优化: D_in 为 512 全量加载，D_out 且分为 16，控制 SRAM 在 32KB 左右
    BLOCK_DIN = D_in
    BLOCK_DOUT = 16 
    
    grid = (B, N, triton.cdiv(D_out, BLOCK_DOUT))
    
    fused_gather_expert_kernel[grid](
        X, Experts, out,
        sorted_indices, cluster_ids,
        X.stride(0), X.stride(1), X.stride(2),
        Experts.stride(0), Experts.stride(1), Experts.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        sorted_indices.stride(0), sorted_indices.stride(1),
        cluster_ids.stride(0), cluster_ids.stride(1),
        BLOCK_DIN=BLOCK_DIN, BLOCK_DOUT=BLOCK_DOUT,
        num_warps=4, num_stages=2
    )
    
    return out, cluster_ids, k_cu_seqlens, sorted_indices

# =====================================================================
# PyTorch 原生对照组 (显存消耗大)
# =====================================================================
def torch_moe_router_and_project(X, Router, Experts):
    B, N, D_in = X.shape
    M, _, D_out = Experts.shape
    
    logits = torch.matmul(X, Router.transpose(0, 1))
    cluster_ids = torch.argmax(logits, dim=-1).to(torch.int32)
    sorted_indices = torch.argsort(cluster_ids, dim=-1).to(torch.int32)
    
    # 内存重排 X
    batch_indices = torch.arange(B, device=X.device).unsqueeze(1).expand(B, N)
    X_sorted = X[batch_indices, sorted_indices, :]
    
    # 极度消耗显存的操作：按照每个 Token 的归属 Gather 整个权重矩阵
    sorted_cluster_ids = cluster_ids[batch_indices, sorted_indices]
    W_selected = Experts[sorted_cluster_ids] # 形状: [B, N, D_in, D_out]
    
    # 批次矩阵乘法
    out = torch.matmul(X_sorted.unsqueeze(-2), W_selected).squeeze(-2)
    
    k_cu_seqlens = torch.zeros((B, M + 1), dtype=torch.int32, device=X.device)
    for b in range(B):
         k_cu_seqlens[b, 1:] = torch.cumsum(torch.bincount(cluster_ids[b], minlength=M), dim=0)
         
    return out, cluster_ids, k_cu_seqlens, sorted_indices

# =====================================================================
# 基准测试与验证
# =====================================================================
def benchmark_routing_and_expert():
    BATCH_SIZE = 8
    N = 512
    D = 512
    M = 4 # 簇的数量
    device = torch.device('cuda')
    
    # 模拟输入、路由器和专家权重
    X = torch.randn((BATCH_SIZE, N, D), dtype=torch.float32, device=device)
    Router = torch.randn((M, D), dtype=torch.float32, device=device)
    Experts = torch.randn((M, D, D), dtype=torch.float32, device=device)
    
    # 正确性验证
    out_torch, cid_torch, cu_torch, sort_torch = torch_moe_router_and_project(X, Router, Experts)
    out_triton, cid_triton, cu_triton, sort_triton = triton_moe_router_and_project(X, Router, Experts)
    
    assert torch.equal(sort_torch, sort_triton), "排序索引不一致！"
    assert torch.allclose(out_torch, out_triton, atol=1e-3), "映射特征精度对比失败！"
    print("路由与专家映射精度验证通过！")

    # 性能测试
    quantiles = [0.5, 0.2, 0.8]
    ms_torch, min_ms_torch, max_ms_torch = triton.testing.do_bench(
        lambda: torch_moe_router_and_project(X, Router, Experts), quantiles=quantiles
    )
    print(f"PyTorch 原生 (高显存占用): {ms_torch:.4f} ms")

    ms_triton, min_ms_triton, max_ms_triton = triton.testing.do_bench(
        lambda: triton_moe_router_and_project(X, Router, Experts), quantiles=quantiles
    )
    print(f"Triton 融合重排 (零中间显存): {ms_triton:.4f} ms")

if __name__ == '__main__':
    benchmark_routing_and_expert()