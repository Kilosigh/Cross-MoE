import torch
import triton
import triton.language as tl

# =====================================================================
# Phase 1: Batched Local Attention (Split-K)
# =====================================================================
@triton.jit
def batched_flash_decoding_phase1_kernel(
    Q, K, V,
    mid_acc, mid_m, mid_l,
    q_cluster_ids, k_cu_seqlens,
    sm_scale,
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_mid_acc_b, stride_mid_acc_q, stride_mid_acc_s, stride_mid_acc_d,
    stride_mid_m_b, stride_mid_m_q, stride_mid_m_s,
    stride_mid_l_b, stride_mid_l_q, stride_mid_l_s,
    stride_q_cid_b, stride_q_cid_n,
    stride_k_cu_b, stride_k_cu_m,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    q_idx = tl.program_id(0)
    split_idx = tl.program_id(1)
    b_idx = tl.program_id(2)  # 新增：获取当前所处的 Batch 索引
    num_splits = tl.num_programs(1)

    # 1. 获取当前 Batch 下，该 Query 的路由信息
    cluster_id = tl.load(q_cluster_ids + b_idx * stride_q_cid_b + q_idx * stride_q_cid_n)
    
    # 获取当前 Batch 下，该 Cluster 对应的 K/V 起止位置
    k_start = tl.load(k_cu_seqlens + b_idx * stride_k_cu_b + cluster_id * stride_k_cu_m)
    k_end = tl.load(k_cu_seqlens + b_idx * stride_k_cu_b + (cluster_id + 1) * stride_k_cu_m)
    seq_len = k_end - k_start

    # 2. 计算当前 Split 负责的 KV 起止位置
    chunk_size = tl.cdiv(seq_len, num_splits)
    chunk_start = k_start + split_idx * chunk_size
    chunk_end = tl.minimum(chunk_start + chunk_size, k_end)

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    m_i = -float('inf')
    l_i = 0.0

    if chunk_start < chunk_end:
        offs_d = tl.arange(0, BLOCK_D)
        # 加载 Query (加入 b_idx 偏移)
        q_ptrs = Q + b_idx * stride_qb + q_idx * stride_qn + offs_d * stride_qd
        q = tl.load(q_ptrs) * sm_scale

        for start_n in range(chunk_start, chunk_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k_mask = offs_n < chunk_end

            # 加载 K (加入 b_idx 偏移)
            k_ptrs = K + b_idx * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)

            qk = tl.sum(q[None, :] * k, axis=1)
            qk = tl.where(k_mask, qk, -float('inf'))

            m_ij = tl.maximum(m_i, tl.max(qk, 0))
            p = tl.math.exp(qk - m_ij)
            l_ij = tl.sum(p, 0)

            alpha = tl.math.exp(m_i - m_ij)
            acc = acc * alpha

            # 加载 V (加入 b_idx 偏移)
            v_ptrs = V + b_idx * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
            acc += tl.sum(p[:, None] * v, axis=0)

            m_i = m_ij
            l_i = l_i * alpha + l_ij

    # 3. 写回 Workspace (加入 b_idx 偏移)
    mid_acc_ptrs = mid_acc + b_idx * stride_mid_acc_b + q_idx * stride_mid_acc_q + split_idx * stride_mid_acc_s + tl.arange(0, BLOCK_D) * stride_mid_acc_d
    mid_m_ptr = mid_m + b_idx * stride_mid_m_b + q_idx * stride_mid_m_q + split_idx * stride_mid_m_s
    mid_l_ptr = mid_l + b_idx * stride_mid_l_b + q_idx * stride_mid_l_q + split_idx * stride_mid_l_s

    tl.store(mid_acc_ptrs, acc)
    tl.store(mid_m_ptr, m_i)
    tl.store(mid_l_ptr, l_i)


# =====================================================================
# Phase 2: Batched Global Reduction
# =====================================================================
@triton.jit
def batched_flash_decoding_phase2_kernel(
    mid_acc, mid_m, mid_l, Out,
    stride_mid_acc_b, stride_mid_acc_q, stride_mid_acc_s, stride_mid_acc_d,
    stride_mid_m_b, stride_mid_m_q, stride_mid_m_s,
    stride_mid_l_b, stride_mid_l_q, stride_mid_l_s,
    stride_ob, stride_on, stride_od,
    NUM_SPLITS: tl.constexpr, BLOCK_D: tl.constexpr
):
    q_idx = tl.program_id(0)
    b_idx = tl.program_id(1)  # Grid为 (N_q, B)
    offs_d = tl.arange(0, BLOCK_D)
    offs_s = tl.arange(0, NUM_SPLITS)

    # 读取当前 Batch 和 Query 的所有局部最大值 m
    mid_m_ptrs = mid_m + b_idx * stride_mid_m_b + q_idx * stride_mid_m_q + offs_s * stride_mid_m_s
    m_locals = tl.load(mid_m_ptrs)
    m_global = tl.max(m_locals, 0)

    # 读取局部指数和 l
    mid_l_ptrs = mid_l + b_idx * stride_mid_l_b + q_idx * stride_mid_l_q + offs_s * stride_mid_l_s
    l_locals = tl.load(mid_l_ptrs)

    weights = tl.math.exp(m_locals - m_global)
    l_global = tl.sum(l_locals * weights, 0)

    acc_global = tl.zeros([BLOCK_D], dtype=tl.float32)
    
    for s in range(NUM_SPLITS):
        w = tl.load(mid_m + b_idx * stride_mid_m_b + q_idx * stride_mid_m_q + s * stride_mid_m_s)
        w = tl.math.exp(w - m_global)
        
        mid_acc_ptrs = mid_acc + b_idx * stride_mid_acc_b + q_idx * stride_mid_acc_q + s * stride_mid_acc_s + offs_d * stride_mid_acc_d
        acc_local = tl.load(mid_acc_ptrs)
        acc_global += acc_local * w

    out = acc_global / l_global
    out_ptrs = Out + b_idx * stride_ob + q_idx * stride_on + offs_d * stride_od
    tl.store(out_ptrs, out)


# =====================================================================
# PyTorch 封装与多 Batch 测试代码
# =====================================================================
def batched_flash_decoding_attention(q, k, v, q_cluster_ids, k_cu_seqlens, num_splits=16):
    B, N_q, D = q.shape
    
    # Workspace 增加 B 维度
    mid_acc = torch.empty((B, N_q, num_splits, D), dtype=torch.float32, device=q.device)
    mid_m = torch.empty((B, N_q, num_splits), dtype=torch.float32, device=q.device)
    mid_l = torch.empty((B, N_q, num_splits), dtype=torch.float32, device=q.device)
    out = torch.empty_like(q)
    
    sm_scale = 1.0 / (D ** 0.5)

    # Phase 1: Grid 变为 3D -> (N_q, num_splits, B)
    grid_1 = (N_q, num_splits, B)
    batched_flash_decoding_phase1_kernel[grid_1](
        q, k, v,
        mid_acc, mid_m, mid_l,
        q_cluster_ids, k_cu_seqlens,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        mid_acc.stride(0), mid_acc.stride(1), mid_acc.stride(2), mid_acc.stride(3),
        mid_m.stride(0), mid_m.stride(1), mid_m.stride(2),
        mid_l.stride(0), mid_l.stride(1), mid_l.stride(2),
        q_cluster_ids.stride(0), q_cluster_ids.stride(1),
        k_cu_seqlens.stride(0), k_cu_seqlens.stride(1),
        BLOCK_N=16, BLOCK_D=D,
        num_warps=4
    )

    # Phase 2: Grid 变为 2D -> (N_q, B)
    grid_2 = (N_q, B)
    batched_flash_decoding_phase2_kernel[grid_2](
        mid_acc, mid_m, mid_l, out,
        mid_acc.stride(0), mid_acc.stride(1), mid_acc.stride(2), mid_acc.stride(3),
        mid_m.stride(0), mid_m.stride(1), mid_m.stride(2),
        mid_l.stride(0), mid_l.stride(1), mid_l.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        NUM_SPLITS=num_splits, BLOCK_D=D,
        num_warps=4
    )
    
    return out

# 标准 PyTorch 的 Batched 对照实现
def batched_torch_attention(q, k, v, q_cluster_ids, k_cu_seqlens):
    out = torch.zeros_like(q)
    sm_scale = 1.0 / (q.shape[2] ** 0.5)
    B, N_q, D = q.shape
    
    for b in range(B):
        for i in range(N_q):
            c_id = q_cluster_ids[b, i]
            start, end = k_cu_seqlens[b, c_id], k_cu_seqlens[b, c_id+1]
            if start >= end: continue
            
            local_k = k[b, start:end]
            local_v = v[b, start:end]
            
            scores = torch.matmul(q[b, i:i+1], local_k.transpose(0, 1)) * sm_scale
            attn = torch.softmax(scores, dim=-1)
            out[b, i:i+1] = torch.matmul(attn, local_v)
    return out

def benchmark():
    BATCH_SIZE = 32
    N_Q, N_K, D = 3, 512, 512
    M = 4 # 假设共有 4 个簇 (Clusters)
    device = torch.device('cuda')
    
    # 构造带有 Batch 维度的数据
    q = torch.randn((BATCH_SIZE, N_Q, D), dtype=torch.float32, device=device)
    k = torch.randn((BATCH_SIZE, N_K, D), dtype=torch.float32, device=device)
    v = torch.randn((BATCH_SIZE, N_K, D), dtype=torch.float32, device=device)
    
    # 随机生成每个 Token 路由到的簇 ID
    q_cluster_ids = torch.randint(0, M, (BATCH_SIZE, N_Q), dtype=torch.int32, device=device)
    
    # 模拟每个 Batch 内，4个簇各自占据的 K/V 序列边界 [B, M+1]
    # 这里为了简单测试，假设每个簇均匀分配到了 128 个 K/V
    # 实际应用中，这里应由 `torch.bincount` 或类似操作动态生成
    base_seqlens = torch.tensor([0, 128, 256, 384, 512], dtype=torch.int32, device=device)
    k_cu_seqlens = base_seqlens.unsqueeze(0).expand(BATCH_SIZE, M + 1).contiguous()

    # 验证正确性
    out_torch = batched_torch_attention(q, k, v, q_cluster_ids, k_cu_seqlens)
    out_triton = batched_flash_decoding_attention(q, k, v, q_cluster_ids, k_cu_seqlens, num_splits=16)
    
    assert torch.allclose(out_torch, out_triton, atol=1e-3), "多 Batch 精度对比失败！"
    print(f"[{BATCH_SIZE} Batch] 精度验证通过！")

    # 性能测试
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: batched_torch_attention(q, k, v, q_cluster_ids, k_cu_seqlens), quantiles=quantiles
    )
    print(f"PyTorch Batched 原生: {ms:.4f} ms")

    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: batched_flash_decoding_attention(q, k, v, q_cluster_ids, k_cu_seqlens, num_splits=16), quantiles=quantiles
    )
    print(f"Two-Stage Batched Triton: {ms:.4f} ms")

if __name__ == '__main__':
    benchmark()