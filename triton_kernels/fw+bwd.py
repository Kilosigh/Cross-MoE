import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# =====================================================================
# 1. Triton Attention Backward Kernel (简化版)
# =====================================================================
@triton.jit
def bwd_attention_kernel(
    Q, K, V, Out, dO,
    dQ, dK, dV,
    q_cluster_ids, k_cu_seqlens,
    sm_scale,
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_q_cid_b, stride_q_cid_n,
    stride_k_cu_b, stride_k_cu_m,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    # 此 Kernel 以 Q 的 Token 为粒度并行 (简化处理，实际生产中通常会以 K 分块来做减少 dK/dV 的原子操作)
    q_idx = tl.program_id(0)
    b_idx = tl.program_id(1)

    # 获取当前 Q 对应的 Cluster 和 K/V 的边界
    cluster_id = tl.load(q_cluster_ids + b_idx * stride_q_cid_b + q_idx * stride_q_cid_n)
    k_start = tl.load(k_cu_seqlens + b_idx * stride_k_cu_b + cluster_id * stride_k_cu_m)
    k_end = tl.load(k_cu_seqlens + b_idx * stride_k_cu_b + (cluster_id + 1) * stride_k_cu_m)
    seq_len = k_end - k_start

    offs_d = tl.arange(0, BLOCK_D)
    
    # 加载 Q, Out, dO (1D 向量)
    q_ptrs = Q + b_idx * stride_qb + q_idx * stride_qn + offs_d * stride_qd
    out_ptrs = Out + b_idx * stride_qb + q_idx * stride_qn + offs_d * stride_qd
    do_ptrs = dO + b_idx * stride_qb + q_idx * stride_qn + offs_d * stride_qd
    
    q = tl.load(q_ptrs)
    out = tl.load(out_ptrs)
    do = tl.load(do_ptrs)

    # 计算 Di = sum(dO * Out) 用于 dS 的偏置项
    Di = tl.sum(do * out, axis=0)
    
    dq_acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    if seq_len > 0:
        # 重计算 Logits 并找 Max (模拟缺少 LSE 时的局部重算)
        m_i = -float('inf')
        l_i = 0.0
        
        # Pass 1: 求局部 Max 和 SumExp (LSE)
        for start_n in range(k_start, k_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k_mask = offs_n < k_end
            k_ptrs = K + b_idx * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
            
            qk = tl.sum(q[None, :] * k, axis=1) * sm_scale
            qk = tl.where(k_mask, qk, -float('inf'))
            
            m_ij = tl.maximum(m_i, tl.max(qk, 0))
            p = tl.math.exp(qk - m_ij)
            l_i = l_i * tl.math.exp(m_i - m_ij) + tl.sum(p, 0)
            m_i = m_ij
            
        # Pass 2: 计算梯度
        for start_n in range(k_start, k_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k_mask = offs_n < k_end
            
            k_ptrs = K + b_idx * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            v_ptrs = V + b_idx * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            
            k = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
            
            # 1. 还原 P = softmax(QK^T)
            qk = tl.sum(q[None, :] * k, axis=1) * sm_scale
            p = tl.math.exp(qk - m_i) / l_i
            p = tl.where(k_mask, p, 0.0)
            
            # 2. dP = dO * V^T
            dp = tl.sum(do[None, :] * v, axis=1)
            
            # 3. dS = P * (dP - Di) * sm_scale
            ds = p * (dp - Di) * sm_scale
            
            # 4. dQ = dS * K
            dq_acc += tl.sum(ds[:, None] * k, axis=0)
            
            # 5. dK = dS^T * Q 
            # 6. dV = P^T * dO
            # 注意：这里多个 Q 可能会写同一个 K/V，所以生产中必须用 tl.atomic_add。
            # 为了示例能够跑通，我们在 Triton 中使用原子加。
            dk_ptrs = dK + b_idx * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            dv_ptrs = dV + b_idx * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            
            dk_val = ds[:, None] * q[None, :]
            dv_val = p[:, None] * do[None, :]
            
            tl.atomic_add(dk_ptrs, dk_val, mask=k_mask[:, None])
            tl.atomic_add(dv_ptrs, dv_val, mask=k_mask[:, None])

    # 存回 dQ
    dq_ptrs = dQ + b_idx * stride_qb + q_idx * stride_qn + offs_d * stride_qd
    tl.store(dq_ptrs, dq_acc)

def triton_attention_backward(Q, K, V, Out, dO, q_cluster_ids, k_cu_seqlens):
    B, N_q, D = Q.shape
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)
    sm_scale = 1.0 / (D ** 0.5)

    grid = (N_q, B)
    bwd_attention_kernel[grid](
        Q, K, V, Out, dO,
        dQ, dK, dV,
        q_cluster_ids, k_cu_seqlens,
        sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        q_cluster_ids.stride(0), q_cluster_ids.stride(1),
        k_cu_seqlens.stride(0), k_cu_seqlens.stride(1),
        BLOCK_N=32, BLOCK_D=D, num_warps=4
    )
    return dQ, dK, dV


# =====================================================================
# 2. 核心 autograd.Function 封装
# =====================================================================
class FastCrossMoEClusteredAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, router_q, router_k, experts_q, experts_k, w_v_weight):
        B, N, D_in = X.shape
        D_out = experts_q.shape[-1]
        batch_indices = torch.arange(B, device=X.device).unsqueeze(1).expand(B, N)
        
        # --- Phase 1: 路由与排序 (纯 PyTorch 高效实现版) ---
        c_q = torch.argmax(X @ router_q.t(), dim=-1).to(torch.int32)
        q_sorted_indices = torch.argsort(c_q, dim=-1).to(torch.int64)
        c_k = torch.argmax(X @ router_k.t(), dim=-1).to(torch.int32)
        k_sorted_indices = torch.argsort(c_k, dim=-1).to(torch.int64)

        M = experts_q.shape[0]
        q_sorted = torch.zeros((B, N, D_out), device=X.device, dtype=X.dtype)
        k_sorted = torch.zeros((B, N, D_out), device=X.device, dtype=X.dtype)
        
        for m in range(M):
            mask_q = (c_q == m)
            if mask_q.any(): q_sorted[mask_q] = X[mask_q] @ experts_q[m]
            mask_k = (c_k == m)
            if mask_k.any(): k_sorted[mask_k] = X[mask_k] @ experts_k[m]
                
        q_sorted = q_sorted[batch_indices, q_sorted_indices]
        k_sorted = k_sorted[batch_indices, k_sorted_indices]
        v_sorted = F.linear(X, w_v_weight)[batch_indices, k_sorted_indices]

        # --- Phase 2: k_cu_seqlens ---
        k_cu_seqlens = torch.zeros((B, M + 1), dtype=torch.int32, device=X.device)
        for b in range(B):
            k_cu_seqlens[b, 1:] = torch.cumsum(torch.bincount(c_k[b], minlength=M), dim=0)
            
        q_cluster_ids_sorted = c_q[batch_indices, q_sorted_indices]

        # --- Phase 3: Attention (由于你之前已有 Flash-Decoding, 这里简写为普通循环计算作为 Reference) ---
        # 实际使用中替换为你的 batched_flash_decoding_attention
        out_sorted = torch.zeros_like(q_sorted)
        sm_scale = 1.0 / (D_out ** 0.5)
        for b in range(B):
            for i in range(N):
                cluster = q_cluster_ids_sorted[b, i]
                start, end = k_cu_seqlens[b, cluster], k_cu_seqlens[b, cluster+1]
                if start < end:
                    scores = (q_sorted[b, i:i+1] @ k_sorted[b, start:end].t()) * sm_scale
                    attn = torch.softmax(scores, dim=-1)
                    out_sorted[b, i:i+1] = attn @ v_sorted[b, start:end]

        q_unsort_indices = torch.argsort(q_sorted_indices, dim=-1)
        final_out = out_sorted[batch_indices, q_unsort_indices]
        
        ctx.save_for_backward(X, experts_q, experts_k, w_v_weight, 
                              c_q, c_k, q_sorted_indices, k_sorted_indices,
                              q_sorted, k_sorted, v_sorted, out_sorted,
                              q_cluster_ids_sorted, k_cu_seqlens)
        ctx.M = M
        ctx.batch_indices = batch_indices
        return final_out

    @staticmethod
    def backward(ctx, grad_output):
        X, experts_q, experts_k, w_v_weight, c_q, c_k, q_sorted_indices, k_sorted_indices, \
        q_sorted, k_sorted, v_sorted, out_sorted, q_cluster_ids_sorted, k_cu_seqlens = ctx.saved_tensors
        
        B, N, D_in = X.shape
        batch_indices = ctx.batch_indices
        M = ctx.M

        # 1. 梯度对齐到 Q 排好序的形状
        grad_out_sorted = grad_output[batch_indices, q_sorted_indices]

        # 2. 调用 Triton Attention Backward
        dq_sorted, dk_sorted, dv_sorted = triton_attention_backward(
            q_sorted, k_sorted, v_sorted, out_sorted, grad_out_sorted, 
            q_cluster_ids_sorted, k_cu_seqlens
        )

        # 3. 逆排序回原生物理位置
        dq = torch.zeros_like(grad_output)
        dk = torch.zeros_like(grad_output)
        dv = torch.zeros_like(grad_output)
        
        dq[batch_indices, q_sorted_indices] = dq_sorted
        dk[batch_indices, k_sorted_indices] = dk_sorted
        dv[batch_indices, k_sorted_indices] = dv_sorted

        # 4. 权重梯度计算
        grad_X = torch.zeros_like(X)
        grad_w_v_weight = torch.matmul(dv.transpose(1, 2), X).sum(dim=0)
        grad_X += torch.matmul(dv, w_v_weight)

        grad_experts_q = torch.zeros_like(experts_q)
        grad_experts_k = torch.zeros_like(experts_k)

        for m in range(M):
            mask_q = (c_q == m)
            if mask_q.any():
                X_m_q = X[mask_q]
                dq_m = dq[mask_q]
                grad_experts_q[m] = torch.matmul(X_m_q.t(), dq_m)
                grad_X[mask_q] += torch.matmul(dq_m, experts_q[m].t())

            mask_k = (c_k == m)
            if mask_k.any():
                X_m_k = X[mask_k]
                dk_m = dk[mask_k]
                grad_experts_k[m] = torch.matmul(X_m_k.t(), dk_m)
                grad_X[mask_k] += torch.matmul(dk_m, experts_k[m].t())

        return grad_X, None, None, grad_experts_q, grad_experts_k, grad_w_v_weight


class EndToEndCrossMoEAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, num_clusters: int):
        super().__init__()
        self.router_q = nn.Parameter(torch.randn(num_clusters, d_in) * (d_in ** -0.5))
        self.router_k = nn.Parameter(torch.randn(num_clusters, d_in) * (d_in ** -0.5))
        self.experts_q = nn.Parameter(torch.randn(num_clusters, d_in, d_out) * (d_in ** -0.5))
        self.experts_k = nn.Parameter(torch.randn(num_clusters, d_in, d_out) * (d_in ** -0.5))
        self.w_v = nn.Linear(d_in, d_out, bias=False)

    def forward(self, X):
        return FastCrossMoEClusteredAttentionFunc.apply(
            X, self.router_q, self.router_k, self.experts_q, self.experts_k, self.w_v.weight
        )


# =====================================================================
# 3. 性能压测与校验 (Benchmark)
# =====================================================================
def run_e2e_benchmark():
    B = 16
    N = 1024
    D_IN = 128
    D_OUT = 128
    M = 4
    device = torch.device('cuda')

    print(f"🚀 初始化环境... [B={B}, N={N}, D={D_IN}, Clusters={M}]")
    X = torch.randn((B, N, D_IN), dtype=torch.float32, device=device, requires_grad=True)
    dO = torch.randn((B, N, D_OUT), dtype=torch.float32, device=device)

    model = EndToEndCrossMoEAttention(D_IN, D_OUT, M).to(device)

    # --- 1. 热身 (Warmup) ---
    print("🔥 正在预热 GPU...")
    for _ in range(3):
        out = model(X)
        out.backward(dO, retain_graph=True)
        model.zero_grad()
        X.grad = None

    # --- 2. 测量前向耗时 ---
    def fwd_fn():
        return model(X)

    # --- 3. 测量反向耗时 ---
    out = fwd_fn()
    def bwd_fn():
        out.backward(dO, retain_graph=True)

    quantiles = [0.5, 0.2, 0.8]
    ms_fwd, min_fwd, max_fwd = triton.testing.do_bench(fwd_fn, quantiles=quantiles)
    ms_bwd, min_bwd, max_bwd = triton.testing.do_bench(bwd_fn, quantiles=quantiles)

    print("\n" + "="*50)
    print("📊 压测结果 (端到端 Forward + Backward)")
    print("="*50)
    print(f"前向耗时 (Forward)  : {ms_fwd:.4f} ms")
    print(f"反向耗时 (Backward) : {ms_bwd:.4f} ms")
    print(f"总计耗时 (Total)    : {ms_fwd + ms_bwd:.4f} ms")
    print("="*50)
    
    # --- 4. 验证梯度是否成功生成 ---
    out = model(X)
    out.backward(dO)
    print("\n✅ 梯度状态检查：")
    print(f"dX_grad       生成成功? : {X.grad is not None}")
    print(f"dW_expert_q   生成成功? : {model.experts_q.grad is not None}")
    print(f"dW_v_weight   生成成功? : {model.w_v.weight.grad is not None}")


if __name__ == '__main__':
    run_e2e_benchmark()