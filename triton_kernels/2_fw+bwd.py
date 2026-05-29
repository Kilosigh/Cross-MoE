import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from Fused_model import triton_moe_router_and_project, batched_flash_decoding_attention, TorchCrossMoEClusteredAttention

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



def torch_native_backward(grad_output, X, q, k, v, c_q, c_k, experts_q, experts_k, w_v):
    """
    原生 PyTorch 反向传播对照组 (手动推导版本)
    输入均为前向传播时的中间结果 (未排序状态)
    """
    B, N, D = X.shape
    M = experts_q.shape[0]
    sm_scale = 1.0 / (D ** 0.5)

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    # -----------------------------------------------------------------
    # Phase 1: Attention Backward (双重循环匹配前向)
    # -----------------------------------------------------------------
    for b in range(B):
        for i in range(N):
            cluster = c_q[b, i]
            mask = (c_k[b] == cluster)
            if not mask.any():
                continue

            k_local = k[b, mask]
            v_local = v[b, mask]
            
            # 前向重算 (Recompute)
            scores = torch.matmul(q[b, i:i+1], k_local.transpose(0, 1)) * sm_scale
            attn = torch.softmax(scores, dim=-1)
            out_i = torch.matmul(attn, v_local)

            do_i = grad_output[b, i:i+1] # 当前 token 的梯度 [1, D]

            # 1. dV = attn^T @ dO
            dv_local = torch.matmul(attn.transpose(0, 1), do_i)
            dv[b, mask] += dv_local # 累加到全局 dV

            # 2. dP = dO @ V^T
            dp = torch.matmul(do_i, v_local.transpose(0, 1))

            # 3. dS = P * (dP - sum(dO * O)) * scale
            di = (do_i * out_i).sum(dim=-1, keepdim=True)
            ds = attn * (dp - di) * sm_scale

            # 4. dQ = dS @ K
            dq[b, i:i+1] += torch.matmul(ds, k_local)

            # 5. dK = dS^T @ Q
            dk_local = torch.matmul(ds.transpose(0, 1), q[b, i:i+1])
            dk[b, mask] += dk_local # 累加到全局 dK

    # -----------------------------------------------------------------
    # Phase 2: MoE与线性层 Backward (分发累加)
    # -----------------------------------------------------------------
    grad_X = torch.zeros_like(X)
    
    # Value 层反向
    grad_w_v = torch.matmul(dv.view(-1, D).t(), X.view(-1, D))
    grad_X += torch.matmul(dv, w_v)

    grad_experts_q = torch.zeros_like(experts_q)
    grad_experts_k = torch.zeros_like(experts_k)

    # Q & K 专家层反向
    for m in range(M):
        mask_q = (c_q == m)
        if mask_q.any():
            grad_experts_q[m] = torch.matmul(X[mask_q].t(), dq[mask_q])
            grad_X[mask_q] += torch.matmul(dq[mask_q], experts_q[m].t())

        mask_k = (c_k == m)
        if mask_k.any():
            grad_experts_k[m] = torch.matmul(X[mask_k].t(), dk[mask_k])
            grad_X[mask_k] += torch.matmul(dk[mask_k], experts_k[m].t())

    return grad_X, grad_experts_q, grad_experts_k, grad_w_v


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


class FinalCrossMoEAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, router_q, router_k, experts_q, experts_k, w_v_weight, num_splits=16):
        B, N, _ = X.shape
        batch_indices = torch.arange(B, device=X.device).unsqueeze(1).expand(B, N)

        # 1. 严格调用你的 Triton Forward Kernel (路由与重排)
        q_sorted, q_cluster_ids_orig, _, q_sorted_indices = triton_moe_router_and_project(
            X, router_q, experts_q
        )
        k_sorted, _, k_cu_seqlens, k_sorted_indices = triton_moe_router_and_project(
            X, router_k, experts_k
        )

        # 2. V 的映射与内存对齐
        V = F.linear(X, w_v_weight)
        v_sorted = V[batch_indices, k_sorted_indices]

        # 3. 严格调用你的 Triton Attention Kernel
        q_cluster_ids_sorted = q_cluster_ids_orig[batch_indices, q_sorted_indices]
        out_sorted = batched_flash_decoding_attention(
            q_sorted, k_sorted, v_sorted, 
            q_cluster_ids_sorted, k_cu_seqlens, 
            num_splits=num_splits
        )

        # 4. 逆排序
        q_unsort_indices = torch.argsort(q_sorted_indices, dim=-1)
        final_out = out_sorted[batch_indices, q_unsort_indices]

        # -------------------------------------------------------------
        # 保存上下文给 Backward
        # -------------------------------------------------------------
        ctx.save_for_backward(
            X, experts_q, experts_k, w_v_weight, 
            q_cluster_ids_orig, k_sorted_indices, # 记录原始路由和K的排序
            q_sorted_indices, k_cu_seqlens,
            q_sorted, k_sorted, v_sorted, out_sorted, q_cluster_ids_sorted
        )
        ctx.batch_indices = batch_indices
        ctx.M = experts_q.shape[0]
        
        return final_out

    @staticmethod
    def backward(ctx, grad_output):
        # 提取 Context
        X, experts_q, experts_k, w_v_weight, \
        c_q_orig, k_sorted_indices, q_sorted_indices, k_cu_seqlens, \
        q_sorted, k_sorted, v_sorted, out_sorted, q_cluster_ids_sorted = ctx.saved_tensors
        
        batch_indices = ctx.batch_indices
        M = ctx.M

        # 1. 梯度按 Q 排好序的形状对齐
        grad_out_sorted = grad_output[batch_indices, q_sorted_indices]

        # 2. 调用上一轮写的 Triton Attention Backward (由于依赖结构相同，直接复用)
        # 注意：你需要确保上一轮代码中的 triton_attention_backward 可用
        dq_sorted, dk_sorted, dv_sorted = triton_attention_backward(
            q_sorted, k_sorted, v_sorted, out_sorted, grad_out_sorted, 
            q_cluster_ids_sorted, k_cu_seqlens
        )

        # 3. 逆排序 dQ, dK, dV (解包回 Token 的原始物理顺序)
        dq = torch.zeros_like(grad_output)
        dk = torch.zeros_like(grad_output)
        dv = torch.zeros_like(grad_output)
        
        dq[batch_indices, q_sorted_indices] = dq_sorted
        dk[batch_indices, k_sorted_indices] = dk_sorted
        dv[batch_indices, k_sorted_indices] = dv_sorted

        # 4. 计算底层权重梯度 (复用 MoE 聚合计算)
        grad_X = torch.zeros_like(X)
        grad_w_v_weight = torch.matmul(dv.transpose(1, 2), X).sum(dim=0)
        grad_X += torch.matmul(dv, w_v_weight)

        grad_experts_q = torch.zeros_like(experts_q)
        grad_experts_k = torch.zeros_like(experts_k)

        # 这里我们需要还原 c_k_orig 以便计算 K 的掩码
        c_k_orig = torch.zeros_like(c_q_orig)
        # 从 k_cu_seqlens 推导 c_k_sorted, 再逆排回 c_k_orig
        for m in range(M):
            # Q 的掩码
            mask_q = (c_q_orig == m)
            if mask_q.any():
                grad_experts_q[m] = torch.matmul(X[mask_q].t(), dq[mask_q])
                grad_X[mask_q] += torch.matmul(dq[mask_q], experts_q[m].t())

        # 推导 K 的聚类掩码 (为了不重新计算 logits_k)
        c_k_sorted = torch.zeros_like(c_q_orig)
        for b in range(X.shape[0]):
            for m in range(M):
                start, end = k_cu_seqlens[b, m], k_cu_seqlens[b, m+1]
                c_k_sorted[b, start:end] = m
        k_unsort_indices = torch.argsort(k_sorted_indices, dim=-1)
        c_k_orig = c_k_sorted[batch_indices, k_unsort_indices]

        for m in range(M):
            mask_k = (c_k_orig == m)
            if mask_k.any():
                grad_experts_k[m] = torch.matmul(X[mask_k].t(), dk[mask_k])
                grad_X[mask_k] += torch.matmul(dk[mask_k], experts_k[m].t())

        # Router 参数由辅助 Loss 驱动，返回 None
        return grad_X, None, None, grad_experts_q, grad_experts_k, grad_w_v_weight, None

# 封装为 nn.Module
class FinalMoEAttention(nn.Module):
    def __init__(self, d_in, d_out, num_clusters):
        super().__init__()
        self.router_q = nn.Parameter(torch.randn(num_clusters, d_in) * (d_in ** -0.5))
        self.router_k = nn.Parameter(torch.randn(num_clusters, d_in) * (d_in ** -0.5))
        self.experts_q = nn.Parameter(torch.randn(num_clusters, d_in, d_out) * (d_in ** -0.5))
        self.experts_k = nn.Parameter(torch.randn(num_clusters, d_in, d_out) * (d_in ** -0.5))
        self.w_v = nn.Linear(d_in, d_out, bias=False)

    def forward(self, X):
        return FinalCrossMoEAttentionFunc.apply(
            X, self.router_q, self.router_k, self.experts_q, self.experts_k, self.w_v.weight
        )


# =====================================================================
# 终极测试：前/反向精度校验与性能压测
# =====================================================================
def run_full_verification_and_benchmark():
    B = 16          # Batch Size
    N = 1024        # Sequence Length
    D_IN = 128
    D_OUT = 128
    M = 4           # 专家/簇数量
    device = torch.device('cuda')

    print(f"🚀 初始化测试环境... [B={B}, N={N}, D_in={D_IN}, D_out={D_OUT}, M={M}]")
    
    # 1. 准备相同的输入和随机梯度
    # 使用 clone().detach().requires_grad_(True) 确保两边输入的物理内存隔离，且都能记录梯度
    X_base = torch.randn((B, N, D_IN), dtype=torch.float32, device=device)
    X_torch = X_base.clone().detach().requires_grad_(True)
    X_triton = X_base.clone().detach().requires_grad_(True)
    
    # 模拟从下游传回来的统一梯度 dO
    dO = torch.randn((B, N, D_OUT), dtype=torch.float32, device=device)

    # 2. 实例化两个模型
    torch_model = TorchCrossMoEClusteredAttention(D_IN, D_OUT, M).to(device)
    triton_model = FinalMoEAttention(D_IN, D_OUT, M).to(device)

    # 3. 强制对齐权重 (让起点完全一致)
    with torch.no_grad():
        triton_model.router_q.copy_(torch_model.router_q)
        triton_model.router_k.copy_(torch_model.router_k)
        triton_model.experts_q.copy_(torch_model.experts_q)
        triton_model.experts_k.copy_(torch_model.experts_k)
        triton_model.w_v.weight.copy_(torch_model.w_v.weight)

    # ---------------------------------------------------------
    # 第一环节：前向与反向精度校验 (Correctness Verification)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("🔍 第一环节：精度校验 (Triton vs PyTorch Native)")
    print("="*50)

    # --- 前向传播比对 ---
    out_torch = torch_model(X_torch)
    out_triton = triton_model(X_triton)
    
    fwd_match = torch.allclose(out_torch, out_triton, atol=1e-3, rtol=1e-3)
    print(f"[前向传播] 输出 Output 对齐状态 : {'✅ 成功' if fwd_match else '❌ 失败'}")
    if not fwd_match:
        print(f"  最大误差: {(out_torch - out_triton).abs().max().item()}")

    # --- 反向传播比对 ---
    # 触发原生 PyTorch 的 Autograd
    out_torch.backward(dO)
    # 触发我们手写的 Triton Autograd
    out_triton.backward(dO)

    # 由于 Triton 中的原子累加(atomic_add)和 PyTorch 内部计算顺序不同，
    # 浮点数误差(FP32)会累积，所以将 atol 稍微放宽至 1e-2 是合理的底层验证标准。
    atol_bwd = 1e-2 

    # 检查输入特征 X 的梯度
    dx_match = torch.allclose(X_torch.grad, X_triton.grad, atol=atol_bwd, rtol=1e-2)
    print(f"[反向传播] dX 梯度对齐状态       : {'✅ 成功' if dx_match else '❌ 失败'}")
    if not dx_match:
        print(f"  dX 最大误差: {(X_torch.grad - X_triton.grad).abs().max().item()}")

    # 检查 Q 专家权重的梯度
    dq_match = torch.allclose(torch_model.experts_q.grad, triton_model.experts_q.grad, atol=atol_bwd, rtol=1e-2)
    print(f"[反向传播] dW_expert_q 梯度对齐  : {'✅ 成功' if dq_match else '❌ 失败'}")

    # 检查 K 专家权重的梯度
    dk_match = torch.allclose(torch_model.experts_k.grad, triton_model.experts_k.grad, atol=atol_bwd, rtol=1e-2)
    print(f"[反向传播] dW_expert_k 梯度对齐  : {'✅ 成功' if dk_match else '❌ 失败'}")

    # 检查 V 投影权重的梯度
    dv_match = torch.allclose(torch_model.w_v.weight.grad, triton_model.w_v.weight.grad, atol=atol_bwd, rtol=1e-2)
    print(f"[反向传播] dW_v 梯度对齐         : {'✅ 成功' if dv_match else '❌ 失败'}")

# ---------------------------------------------------------
    # 第二环节：性能基准压测 (Benchmarking)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("⏱️ 第二环节：性能压测 (Forward & Backward Timing)")
    print("="*50)

    # 1. 清空第一环节积累的梯度，避免显存爆掉或精度溢出
    X_torch.grad = None
    X_triton.grad = None
    torch_model.zero_grad()
    triton_model.zero_grad()

    # 2. 重新生成专供测速的计算图
    out_torch_bench = torch_model(X_torch)
    out_triton_bench = triton_model(X_triton)

    # 3. 预热 (Warmup)
    for _ in range(3):
        out_torch_bench.backward(dO, retain_graph=True)
        out_triton_bench.backward(dO, retain_graph=True)

    quantiles = [0.5, 0.2, 0.8]

    # --- 测试闭包 ---
    def fwd_torch():
        return torch_model(X_torch)
        
    def bwd_torch():
        # 这里必须加 retain_graph=True，因为 do_bench 会循环调用数百次
        out_torch_bench.backward(dO, retain_graph=True)

    def fwd_triton():
        return triton_model(X_triton)
        
    def bwd_triton():
        out_triton_bench.backward(dO, retain_graph=True)

    # --- 测速执行 ---
    ms_fwd_pt, _, _ = triton.testing.do_bench(fwd_torch, quantiles=quantiles)
    ms_bwd_pt, _, _ = triton.testing.do_bench(bwd_torch, quantiles=quantiles)
    
    ms_fwd_tr, _, _ = triton.testing.do_bench(fwd_triton, quantiles=quantiles)
    ms_bwd_tr, _, _ = triton.testing.do_bench(bwd_triton, quantiles=quantiles)

    # --- 打印排版 ---
    print(f"{'Metric (指标)':<20} | {'PyTorch Native (ms)':<20} | {'Triton Custom (ms)':<20} | {'Speedup (加速比)'}")
    print("-" * 80)
    
    fwd_speedup = ms_fwd_pt / ms_fwd_tr
    print(f"{'Forward Pass':<20} | {ms_fwd_pt:<20.4f} | {ms_fwd_tr:<20.4f} | {fwd_speedup:.2f}x")
    
    bwd_speedup = ms_bwd_pt / ms_bwd_tr
    print(f"{'Backward Pass':<20} | {ms_bwd_pt:<20.4f} | {ms_bwd_tr:<20.4f} | {bwd_speedup:.2f}x")
    
    total_pt = ms_fwd_pt + ms_bwd_pt
    total_tr = ms_fwd_tr + ms_bwd_tr
    total_speedup = total_pt / total_tr
    print(f"{'Total (Fwd + Bwd)':<20} | {total_pt:<20.4f} | {total_tr:<20.4f} | 🔥 {total_speedup:.2f}x")
    print("=" * 80)

if __name__ == '__main__':
    run_full_verification_and_benchmark()
