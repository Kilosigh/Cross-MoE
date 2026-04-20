import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# =====================================================================
# 1. 前向传播 (Forward Kernels)
# =====================================================================
@triton.jit
def fused_gather_expert_mha_kernel(
    X, Experts, Out, sorted_indices, cluster_ids,
    stride_xb, stride_xn, stride_xh, stride_xd,
    stride_em, stride_eh, stride_ed_in, stride_ed_out,
    stride_ob, stride_on, stride_oh, stride_od,
    stride_idx_b, stride_idx_n, stride_cid_b, stride_cid_n,
    actual_d_in, actual_d_out, H,
    BLOCK_DIN: tl.constexpr, BLOCK_DOUT: tl.constexpr
):
    b_idx = tl.program_id(0)
    out_token_idx = tl.program_id(1)
    pid_z = tl.program_id(2)
    
    num_d_blocks = tl.cdiv(actual_d_out, BLOCK_DOUT)
    h_idx = pid_z // num_d_blocks
    out_d_idx = pid_z % num_d_blocks

    src_token_idx = tl.load(sorted_indices + b_idx * stride_idx_b + out_token_idx * stride_idx_n)
    c_id = tl.load(cluster_ids + b_idx * stride_cid_b + src_token_idx * stride_cid_n)

    offs_din = tl.arange(0, BLOCK_DIN)
    offs_dout = out_d_idx * BLOCK_DOUT + tl.arange(0, BLOCK_DOUT)

    mask_din = offs_din < actual_d_in
    mask_dout = offs_dout < actual_d_out

    x_ptrs = X + b_idx * stride_xb + src_token_idx * stride_xn + h_idx * stride_xh + offs_din * stride_xd
    x = tl.load(x_ptrs, mask=mask_din, other=0.0) 

    w_ptrs = Experts + c_id * stride_em + h_idx * stride_eh + offs_din[:, None] * stride_ed_in + offs_dout[None, :] * stride_ed_out
    w = tl.load(w_ptrs, mask=mask_din[:, None] & mask_dout[None, :], other=0.0)

    out = tl.sum(x[:, None] * w, axis=0)
    out_ptrs = Out + b_idx * stride_ob + out_token_idx * stride_on + h_idx * stride_oh + offs_dout * stride_od
    tl.store(out_ptrs, out, mask=mask_dout)

def triton_moe_router_and_project_mha(X, Router, Experts, H):
    B, N, D_in = X.shape
    M, _, D_in_head, D_out_head = Experts.shape
    
    logits = torch.matmul(X, Router.transpose(0, 1))
    cluster_ids = torch.argmax(logits, dim=-1).to(torch.int32)
    
    sorted_indices = torch.argsort(cluster_ids, dim=-1).to(torch.int32)
    cu_seqlens = torch.zeros((B, M + 1), dtype=torch.int32, device=X.device)
    for b in range(B):
        cu_seqlens[b, 1:] = torch.cumsum(torch.bincount(cluster_ids[b], minlength=M), dim=0)

    X_view = X.view(B, N, H, D_in_head)
    out = torch.empty((B, N, H, D_out_head), dtype=X.dtype, device=X.device)
    
    BLOCK_DIN = triton.next_power_of_2(D_in_head)
    BLOCK_DOUT = 32 if D_out_head > 16 else 16 
    
    grid = (B, N, H * triton.cdiv(D_out_head, BLOCK_DOUT))
    fused_gather_expert_mha_kernel[grid](
        X_view, Experts, out, sorted_indices, cluster_ids,
        X_view.stride(0), X_view.stride(1), X_view.stride(2), X_view.stride(3),
        Experts.stride(0), Experts.stride(1), Experts.stride(2), Experts.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        sorted_indices.stride(0), sorted_indices.stride(1),
        cluster_ids.stride(0), cluster_ids.stride(1),
        D_in_head, D_out_head, H,
        BLOCK_DIN=BLOCK_DIN, BLOCK_DOUT=BLOCK_DOUT, num_warps=4, num_stages=2
    )
    return out, cluster_ids, cu_seqlens, sorted_indices


@triton.jit
def batched_flash_decoding_mha_phase1(
    Q, K, V, mid_acc, mid_m, mid_l, q_cluster_ids, k_cu_seqlens, sm_scale,
    stride_qb, stride_qn, stride_qh, stride_qd,
    stride_kb, stride_kn, stride_kh, stride_kd,
    stride_vb, stride_vn, stride_vh, stride_vd,
    stride_mid_acc_b, stride_mid_acc_h, stride_mid_acc_q, stride_mid_acc_s, stride_mid_acc_d,
    stride_mid_m_b, stride_mid_m_h, stride_mid_m_q, stride_mid_m_s,
    stride_mid_l_b, stride_mid_l_h, stride_mid_l_q, stride_mid_l_s,
    stride_q_cid_b, stride_q_cid_n, stride_k_cu_b, stride_k_cu_m,
    actual_d, H, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    q_idx = tl.program_id(0)
    split_idx = tl.program_id(1)
    pid_bh = tl.program_id(2)
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    num_splits = tl.num_programs(1)

    cluster_id = tl.load(q_cluster_ids + b_idx * stride_q_cid_b + q_idx * stride_q_cid_n)
    k_start = tl.load(k_cu_seqlens + b_idx * stride_k_cu_b + cluster_id * stride_k_cu_m)
    k_end = tl.load(k_cu_seqlens + b_idx * stride_k_cu_b + (cluster_id + 1) * stride_k_cu_m)
    seq_len = k_end - k_start

    chunk_size = tl.cdiv(seq_len, num_splits)
    chunk_start = k_start + split_idx * chunk_size
    chunk_end = tl.minimum(chunk_start + chunk_size, k_end)

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    m_i = -float('inf')
    l_i = 0.0

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < actual_d 

    if chunk_start < chunk_end:
        q_ptrs = Q + b_idx * stride_qb + q_idx * stride_qn + h_idx * stride_qh + offs_d * stride_qd
        q = tl.load(q_ptrs, mask=mask_d, other=0.0) * sm_scale

        for start_n in range(chunk_start, chunk_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k_mask = offs_n < chunk_end
            mask_2d = k_mask[:, None] & mask_d[None, :]

            k_ptrs = K + b_idx * stride_kb + offs_n[:, None] * stride_kn + h_idx * stride_kh + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=mask_2d, other=0.0)

            qk = tl.sum(q[None, :] * k, axis=1)
            qk = tl.where(k_mask, qk, -float('inf'))

            m_ij = tl.maximum(m_i, tl.max(qk, 0))
            p = tl.math.exp(qk - m_ij)
            l_ij = tl.sum(p, 0)

            alpha = tl.math.exp(m_i - m_ij)
            acc = acc * alpha

            v_ptrs = V + b_idx * stride_vb + offs_n[:, None] * stride_vn + h_idx * stride_vh + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=mask_2d, other=0.0)
            acc += tl.sum(p[:, None] * v, axis=0)

            m_i = m_ij
            l_i = l_i * alpha + l_ij

    tl.store(mid_acc + b_idx * stride_mid_acc_b + h_idx * stride_mid_acc_h + q_idx * stride_mid_acc_q + split_idx * stride_mid_acc_s + offs_d * stride_mid_acc_d, acc, mask=mask_d)
    tl.store(mid_m + b_idx * stride_mid_m_b + h_idx * stride_mid_m_h + q_idx * stride_mid_m_q + split_idx * stride_mid_m_s, m_i)
    tl.store(mid_l + b_idx * stride_mid_l_b + h_idx * stride_mid_l_h + q_idx * stride_mid_l_q + split_idx * stride_mid_l_s, l_i)

@triton.jit
def batched_flash_decoding_mha_phase2(
    mid_acc, mid_m, mid_l, Out, LSE, 
    stride_mid_acc_b, stride_mid_acc_h, stride_mid_acc_q, stride_mid_acc_s, stride_mid_acc_d,
    stride_mid_m_b, stride_mid_m_h, stride_mid_m_q, stride_mid_m_s,
    stride_mid_l_b, stride_mid_l_h, stride_mid_l_q, stride_mid_l_s,
    stride_ob, stride_on, stride_oh, stride_od,
    stride_lse_b, stride_lse_n, stride_lse_h,    
    actual_d, H, NUM_SPLITS: tl.constexpr, BLOCK_D: tl.constexpr
):
    q_idx = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    
    offs_d = tl.arange(0, BLOCK_D)
    offs_s = tl.arange(0, NUM_SPLITS)
    mask_d = offs_d < actual_d 

    m_locals = tl.load(mid_m + b_idx * stride_mid_m_b + h_idx * stride_mid_m_h + q_idx * stride_mid_m_q + offs_s * stride_mid_m_s)
    m_global = tl.max(m_locals, 0)

    l_locals = tl.load(mid_l + b_idx * stride_mid_l_b + h_idx * stride_mid_l_h + q_idx * stride_mid_l_q + offs_s * stride_mid_l_s)
    weights = tl.math.exp(m_locals - m_global)
    l_global = tl.sum(l_locals * weights, 0)

    tl.store(LSE + b_idx * stride_lse_b + q_idx * stride_lse_n + h_idx * stride_lse_h, m_global + tl.math.log(l_global))

    acc_global = tl.zeros([BLOCK_D], dtype=tl.float32)
    for s in range(NUM_SPLITS):
        w = tl.math.exp(tl.load(mid_m + b_idx * stride_mid_m_b + h_idx * stride_mid_m_h + q_idx * stride_mid_m_q + s * stride_mid_m_s) - m_global)
        acc_local = tl.load(mid_acc + b_idx * stride_mid_acc_b + h_idx * stride_mid_acc_h + q_idx * stride_mid_acc_q + s * stride_mid_acc_s + offs_d * stride_mid_acc_d, mask=mask_d, other=0.0) 
        acc_global += acc_local * w

    tl.store(Out + b_idx * stride_ob + q_idx * stride_on + h_idx * stride_oh + offs_d * stride_od, acc_global / l_global, mask=mask_d)

def batched_flash_decoding_mha(q, k, v, q_cluster_ids, k_cu_seqlens, num_splits=16):
    B, N_q, H, D = q.shape
    mid_acc = torch.empty((B, H, N_q, num_splits, D), dtype=torch.float32, device=q.device)
    mid_m = torch.empty((B, H, N_q, num_splits), dtype=torch.float32, device=q.device)
    mid_l = torch.empty((B, H, N_q, num_splits), dtype=torch.float32, device=q.device)
    out = torch.empty_like(q)
    LSE = torch.empty((B, N_q, H), dtype=torch.float32, device=q.device) 
    
    sm_scale = 1.0 / (D ** 0.5)
    BLOCK_D = triton.next_power_of_2(D)

    batched_flash_decoding_mha_phase1[(N_q, num_splits, B * H)](
        q, k, v, mid_acc, mid_m, mid_l, q_cluster_ids, k_cu_seqlens, sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        mid_acc.stride(0), mid_acc.stride(1), mid_acc.stride(2), mid_acc.stride(3), mid_acc.stride(4),
        mid_m.stride(0), mid_m.stride(1), mid_m.stride(2), mid_m.stride(3),
        mid_l.stride(0), mid_l.stride(1), mid_l.stride(2), mid_l.stride(3),
        q_cluster_ids.stride(0), q_cluster_ids.stride(1), k_cu_seqlens.stride(0), k_cu_seqlens.stride(1),
        actual_d=D, H=H, BLOCK_N=32, BLOCK_D=BLOCK_D, num_warps=4
    )

    batched_flash_decoding_mha_phase2[(N_q, B * H)](
        mid_acc, mid_m, mid_l, out, LSE,
        mid_acc.stride(0), mid_acc.stride(1), mid_acc.stride(2), mid_acc.stride(3), mid_acc.stride(4),
        mid_m.stride(0), mid_m.stride(1), mid_m.stride(2), mid_m.stride(3),
        mid_l.stride(0), mid_l.stride(1), mid_l.stride(2), mid_l.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        actual_d=D, H=H, NUM_SPLITS=num_splits, BLOCK_D=BLOCK_D, num_warps=4
    )
    return out, LSE


# =====================================================================
# 2. 反向传播 (Backward Kernels Multi-Head 改造)
# =====================================================================
@triton.jit
def bwd_kernel_dq_mha(
    Q, K, V, dO, LSE, Delta, dQ,
    q_cu_seqlens, k_cu_seqlens, sm_scale,
    stride_qb, stride_qn, stride_qh, stride_qd,
    stride_kb, stride_kn, stride_kh, stride_kd,
    stride_vb, stride_vn, stride_vh, stride_vd,   
    stride_lse_b, stride_lse_n, stride_lse_h,
    stride_delta_b, stride_delta_n, stride_delta_h,
    stride_q_cu_b, stride_k_cu_b,
    actual_d: tl.constexpr, H: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    cluster_id = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    q_block_idx = tl.program_id(2)

    q_start = tl.load(q_cu_seqlens + b_idx * stride_q_cu_b + cluster_id)
    q_end = tl.load(q_cu_seqlens + b_idx * stride_q_cu_b + cluster_id + 1)
    
    offs_q = q_start + q_block_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = offs_q < q_end
    
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < actual_d

    q_base = Q + b_idx * stride_qb + h_idx * stride_qh
    do_base = dO + b_idx * stride_qb + h_idx * stride_qh
    lse_base = LSE + b_idx * stride_lse_b + h_idx * stride_lse_h
    delta_base = Delta + b_idx * stride_delta_b + h_idx * stride_delta_h
    k_base = K + b_idx * stride_kb + h_idx * stride_kh
    v_base = V + b_idx * stride_vb + h_idx * stride_vh

    q = tl.load(q_base + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd, mask=q_mask[:, None] & mask_d[None, :], other=0.0)
    do = tl.load(do_base + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd, mask=q_mask[:, None] & mask_d[None, :], other=0.0)
    
    lse = tl.load(lse_base + offs_q * stride_lse_n, mask=q_mask, other=0.0).to(tl.float32)
    delta = tl.load(delta_base + offs_q * stride_delta_n, mask=q_mask, other=0.0).to(tl.float32)

    dq_acc = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)

    k_start = tl.load(k_cu_seqlens + b_idx * stride_k_cu_b + cluster_id)
    k_end = tl.load(k_cu_seqlens + b_idx * stride_k_cu_b + cluster_id + 1)

    # 🔥 核心修复：就算 k_start == k_end（空簇），强行让它执行一次，绕过 LLVM Bug
    k_loop_end = tl.maximum(k_end, k_start + 1)

    for k_offs_base in range(k_start, k_loop_end, BLOCK_K):
        offs_k = k_offs_base + tl.arange(0, BLOCK_K)
        # 如果是空簇(k_end==k_start)，k_mask 全部为 False，下面所有的 load 和 dot 都会变成空转计算 0
        k_mask = offs_k < k_end

        k = tl.load(k_base + offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd, mask=k_mask[:, None] & mask_d[None, :], other=0.0)
        v = tl.load(v_base + offs_k[:, None] * stride_vn + offs_d[None, :] * stride_vd, mask=k_mask[:, None] & mask_d[None, :], other=0.0)

        qk = tl.dot(q, tl.trans(k)) * sm_scale
        p = tl.math.exp(qk.to(tl.float32) - lse[:, None])
        p = tl.where(q_mask[:, None] & k_mask[None, :], p, 0.0)

        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp.to(tl.float32) - delta[:, None]) * sm_scale
        
        ds_cast = ds.to(q.dtype)
        dq_acc += tl.dot(ds_cast, k)

    dq_ptrs = dQ + b_idx * stride_qb + h_idx * stride_qh + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd
    tl.store(dq_ptrs, dq_acc.to(q.dtype), mask=q_mask[:, None] & mask_d[None, :])

@triton.jit
def bwd_kernel_dk_dv_mha(
    Q, K, V, dO, LSE, Delta, dK, dV,
    q_cu_seqlens, k_cu_seqlens, sm_scale,
    stride_qb, stride_qn, stride_qh, stride_qd,
    stride_kb, stride_kn, stride_kh, stride_kd,
    stride_vb, stride_vn, stride_vh, stride_vd,
    stride_lse_b, stride_lse_n, stride_lse_h,
    stride_delta_b, stride_delta_n, stride_delta_h,
    stride_q_cu_b, stride_k_cu_b,
    actual_d: tl.constexpr, H: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    cluster_id = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    k_block_idx = tl.program_id(2)

    k_start = tl.load(k_cu_seqlens + b_idx * stride_k_cu_b + cluster_id)
    k_end = tl.load(k_cu_seqlens + b_idx * stride_k_cu_b + cluster_id + 1)
    
    offs_k = k_start + k_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = offs_k < k_end
    
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < actual_d

    q_base = Q + b_idx * stride_qb + h_idx * stride_qh
    do_base = dO + b_idx * stride_qb + h_idx * stride_qh
    lse_base = LSE + b_idx * stride_lse_b + h_idx * stride_lse_h
    delta_base = Delta + b_idx * stride_delta_b + h_idx * stride_delta_h
    k_base = K + b_idx * stride_kb + h_idx * stride_kh
    v_base = V + b_idx * stride_vb + h_idx * stride_vh

    k = tl.load(k_base + offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd, mask=k_mask[:, None] & mask_d[None, :], other=0.0)
    v = tl.load(v_base + offs_k[:, None] * stride_vn + offs_d[None, :] * stride_vd, mask=k_mask[:, None] & mask_d[None, :], other=0.0)

    dk_acc = tl.zeros([BLOCK_K, BLOCK_D], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_K, BLOCK_D], dtype=tl.float32)

    q_start = tl.load(q_cu_seqlens + b_idx * stride_q_cu_b + cluster_id)
    q_end = tl.load(q_cu_seqlens + b_idx * stride_q_cu_b + cluster_id + 1)

    # 🔥 核心修复：防止 Sq 过短导致的 0 次查询空循环
    q_loop_end = tl.maximum(q_end, q_start + 1)

    for q_offs_base in range(q_start, q_loop_end, BLOCK_Q):
        offs_q = q_offs_base + tl.arange(0, BLOCK_Q)
        q_mask = offs_q < q_end

        q = tl.load(q_base + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd, mask=q_mask[:, None] & mask_d[None, :], other=0.0)
        do = tl.load(do_base + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd, mask=q_mask[:, None] & mask_d[None, :], other=0.0)
        
        lse = tl.load(lse_base + offs_q * stride_lse_n, mask=q_mask, other=0.0).to(tl.float32)
        delta = tl.load(delta_base + offs_q * stride_delta_n, mask=q_mask, other=0.0).to(tl.float32)

        qk_t = tl.dot(k, tl.trans(q)) * sm_scale
        pt = tl.math.exp(qk_t.to(tl.float32) - lse[None, :])
        pt = tl.where(k_mask[:, None] & q_mask[None, :], pt, 0.0)

        pt_cast = pt.to(do.dtype)
        dv_acc += tl.dot(pt_cast, do)

        dp_t = tl.dot(v, tl.trans(do))
        ds_t = pt * (dp_t.to(tl.float32) - delta[None, :]) * sm_scale
        
        ds_t_cast = ds_t.to(q.dtype)
        dk_acc += tl.dot(ds_t_cast, q)

    dk_ptrs = dK + b_idx * stride_kb + h_idx * stride_kh + offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd
    dv_ptrs = dV + b_idx * stride_vb + h_idx * stride_vh + offs_k[:, None] * stride_vn + offs_d[None, :] * stride_vd
    
    tl.store(dk_ptrs, dk_acc.to(k.dtype), mask=k_mask[:, None] & mask_d[None, :])
    tl.store(dv_ptrs, dv_acc.to(v.dtype), mask=k_mask[:, None] & mask_d[None, :])


def triton_attention_backward_split_mha(Q_sorted, K_sorted, V_sorted, Out_sorted, dO_sorted, 
                                        LSE_sorted, q_cluster_ids, k_cu_seqlens, M, H):
    B, N, H_dim, D_head = Q_sorted.shape
    dQ = torch.zeros_like(Q_sorted)
    dK = torch.zeros_like(K_sorted)
    dV = torch.zeros_like(V_sorted)
    sm_scale = 1.0 / (D_head ** 0.5)

    Delta = (dO_sorted * Out_sorted).sum(dim=-1).contiguous()

    q_cu_seqlens = torch.zeros((B, M + 1), dtype=torch.int32, device=Q_sorted.device)
    for b in range(B):
        q_cu_seqlens[b, 1:] = torch.cumsum(torch.bincount(q_cluster_ids[b], minlength=M), dim=0)

    BLOCK_Q = 32
    BLOCK_K = 32
    BLOCK_D = triton.next_power_of_2(D_head)

    q_lens = q_cu_seqlens[:, 1:] - q_cu_seqlens[:, :-1]
    k_lens = k_cu_seqlens[:, 1:] - k_cu_seqlens[:, :-1]
    
    max_q_len = max(q_lens.max().item(), 1)
    max_k_len = max(k_lens.max().item(), 1)
    max_q_blocks = triton.cdiv(max_q_len, BLOCK_Q)
    max_k_blocks = triton.cdiv(max_k_len, BLOCK_K)

    grid_dq = (M, B * H, max_q_blocks)
    bwd_kernel_dq_mha[grid_dq](
        Q_sorted, K_sorted, V_sorted, dO_sorted, LSE_sorted, Delta, dQ,
        q_cu_seqlens, k_cu_seqlens, sm_scale,
        Q_sorted.stride(0), Q_sorted.stride(1), Q_sorted.stride(2), Q_sorted.stride(3),
        K_sorted.stride(0), K_sorted.stride(1), K_sorted.stride(2), K_sorted.stride(3),
        V_sorted.stride(0), V_sorted.stride(1), V_sorted.stride(2), V_sorted.stride(3), 
        LSE_sorted.stride(0), LSE_sorted.stride(1), LSE_sorted.stride(2), 
        Delta.stride(0), Delta.stride(1), Delta.stride(2),             
        q_cu_seqlens.stride(0), k_cu_seqlens.stride(0),    
        actual_d=D_head, H=H, BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D, num_stages=1
    )

    grid_dk_dv = (M, B * H, max_k_blocks)
    bwd_kernel_dk_dv_mha[grid_dk_dv](
        Q_sorted, K_sorted, V_sorted, dO_sorted, LSE_sorted, Delta, dK, dV,
        q_cu_seqlens, k_cu_seqlens, sm_scale,
        Q_sorted.stride(0), Q_sorted.stride(1), Q_sorted.stride(2), Q_sorted.stride(3),
        K_sorted.stride(0), K_sorted.stride(1), K_sorted.stride(2), K_sorted.stride(3),
        V_sorted.stride(0), V_sorted.stride(1), V_sorted.stride(2), V_sorted.stride(3), 
        LSE_sorted.stride(0), LSE_sorted.stride(1), LSE_sorted.stride(2), 
        Delta.stride(0), Delta.stride(1), Delta.stride(2),             
        q_cu_seqlens.stride(0), k_cu_seqlens.stride(0),    
        actual_d=D_head, H=H, BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D, num_stages=1
    )

    return dQ, dK, dV


# =====================================================================
# 3. 核心 PyTorch 封装 (Autograd & Module)
# =====================================================================
class FinalCrossMoEMultiHeadAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X_q, X_kv, router_q, router_k, experts_q, experts_k, w_v_weight, H=8, num_splits=16):
        B, N_q, _ = X_q.shape
        _, N_kv, _ = X_kv.shape
        D_out_head = experts_q.shape[-1]
        
        batch_indices_q = torch.arange(B, device=X_q.device).unsqueeze(1).expand(B, N_q)
        batch_indices_kv = torch.arange(B, device=X_kv.device).unsqueeze(1).expand(B, N_kv)

        q_sorted, q_cluster_ids_orig, _, q_sorted_indices = triton_moe_router_and_project_mha(
            X_q, router_q, experts_q, H
        )
        k_sorted, _, k_cu_seqlens, k_sorted_indices = triton_moe_router_and_project_mha(
            X_kv, router_k, experts_k, H
        )

        V = F.linear(X_kv, w_v_weight).view(B, N_kv, H, D_out_head)
        v_sorted = V[batch_indices_kv, k_sorted_indices]

        q_cluster_ids_sorted = q_cluster_ids_orig[batch_indices_q, q_sorted_indices]
        out_sorted, LSE_sorted = batched_flash_decoding_mha(
            q_sorted, k_sorted, v_sorted, q_cluster_ids_sorted, k_cu_seqlens, num_splits=num_splits
        )

        q_unsort_indices = torch.argsort(q_sorted_indices, dim=-1)
        final_out = out_sorted[batch_indices_q, q_unsort_indices]

        ctx.save_for_backward(
            X_q, X_kv, experts_q, experts_k, w_v_weight, 
            q_cluster_ids_orig, k_sorted_indices, q_sorted_indices, k_cu_seqlens,
            q_sorted, k_sorted, v_sorted, out_sorted, q_cluster_ids_sorted, LSE_sorted
        )
        ctx.batch_indices_q = batch_indices_q
        ctx.batch_indices_kv = batch_indices_kv
        ctx.M = experts_q.shape[0]
        ctx.H = H
        ctx.D_in_head = experts_q.shape[-2]
        
        return final_out.view(B, N_q, -1)

    @staticmethod
    def backward(ctx, grad_output):
        X_q, X_kv, experts_q, experts_k, w_v_weight, \
        c_q_orig, k_sorted_indices, q_sorted_indices, k_cu_seqlens, \
        q_sorted, k_sorted, v_sorted, out_sorted, q_cluster_ids_sorted, \
        LSE_sorted = ctx.saved_tensors
        
        grad_output = grad_output.to(X_q.dtype)

        B, N_q, _ = X_q.shape
        _, N_kv, _ = X_kv.shape
        batch_indices_q = ctx.batch_indices_q
        batch_indices_kv = ctx.batch_indices_kv
        M = ctx.M
        H = ctx.H
        D_in_head = ctx.D_in_head
        D_out_head = experts_q.shape[-1]
        
        # 将梯度按照 Q 物理位置转为多头 Sorted 形状
        grad_output_mha = grad_output.view(B, N_q, H, D_out_head)
        grad_out_sorted = grad_output_mha[batch_indices_q, q_sorted_indices].contiguous()

        v_sorted = v_sorted.contiguous()

        # 🔥 将收到的梯度和所有的传入张量强行刷成连续内存
        grad_output_mha = grad_output.contiguous().view(B, N_q, H, D_out_head)
        grad_out_sorted = grad_output_mha[batch_indices_q, q_sorted_indices].contiguous()

        q_sorted = q_sorted.contiguous()
        k_sorted = k_sorted.contiguous()
        v_sorted = v_sorted.contiguous()
        out_sorted = out_sorted.contiguous()
        LSE_sorted = LSE_sorted.contiguous()
        q_cluster_ids_sorted = q_cluster_ids_sorted.contiguous()
        k_cu_seqlens = k_cu_seqlens.contiguous()


        dq_sorted, dk_sorted, dv_sorted = triton_attention_backward_split_mha(
            q_sorted, k_sorted, v_sorted, out_sorted, grad_out_sorted, 
            LSE_sorted, q_cluster_ids_sorted, k_cu_seqlens, M, H
        )

        dq = torch.zeros_like(grad_output_mha) 
        dk = torch.zeros((B, N_kv, H, D_out_head), dtype=grad_output.dtype, device=grad_output.device)
        dv = torch.zeros_like(dk)
        
        dq[batch_indices_q, q_sorted_indices] = dq_sorted
        dk[batch_indices_kv, k_sorted_indices] = dk_sorted
        dv[batch_indices_kv, k_sorted_indices] = dv_sorted

        grad_X_q = torch.zeros_like(X_q)
        grad_X_kv = torch.zeros_like(X_kv)
        
        # Value 投影梯度
        dv_flat = dv.view(-1, H * D_out_head)
        X_kv_flat = X_kv.view(-1, H * D_in_head)
        grad_w_v_weight = torch.matmul(dv_flat.t(), X_kv_flat)
        grad_X_kv += torch.matmul(dv_flat, w_v_weight).view(B, N_kv, -1)

        grad_experts_q = torch.zeros_like(experts_q)
        grad_experts_k = torch.zeros_like(experts_k)

        # 映射 X 供 einsum 切片使用
        X_q_view = X_q.view(B, N_q, H, D_in_head)
        grad_X_q_view = grad_X_q.view(B, N_q, H, D_in_head)
        
        X_kv_view = X_kv.view(B, N_kv, H, D_in_head)
        grad_X_kv_view = grad_X_kv.view(B, N_kv, H, D_in_head)

        # 推导 K 的聚类掩码 
        c_k_sorted = torch.zeros_like(k_sorted_indices, dtype=torch.int32)
        for b in range(B):
            for m in range(M):
                start, end = k_cu_seqlens[b, m], k_cu_seqlens[b, m+1]
                c_k_sorted[b, start:end] = m
        k_unsort_indices = torch.argsort(k_sorted_indices, dim=-1)
        c_k_orig = c_k_sorted[batch_indices_kv, k_unsort_indices]

        # 计算权重与输入梯度 (支持多头 einsum)
        for m in range(M):
            mask_q = (c_q_orig == m)
            if mask_q.any():
                x_m_q = X_q_view[mask_q]
                dq_m = dq[mask_q]
                grad_experts_q[m] = torch.einsum('thd, tho -> hdo', x_m_q, dq_m)
                grad_X_q_view[mask_q] += torch.einsum('tho, hdo -> thd', dq_m, experts_q[m])

            mask_k = (c_k_orig == m)
            if mask_k.any():
                x_m_k = X_kv_view[mask_k]
                dk_m = dk[mask_k]
                grad_experts_k[m] = torch.einsum('thd, tho -> hdo', x_m_k, dk_m)
                grad_X_kv_view[mask_k] += torch.einsum('tho, hdo -> thd', dk_m, experts_k[m])

        return grad_X_q, grad_X_kv, None, None, grad_experts_q, grad_experts_k, grad_w_v_weight, None, None

class FinalCrossMoEMultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, num_clusters):
        super().__init__()
        self.H = num_heads
        d_in_head = d_in // num_heads
        d_out_head = d_out // num_heads
        self.router_q = nn.Parameter(torch.randn(num_clusters, d_in) * (d_in ** -0.5))
        self.router_k = nn.Parameter(torch.randn(num_clusters, d_in) * (d_in ** -0.5))
        self.experts_q = nn.Parameter(torch.randn(num_clusters, num_heads, d_in_head, d_out_head) * (d_in_head ** -0.5))
        self.experts_k = nn.Parameter(torch.randn(num_clusters, num_heads, d_in_head, d_out_head) * (d_in_head ** -0.5))
        self.w_v = nn.Linear(d_in, d_out, bias=False)

    def forward(self, X_q, X_kv, num_splits=2):
        return FinalCrossMoEMultiHeadAttentionFunc.apply(
            X_q, X_kv, self.router_q, self.router_k, self.experts_q, self.experts_k, self.w_v.weight, self.H, num_splits
        )

# =====================================================================
# 4. PyTorch Native Reference 与 基准测试
# =====================================================================
class TorchCrossMoEMultiHeadAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, num_heads: int, num_clusters: int):
        super().__init__()
        self.M = num_clusters
        self.H = num_heads
        self.d_in_head = d_in // num_heads
        self.d_out_head = d_out // num_heads
        self.router_q = nn.Parameter(torch.randn(num_clusters, d_in) * (d_in ** -0.5))
        self.router_k = nn.Parameter(torch.randn(num_clusters, d_in) * (d_in ** -0.5))
        self.experts_q = nn.Parameter(torch.randn(num_clusters, num_heads, self.d_in_head, self.d_out_head) * (self.d_in_head ** -0.5))
        self.experts_k = nn.Parameter(torch.randn(num_clusters, num_heads, self.d_in_head, self.d_out_head) * (self.d_in_head ** -0.5))
        self.w_v = nn.Linear(d_in, d_out, bias=False)

    def forward(self, X_q: torch.Tensor, X_kv: torch.Tensor):
        B, N_q, _ = X_q.shape
        _, N_k, _ = X_kv.shape
        
        out = torch.zeros((B, N_q, self.H, self.d_out_head), device=X_q.device, dtype=X_q.dtype)
        sm_scale = 1.0 / (self.d_out_head ** 0.5)

        c_q = torch.argmax(torch.matmul(X_q, self.router_q.transpose(0, 1)), dim=-1)
        c_k = torch.argmax(torch.matmul(X_kv, self.router_k.transpose(0, 1)), dim=-1)

        q = torch.zeros((B, N_q, self.H, self.d_out_head), device=X_q.device, dtype=X_q.dtype)
        k = torch.zeros((B, N_k, self.H, self.d_out_head), device=X_q.device, dtype=X_q.dtype) 
        
        for m in range(self.M):
            mask_q = (c_q == m) 
            if mask_q.any(): q[mask_q] = torch.einsum('thd,hdo->tho', X_q[mask_q].view(-1, self.H, self.d_in_head), self.experts_q[m])
            mask_k = (c_k == m)
            if mask_k.any(): k[mask_k] = torch.einsum('thd,hdo->tho', X_kv[mask_k].view(-1, self.H, self.d_in_head), self.experts_k[m])

        v = self.w_v(X_kv).view(B, N_k, self.H, self.d_out_head)

        for b in range(B):
            for i in range(N_q):
                cluster = c_q[b, i]
                mask = (c_k[b] == cluster)
                if not mask.any(): continue
                q_i = q[b, i].unsqueeze(1) 
                k_local = k[b, mask].transpose(0, 1) 
                v_local = v[b, mask].transpose(0, 1) 
                scores = torch.bmm(q_i, k_local.transpose(1, 2)) * sm_scale
                attn = torch.softmax(scores, dim=-1)
                out[b, i] = torch.bmm(attn, v_local).squeeze(1) 
                
        return out.view(B, N_q, -1)

def check_grad(name, grad_torch, grad_triton, atol_bwd=1e-2):
    diff = (grad_torch - grad_triton).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    passed = max_err < (atol_bwd * 10 if "W_expert" in name else atol_bwd)
    print(f"[反向] {name:<12} 梯度对齐 : {'✅ 成功' if passed else '❌ 失败'} (Max Err: {max_err:.5f}, Mean Err: {mean_err:.6f})")

def run_full_verification_and_benchmark():
    B = 4       
    N_q = 5
    N_kv = 512
    H = 6
    d_model = 768
    D_IN = d_model
    D_OUT = d_model
    M = 4         
    device = torch.device('cuda')

    print(f"🚀 初始化环境... [B={B}, N_q={N_q}, N_kv={N_kv}, H={H}, D_in={D_IN}, D_out={D_OUT}, M={M}]")
    
    X_q_base = torch.randn((B, N_q, D_IN), dtype=torch.float16, device=device)
    X_kv_base = torch.randn((B, N_kv, D_IN), dtype=torch.float16, device=device)
    dO = torch.randn((B, N_q, D_OUT), dtype=torch.float16, device=device)
    
    X_q_torch = X_q_base.clone().detach().requires_grad_(True)
    X_kv_torch = X_kv_base.clone().detach().requires_grad_(True)
    X_q_triton = X_q_base.clone().detach().requires_grad_(True)
    X_kv_triton = X_kv_base.clone().detach().requires_grad_(True)
         
    
    torch_model = TorchCrossMoEMultiHeadAttention(D_IN, D_OUT, H, M).to(device).half()
    triton_model = FinalCrossMoEMultiHeadAttention(D_IN, D_OUT, H, M).to(device).half()

    with torch.no_grad():
        triton_model.router_q.copy_(torch_model.router_q)
        triton_model.router_k.copy_(torch_model.router_k)
        triton_model.experts_q.copy_(torch_model.experts_q)
        triton_model.experts_k.copy_(torch_model.experts_k)
        triton_model.w_v.weight.copy_(torch_model.w_v.weight)

    print("\n" + "="*50 + "\n🔍 第一环节：精度校验\n" + "="*50)

    out_torch = torch_model(X_q_torch, X_kv_torch)
    out_triton = triton_model(X_q_triton, X_kv_triton)
    fwd_match = torch.allclose(out_torch, out_triton, atol=1e-3, rtol=1e-3)
    print(f"[前向] 输出 Output 对齐状态 : {'✅ 成功' if fwd_match else '❌ 失败'}")

    out_torch.backward(dO)
    out_triton.backward(dO)

    check_grad("dX_q", X_q_torch.grad, X_q_triton.grad)
    check_grad("dX_kv", X_kv_torch.grad, X_kv_triton.grad)
    check_grad("dW_expert_q", torch_model.experts_q.grad, triton_model.experts_q.grad)
    check_grad("dW_expert_k", torch_model.experts_k.grad, triton_model.experts_k.grad)
    check_grad("dW_v", torch_model.w_v.weight.grad, triton_model.w_v.weight.grad)

    print("\n" + "="*50 + "\n⏱️ 第二环节：性能压测\n" + "="*50)
    
    X_q_torch.grad, X_kv_torch.grad, X_q_triton.grad, X_kv_triton.grad = None, None, None, None
    torch_model.zero_grad()
    triton_model.zero_grad()

    out_torch_bench = torch_model(X_q_torch, X_kv_torch)
    out_triton_bench = triton_model(X_q_triton, X_kv_triton)

    for _ in range(3):
        out_torch_bench.backward(dO, retain_graph=True)
        out_triton_bench.backward(dO, retain_graph=True)

    quantiles = [0.5, 0.2, 0.8]
    ms_fwd_pt, _, _ = triton.testing.do_bench(lambda: torch_model(X_q_torch, X_kv_torch), quantiles=quantiles)
    ms_bwd_pt, _, _ = triton.testing.do_bench(lambda: out_torch_bench.backward(dO, retain_graph=True), quantiles=quantiles)
    ms_fwd_tr, _, _ = triton.testing.do_bench(lambda: triton_model(X_q_triton, X_kv_triton), quantiles=quantiles)
    ms_bwd_tr, _, _ = triton.testing.do_bench(lambda: out_triton_bench.backward(dO, retain_graph=True), quantiles=quantiles)

    print(f"{'Metric (指标)':<20} | {'PyTorch Native (ms)':<20} | {'Triton Custom (ms)':<20} | {'Speedup (加速比)'}")
    print("-" * 80)
    print(f"{'Forward Pass':<20} | {ms_fwd_pt:<20.4f} | {ms_fwd_tr:<20.4f} | {ms_fwd_pt/ms_fwd_tr:.2f}x")
    print(f"{'Backward Pass':<20} | {ms_bwd_pt:<20.4f} | {ms_bwd_tr:<20.4f} | {ms_bwd_pt/ms_bwd_tr:.2f}x")
    print(f"{'Total (Fwd + Bwd)':<20} | {ms_fwd_pt+ms_bwd_pt:<20.4f} | {ms_fwd_tr+ms_bwd_tr:<20.4f} | 🔥 {(ms_fwd_pt+ms_bwd_pt)/(ms_fwd_tr+ms_bwd_tr):.2f}x")
    print("=" * 80)

if __name__ == '__main__':
    run_full_verification_and_benchmark()


    