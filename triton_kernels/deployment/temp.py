import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def batched_expert_project_kernel_v4(
    X, Experts, Out, sorted_indices, cu_seqlens,
    stride_xb, stride_xn, stride_xh, stride_xd,
    stride_em, stride_eh, stride_ed_in, stride_ed_out,
    stride_ob, stride_on, stride_oh, stride_od,
    stride_idx_b, stride_idx_n, stride_cu_b,
    actual_d_in, actual_d_out, M: tl.constexpr, H: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_DIN: tl.constexpr, BLOCK_DOUT: tl.constexpr
):
    cluster_id = tl.program_id(0)
    pid_bh = tl.program_id(1)
    n_block_idx = tl.program_id(2)

    b_idx = pid_bh // H
    h_idx = pid_bh % H

    start_n = tl.load(cu_seqlens + b_idx * stride_cu_b + cluster_id)
    end_n = tl.load(cu_seqlens + b_idx * stride_cu_b + cluster_id + 1)

    curr_n = start_n + n_block_idx * BLOCK_N
    if curr_n >= end_n: 
        return 

    offs_n = curr_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < end_n

    idx_ptrs = sorted_indices + b_idx * stride_idx_b + offs_n * stride_idx_n
    src_idx = tl.load(idx_ptrs, mask=mask_n, other=0)

    offs_din = tl.arange(0, BLOCK_DIN)
    offs_dout = tl.arange(0, BLOCK_DOUT)
    mask_din = offs_din < actual_d_in
    mask_dout = offs_dout < actual_d_out

    x_ptrs = X + b_idx * stride_xb + src_idx[:, None] * stride_xn + h_idx * stride_xh + offs_din[None, :] * stride_xd
    x = tl.load(x_ptrs, mask=mask_n[:, None] & mask_din[None, :], other=0.0)

    w_ptrs = Experts + cluster_id * stride_em + h_idx * stride_eh + offs_din[:, None] * stride_ed_in + offs_dout[None, :] * stride_ed_out
    w = tl.load(w_ptrs, mask=mask_din[:, None] & mask_dout[None, :], other=0.0)

    out = tl.dot(x.to(tl.float16), w.to(tl.float16))

    out_ptrs = Out + b_idx * stride_ob + offs_n[:, None] * stride_on + h_idx * stride_oh + offs_dout[None, :] * stride_od
    tl.store(out_ptrs, out, mask=mask_n[:, None] & mask_dout[None, :])


def triton_moe_project_mha_v4(X, Experts, sorted_indices, cu_seqlens, M, H):
    B, N, D_in = X.shape
    _, _, D_in_head, D_out_head = Experts.shape
    X_view = X.view(B, N, H, D_in_head)
    out = torch.empty((B, N, H, D_out_head), dtype=X.dtype, device=X.device)

    BLOCK_DIN = triton.next_power_of_2(D_in_head)
    BLOCK_DOUT = triton.next_power_of_2(D_out_head)
    BLOCK_N = 64

    lens = cu_seqlens[:, 1:] - cu_seqlens[:, :-1]
    max_len = max(lens.max().item(), 1)
    max_blocks = triton.cdiv(max_len, BLOCK_N)

    grid = (M, B * H, max_blocks)
    
    batched_expert_project_kernel_v4[grid](
        X_view, Experts, out, sorted_indices, cu_seqlens,
        X_view.stride(0), X_view.stride(1), X_view.stride(2), X_view.stride(3),
        Experts.stride(0), Experts.stride(1), Experts.stride(2), Experts.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        sorted_indices.stride(0), sorted_indices.stride(1), cu_seqlens.stride(0),
        actual_d_in=D_in_head, actual_d_out=D_out_head, M=M, H=H,
        BLOCK_N=BLOCK_N, BLOCK_DIN=BLOCK_DIN, BLOCK_DOUT=BLOCK_DOUT, num_warps=4
    )
    return out


@triton.jit
def batched_flash_decoding_mha_phase1_v3(
    Q, K, V, mid_acc, mid_m, mid_l,
    q_cu_seqlens, k_cu_seqlens, 
    q_sorted_indices, k_sorted_indices,
    sm_scale,
    stride_qb, stride_qn, stride_qh, stride_qd,
    stride_kb, stride_kn, stride_kh, stride_kd,
    stride_vb, stride_vn, stride_vh, stride_vd,
    stride_mid_acc_b, stride_mid_acc_h, stride_mid_acc_q, stride_mid_acc_s, stride_mid_acc_d,
    stride_mid_m_b, stride_mid_m_h, stride_mid_m_q, stride_mid_m_s,
    stride_mid_l_b, stride_mid_l_h, stride_mid_l_q, stride_mid_l_s,
    stride_q_cu_b, stride_k_cu_b,
    stride_q_idx_b, stride_q_idx_n,
    stride_k_idx_b, stride_k_idx_n,
    actual_d, M: tl.constexpr, H: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    q_block_idx = tl.program_id(0)
    split_idx = tl.program_id(1)
    pid_m_bh = tl.program_id(2)

    cluster_id = pid_m_bh % M
    pid_bh = pid_m_bh // M
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    num_splits = tl.num_programs(1)

    q_start = tl.load(q_cu_seqlens + b_idx * stride_q_cu_b + cluster_id)
    q_end = tl.load(q_cu_seqlens + b_idx * stride_q_cu_b + cluster_id + 1)

    q_base_idx = q_start + q_block_idx * BLOCK_Q
    if q_base_idx >= q_end: return

    offs_q_local = tl.arange(0, BLOCK_Q)
    offs_q = q_base_idx + offs_q_local
    mask_q = offs_q < q_end
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < actual_d

    q_ptrs = Q + b_idx * stride_qb + h_idx * stride_qh + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_q[:, None] & mask_d[None, :], other=0.0)

    acc = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)
    m_i = tl.full([BLOCK_Q], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)

    k_start = tl.load(k_cu_seqlens + b_idx * stride_k_cu_b + cluster_id)
    k_end = tl.load(k_cu_seqlens + b_idx * stride_k_cu_b + cluster_id + 1)
    seq_len = k_end - k_start

    chunk_size = tl.cdiv(seq_len, num_splits)
    chunk_start = k_start + split_idx * chunk_size
    chunk_end = tl.minimum(chunk_start + chunk_size, k_end)

    if chunk_start < chunk_end:
        for start_k in range(chunk_start, chunk_end, BLOCK_K):
            offs_k_local = tl.arange(0, BLOCK_K)
            offs_k = start_k + offs_k_local
            mask_k = offs_k < chunk_end

            k_ptrs = K + b_idx * stride_kb + h_idx * stride_kh + offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=mask_k[:, None] & mask_d[None, :], other=0.0)
            
            qk = tl.dot(q.to(tl.float16), tl.trans(k.to(tl.float16))) * sm_scale
            qk = tl.where(mask_q[:, None] & mask_k[None, :], qk, -float('inf'))

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.math.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)

            alpha = tl.math.exp(m_i - m_ij)
            acc = acc * alpha[:, None]

            k_idx = tl.load(k_sorted_indices + b_idx * stride_k_idx_b + offs_k * stride_k_idx_n, mask=mask_k, other=0)
            v_ptrs = V + b_idx * stride_vb + h_idx * stride_vh + k_idx[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=mask_k[:, None] & mask_d[None, :], other=0.0)

            acc += tl.dot(p.to(tl.float16), v.to(tl.float16))

            m_i = m_ij
            l_i = l_i * alpha + l_ij

    acc_ptrs = mid_acc + b_idx * stride_mid_acc_b + h_idx * stride_mid_acc_h + offs_q[:, None] * stride_mid_acc_q + split_idx * stride_mid_acc_s + offs_d[None, :] * stride_mid_acc_d
    tl.store(acc_ptrs, acc, mask=mask_q[:, None] & mask_d[None, :])
    
    m_ptrs = mid_m + b_idx * stride_mid_m_b + h_idx * stride_mid_m_h + offs_q * stride_mid_m_q + split_idx * stride_mid_m_s
    tl.store(m_ptrs, m_i, mask=mask_q)
    
    l_ptrs = mid_l + b_idx * stride_mid_l_b + h_idx * stride_mid_l_h + offs_q * stride_mid_l_q + split_idx * stride_mid_l_s
    tl.store(l_ptrs, l_i, mask=mask_q)


@triton.jit
def batched_flash_decoding_mha_phase2_v3(
    mid_acc, mid_m, mid_l, Out, LSE,
    q_cu_seqlens, q_sorted_indices,
    stride_mid_acc_b, stride_mid_acc_h, stride_mid_acc_q, stride_mid_acc_s, stride_mid_acc_d,
    stride_mid_m_b, stride_mid_m_h, stride_mid_m_q, stride_mid_m_s,
    stride_mid_l_b, stride_mid_l_h, stride_mid_l_q, stride_mid_l_s,
    stride_ob, stride_on, stride_oh, stride_od,
    stride_lse_b, stride_lse_n, stride_lse_h,
    stride_q_cu_b, stride_q_idx_b, stride_q_idx_n,
    actual_d, M: tl.constexpr, H: tl.constexpr, NUM_SPLITS: tl.constexpr, 
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr
):
    q_block_idx = tl.program_id(0)
    pid_m_bh = tl.program_id(1)
    
    cluster_id = pid_m_bh % M
    pid_bh = pid_m_bh // M
    b_idx = pid_bh // H
    h_idx = pid_bh % H

    q_start = tl.load(q_cu_seqlens + b_idx * stride_q_cu_b + cluster_id)
    q_end = tl.load(q_cu_seqlens + b_idx * stride_q_cu_b + cluster_id + 1)

    q_base_idx = q_start + q_block_idx * BLOCK_Q
    if q_base_idx >= q_end: return

    offs_q = q_base_idx + tl.arange(0, BLOCK_Q)
    mask_q = offs_q < q_end
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < actual_d 
    offs_s = tl.arange(0, NUM_SPLITS)

    m_ptrs = mid_m + b_idx * stride_mid_m_b + h_idx * stride_mid_m_h + offs_q[:, None] * stride_mid_m_q + offs_s[None, :] * stride_mid_m_s
    m_locals = tl.load(m_ptrs, mask=mask_q[:, None], other=-float('inf'))
    m_global = tl.max(m_locals, axis=1) 

    is_empty = m_global == -float('inf')
    m_global_safe = tl.where(is_empty, 0.0, m_global)

    l_ptrs = mid_l + b_idx * stride_mid_l_b + h_idx * stride_mid_l_h + offs_q[:, None] * stride_mid_l_q + offs_s[None, :] * stride_mid_l_s
    l_locals = tl.load(l_ptrs, mask=mask_q[:, None], other=0.0)
    
    weights = tl.math.exp(m_locals - m_global_safe[:, None])
    l_global = tl.sum(l_locals * weights, axis=1) 

    # 🔥 绝对安全阻断：屏蔽 0.0 被作为除数或对数参数
    l_global_safe = tl.where(is_empty, 1.0, l_global)

    q_idx = tl.load(q_sorted_indices + b_idx * stride_q_idx_b + offs_q * stride_q_idx_n, mask=mask_q, other=0)

    lse = tl.where(is_empty, -float('inf'), m_global + tl.math.log(l_global_safe))
    lse_ptrs = LSE + b_idx * stride_lse_b + h_idx * stride_lse_h + q_idx * stride_lse_n
    tl.store(lse_ptrs, lse, mask=mask_q)

    acc_global = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)
    for s in range(NUM_SPLITS):
        w = tl.math.exp(tl.load(mid_m + b_idx * stride_mid_m_b + h_idx * stride_mid_m_h + offs_q * stride_mid_m_q + s * stride_mid_m_s, mask=mask_q, other=-float('inf')) - m_global_safe)
        acc_ptrs = mid_acc + b_idx * stride_mid_acc_b + h_idx * stride_mid_acc_h + offs_q[:, None] * stride_mid_acc_q + s * stride_mid_acc_s + offs_d[None, :] * stride_mid_acc_d
        acc_local = tl.load(acc_ptrs, mask=mask_q[:, None] & mask_d[None, :], other=0.0) 
        acc_global += acc_local * w[:, None]

    # 🔥 用安全的 l_global_safe 执行除法，阻断 0.0 / 0.0 生成 NaN 的通道
    out_val = acc_global / l_global_safe[:, None]
    out_val = tl.where(is_empty[:, None], 0.0, out_val)

    out_ptrs = Out + b_idx * stride_ob + h_idx * stride_oh + q_idx[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, out_val, mask=mask_q[:, None] & mask_d[None, :])


def batched_flash_decoding_mha_v3(Q, K, V, q_cluster_ids_orig, q_sorted_indices, k_sorted_indices, \
                                   k_cu_seqlens, M, num_splits=1):
    B, N_q, H, D = Q.shape
    mid_acc = torch.empty((B, H, N_q, num_splits, D), dtype=torch.float32, device=Q.device)
    mid_m = torch.empty((B, H, N_q, num_splits), dtype=torch.float32, device=Q.device)
    mid_l = torch.empty((B, H, N_q, num_splits), dtype=torch.float32, device=Q.device)
    
    out = torch.zeros_like(Q)
    LSE = torch.zeros((B, N_q, H), dtype=torch.float32, device=Q.device) 
    
    q_cu_seqlens = torch.zeros((B, M + 1), dtype=torch.int32, device=Q.device)
    q_counts = F.one_hot(q_cluster_ids_orig.to(torch.int64), num_classes=M).sum(dim=1)
    q_cu_seqlens[:, 1:] = torch.cumsum(q_counts, dim=1).to(torch.int32)

    sm_scale = 1.0 / (D ** 0.5)
    BLOCK_Q = 64 if N_q > 64 else 32
    BLOCK_K = 64
    BLOCK_D = triton.next_power_of_2(D)

    q_lens = q_cu_seqlens[:, 1:] - q_cu_seqlens[:, :-1]
    max_q_len = max(q_lens.max().item(), 1)
    max_q_blocks = triton.cdiv(max_q_len, BLOCK_Q)

    grid_phase1 = (max_q_blocks, num_splits, M * B * H)
    batched_flash_decoding_mha_phase1_v3[grid_phase1](
        Q, K, V, mid_acc, mid_m, mid_l,
        q_cu_seqlens, k_cu_seqlens, 
        q_sorted_indices, k_sorted_indices,
        sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        mid_acc.stride(0), mid_acc.stride(1), mid_acc.stride(2), mid_acc.stride(3), mid_acc.stride(4),
        mid_m.stride(0), mid_m.stride(1), mid_m.stride(2), mid_m.stride(3),
        mid_l.stride(0), mid_l.stride(1), mid_l.stride(2), mid_l.stride(3),
        q_cu_seqlens.stride(0), k_cu_seqlens.stride(0),
        q_sorted_indices.stride(0), q_sorted_indices.stride(1),
        k_sorted_indices.stride(0), k_sorted_indices.stride(1),
        actual_d=D, M=M, H=H, BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D, 
        num_warps=2, num_stages=2
    )

    grid_phase2 = (max_q_blocks, M * B * H)
    batched_flash_decoding_mha_phase2_v3[grid_phase2](
        mid_acc, mid_m, mid_l, out, LSE,
        q_cu_seqlens, q_sorted_indices,
        mid_acc.stride(0), mid_acc.stride(1), mid_acc.stride(2), mid_acc.stride(3), mid_acc.stride(4),
        mid_m.stride(0), mid_m.stride(1), mid_m.stride(2), mid_m.stride(3),
        mid_l.stride(0), mid_l.stride(1), mid_l.stride(2), mid_l.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        q_cu_seqlens.stride(0), q_sorted_indices.stride(0), q_sorted_indices.stride(1),
        actual_d=D, M=M, H=H, NUM_SPLITS=num_splits, BLOCK_Q=BLOCK_Q, BLOCK_D=BLOCK_D, 
        num_warps=2
    )
    
    return out, LSE


@triton.jit
def bwd_kernel_dq_mha_v4(
    Q_sorted, K_sorted, V_native, dO_native, LSE, Delta_native, dQ_sorted,
    q_cu_seqlens, k_cu_seqlens,
    q_sorted_indices, k_sorted_indices,
    sm_scale,
    stride_qb, stride_qn, stride_qh, stride_qd,
    stride_kb, stride_kn, stride_kh, stride_kd,
    stride_vb, stride_vn, stride_vh, stride_vd,
    stride_dob, stride_don, stride_doh, stride_dod,
    stride_lse_b, stride_lse_n, stride_lse_h,
    stride_delta_b, stride_delta_n, stride_delta_h,
    stride_q_cu_b, stride_k_cu_b,
    stride_q_idx_b, stride_q_idx_n,
    stride_k_idx_b, stride_k_idx_n,
    actual_d: tl.constexpr, M: tl.constexpr, H: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    cluster_id = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    q_block_idx = tl.program_id(2)

    q_start = tl.load(q_cu_seqlens + b_idx * stride_q_cu_b + cluster_id)
    q_end = tl.load(q_cu_seqlens + b_idx * stride_q_cu_b + cluster_id + 1)
    curr_q = q_start + q_block_idx * BLOCK_Q
    if curr_q >= q_end: return

    offs_q = curr_q + tl.arange(0, BLOCK_Q)
    q_mask = offs_q < q_end
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < actual_d

    q_idx = tl.load(q_sorted_indices + b_idx * stride_q_idx_b + offs_q * stride_q_idx_n, mask=q_mask, other=0)

    q_ptrs = Q_sorted + b_idx * stride_qb + h_idx * stride_qh + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=q_mask[:, None] & mask_d[None, :], other=0.0)

    do_ptrs = dO_native + b_idx * stride_dob + h_idx * stride_doh + q_idx[:, None] * stride_don + offs_d[None, :] * stride_dod
    do = tl.load(do_ptrs, mask=q_mask[:, None] & mask_d[None, :], other=0.0)
    lse = tl.load(LSE + b_idx * stride_lse_b + h_idx * stride_lse_h + q_idx * stride_lse_n, mask=q_mask, other=0.0)
    delta = tl.load(Delta_native + b_idx * stride_delta_b + h_idx * stride_delta_h + q_idx * stride_delta_n, mask=q_mask, other=0.0)

    dq_acc = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)

    k_start = tl.load(k_cu_seqlens + b_idx * stride_k_cu_b + cluster_id)
    k_end = tl.load(k_cu_seqlens + b_idx * stride_k_cu_b + cluster_id + 1)

    if k_start < k_end:
        for start_k in range(k_start, k_end, BLOCK_K):
            offs_k = start_k + tl.arange(0, BLOCK_K)
            k_mask = offs_k < k_end

            k_idx = tl.load(k_sorted_indices + b_idx * stride_k_idx_b + offs_k * stride_k_idx_n, mask=k_mask, other=0)

            k_ptrs = K_sorted + b_idx * stride_kb + h_idx * stride_kh + offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=k_mask[:, None] & mask_d[None, :], other=0.0)
            
            v_ptrs = V_native + b_idx * stride_vb + h_idx * stride_vh + k_idx[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=k_mask[:, None] & mask_d[None, :], other=0.0)

            qk = tl.dot(q, tl.trans(k)) * sm_scale
            p = tl.math.exp(qk - lse[:, None])
            p = tl.where(q_mask[:, None] & k_mask[None, :], p, 0.0)

            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - delta[:, None]) * sm_scale

            dq_acc += tl.dot(ds.to(tl.float16), k.to(tl.float16))

    dq_ptrs = dQ_sorted + b_idx * stride_qb + h_idx * stride_qh + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd
    tl.store(dq_ptrs, dq_acc.to(q.dtype), mask=q_mask[:, None] & mask_d[None, :])


@triton.jit
def bwd_kernel_dk_dv_mha_v4(
    Q_sorted, K_sorted, V_native, dO_native, LSE, Delta_native, dK_sorted, dV_native,
    q_cu_seqlens, k_cu_seqlens,
    q_sorted_indices, k_sorted_indices,
    sm_scale,
    stride_qb, stride_qn, stride_qh, stride_qd,
    stride_kb, stride_kn, stride_kh, stride_kd,
    stride_vb, stride_vn, stride_vh, stride_vd,
    stride_dob, stride_don, stride_doh, stride_dod,
    stride_lse_b, stride_lse_n, stride_lse_h,
    stride_delta_b, stride_delta_n, stride_delta_h,
    stride_q_cu_b, stride_k_cu_b,
    stride_q_idx_b, stride_q_idx_n,
    stride_k_idx_b, stride_k_idx_n,
    actual_d: tl.constexpr, M: tl.constexpr, H: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    cluster_id = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    k_block_idx = tl.program_id(2)

    k_start = tl.load(k_cu_seqlens + b_idx * stride_k_cu_b + cluster_id)
    k_end = tl.load(k_cu_seqlens + b_idx * stride_k_cu_b + cluster_id + 1)
    curr_k = k_start + k_block_idx * BLOCK_K
    if curr_k >= k_end: return

    offs_k = curr_k + tl.arange(0, BLOCK_K)
    k_mask = offs_k < k_end
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < actual_d

    k_idx = tl.load(k_sorted_indices + b_idx * stride_k_idx_b + offs_k * stride_k_idx_n, mask=k_mask, other=0)

    k_ptrs = K_sorted + b_idx * stride_kb + h_idx * stride_kh + offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd
    k = tl.load(k_ptrs, mask=k_mask[:, None] & mask_d[None, :], other=0.0)

    v_ptrs = V_native + b_idx * stride_vb + h_idx * stride_vh + k_idx[:, None] * stride_vn + offs_d[None, :] * stride_vd
    v = tl.load(v_ptrs, mask=k_mask[:, None] & mask_d[None, :], other=0.0)

    dk_acc = tl.zeros([BLOCK_K, BLOCK_D], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_K, BLOCK_D], dtype=tl.float32)

    q_start = tl.load(q_cu_seqlens + b_idx * stride_q_cu_b + cluster_id)
    q_end = tl.load(q_cu_seqlens + b_idx * stride_q_cu_b + cluster_id + 1)

    if q_start < q_end:
        for start_q in range(q_start, q_end, BLOCK_Q):
            offs_q = start_q + tl.arange(0, BLOCK_Q)
            q_mask = offs_q < q_end

            q_idx = tl.load(q_sorted_indices + b_idx * stride_q_idx_b + offs_q * stride_q_idx_n, mask=q_mask, other=0)

            q_ptrs = Q_sorted + b_idx * stride_qb + h_idx * stride_qh + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd
            q = tl.load(q_ptrs, mask=q_mask[:, None] & mask_d[None, :], other=0.0)

            do_ptrs = dO_native + b_idx * stride_dob + h_idx * stride_doh + q_idx[:, None] * stride_don + offs_d[None, :] * stride_dod
            do = tl.load(do_ptrs, mask=q_mask[:, None] & mask_d[None, :], other=0.0)

            lse = tl.load(LSE + b_idx * stride_lse_b + h_idx * stride_lse_h + q_idx * stride_lse_n, mask=q_mask, other=0.0)
            delta = tl.load(Delta_native + b_idx * stride_delta_b + h_idx * stride_delta_h + q_idx * stride_delta_n, mask=q_mask, other=0.0)

            qk_t = tl.dot(k, tl.trans(q)) * sm_scale
            pt = tl.math.exp(qk_t - lse[None, :])
            pt = tl.where(k_mask[:, None] & q_mask[None, :], pt, 0.0)

            dv_acc += tl.dot(pt.to(tl.float16), do.to(tl.float16))

            dp_t = tl.dot(v, tl.trans(do))
            ds_t = pt * (dp_t - delta[None, :]) * sm_scale

            dk_acc += tl.dot(ds_t.to(tl.float16), q.to(tl.float16))

    dk_ptrs = dK_sorted + b_idx * stride_kb + h_idx * stride_kh + offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd
    tl.store(dk_ptrs, dk_acc.to(k.dtype), mask=k_mask[:, None] & mask_d[None, :])

    dv_ptrs = dV_native + b_idx * stride_vb + h_idx * stride_vh + k_idx[:, None] * stride_vn + offs_d[None, :] * stride_vd
    tl.store(dv_ptrs, dv_acc.to(v.dtype), mask=k_mask[:, None] & mask_d[None, :])


@triton.jit
def bwd_expert_project_kernel_v4(
    X_native, dX_out_sorted, Experts,
    dX_native_out, dExperts_out,
    sorted_indices, cu_seqlens,
    stride_xb, stride_xn, stride_xh, stride_xdin,
    stride_dxob, stride_dxon, stride_dxoh, stride_dxodout,
    stride_em, stride_eh, stride_edin, stride_edout,
    stride_idxb, stride_idxn, stride_cub,
    actual_din, actual_dout, M: tl.constexpr, H: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_DIN: tl.constexpr, BLOCK_DOUT: tl.constexpr
):
    # 3D Grid 映射
    pid_m = tl.program_id(0)      # Expert ID
    pid_bh = tl.program_id(1)     # Batch * Head
    pid_n = tl.program_id(2)      # Sequence Block ID

    b_idx = pid_bh // H
    h_idx = pid_bh % H

    # 1. 动态确定当前 Expert 在当前 Batch 下负责的 Sequence 边界
    start_n = tl.load(cu_seqlens + b_idx * stride_cub + pid_m)
    end_n = tl.load(cu_seqlens + b_idx * stride_cub + pid_m + 1)
    
    curr_n = start_n + pid_n * BLOCK_N
    
    # [提前阻断] 如果当前分配的 Sequence 块超出了该 Expert 的实际 Token 数量，直接退出释放线程
    if curr_n >= end_n: 
        return 

    offs_n = curr_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < end_n

    offs_din = tl.arange(0, BLOCK_DIN)
    offs_dout = tl.arange(0, BLOCK_DOUT)
    mask_din = offs_din < actual_din
    mask_dout = offs_dout < actual_dout

    # 2. 加载路由后的原始 Token 索引
    idx_ptrs = sorted_indices + b_idx * stride_idxb + offs_n * stride_idxn
    src_idx = tl.load(idx_ptrs, mask=mask_n, other=0)

    # 3. 加载对应的输入特征 X 与输出梯度 dX_out
    x_ptrs = X_native + b_idx * stride_xb + src_idx[:, None] * stride_xn + h_idx * stride_xh + offs_din[None, :] * stride_xdin
    x = tl.load(x_ptrs, mask=mask_n[:, None] & mask_din[None, :], other=0.0)

    dxo_ptrs = dX_out_sorted + b_idx * stride_dxob + offs_n[:, None] * stride_dxon + h_idx * stride_dxoh + offs_dout[None, :] * stride_dxodout
    dxo = tl.load(dxo_ptrs, mask=mask_n[:, None] & mask_dout[None, :], other=0.0)

    # 加载 Expert 权重
    w_ptrs = Experts + pid_m * stride_em + h_idx * stride_eh + offs_din[:, None] * stride_edin + offs_dout[None, :] * stride_edout
    w = tl.load(w_ptrs, mask=mask_din[:, None] & mask_dout[None, :], other=0.0)

    # 4. 计算并写回 dX_in (对 X 的梯度)
    dxin = tl.dot(dxo.to(tl.float16), tl.trans(w.to(tl.float16)))
    dxin_ptrs = dX_native_out + b_idx * stride_xb + src_idx[:, None] * stride_xn + h_idx * stride_xh + offs_din[None, :] * stride_xdin
    
    # 保持安全的累加逻辑，避免 Top-K 场景下的潜在复写冲突
    existing_dx = tl.load(dxin_ptrs, mask=mask_n[:, None] & mask_din[None, :], other=0.0)
    tl.store(dxin_ptrs, existing_dx + dxin.to(x.dtype), mask=mask_n[:, None] & mask_din[None, :])

    # 5. 计算并累加 dW_local (对 Expert 权重的梯度)
    dw_local = tl.dot(tl.trans(x.to(tl.float16)), dxo.to(tl.float16))
    dw_ptrs = dExperts_out + pid_m * stride_em + h_idx * stride_eh + offs_din[:, None] * stride_edin + offs_dout[None, :] * stride_edout
    
    # 使用 atomic_add 跨 Block 汇聚所有 Batch 和 N 维度的局部权重梯度
    tl.atomic_add(dw_ptrs, dw_local.to(w.dtype), mask=mask_din[:, None] & mask_dout[None, :])


def triton_moe_project_backward_v4(X_native, dX_out_sorted, Experts, dX_native_out, sorted_indices, cu_seqlens, M, H):
    B, N, H_dim, D_in_head = X_native.shape
    _, _, D_in_head, D_out_head = Experts.shape
    
    # 🚨 极其关键：由于我们采用了高度拆分的 Grid 进行局部 dW 计算，
    # 必须初始化为 0 以支撑跨 Thread Blocks 的 atomic_add 累加。
    dExperts = torch.zeros_like(Experts)
    
    BLOCK_DIN = triton.next_power_of_2(D_in_head)
    BLOCK_DOUT = triton.next_power_of_2(D_out_head)
    BLOCK_N = 32

    # 找出各个 Expert 处理过的最长 Token 段，作为第三维度的 Launch 参数
    lens = cu_seqlens[:, 1:] - cu_seqlens[:, :-1]
    max_len = max(lens.max().item(), 1)
    max_blocks = triton.cdiv(max_len, BLOCK_N)

    # 🚀 释放算力：从原先的 (M, H) 扩张为 3D 网格 (M, B * H, Seq_Blocks)
    grid = (M, B * H, max_blocks)
    
    bwd_expert_project_kernel_v4[grid](
        X_native, dX_out_sorted, Experts,
        dX_native_out, dExperts,
        sorted_indices, cu_seqlens,
        X_native.stride(0), X_native.stride(1), X_native.stride(2), X_native.stride(3),
        dX_out_sorted.stride(0), dX_out_sorted.stride(1), dX_out_sorted.stride(2), dX_out_sorted.stride(3),
        Experts.stride(0), Experts.stride(1), Experts.stride(2), Experts.stride(3),
        sorted_indices.stride(0), sorted_indices.stride(1), cu_seqlens.stride(0),
        actual_din=D_in_head, actual_dout=D_out_head, M=M, H=H,
        BLOCK_N=BLOCK_N, BLOCK_DIN=BLOCK_DIN, BLOCK_DOUT=BLOCK_DOUT, 
        num_warps=4,     # 反向通常占用寄存器较多，保持 4 个 warps
        num_stages=2     # 增加 Stage 来覆盖访存延迟
    )
    return dExperts


def triton_attention_backward_v4(
    Q_sorted, K_sorted, V_native, Out_native, dO_native,
    LSE, q_cu_seqlens, k_cu_seqlens,
    q_sorted_indices, k_sorted_indices, M, H
):
    B, N_q, H_dim, D_head = Q_sorted.shape
    _, N_kv, _, _ = V_native.shape

    dQ_sorted = torch.zeros_like(Q_sorted)
    dK_sorted = torch.zeros_like(K_sorted)
    dV_native = torch.zeros_like(V_native)
    sm_scale = 1.0 / (D_head ** 0.5)

    Delta_native = (dO_native * Out_native).sum(dim=-1).contiguous()

    BLOCK_Q = 32
    BLOCK_K = 32
    BLOCK_D = triton.next_power_of_2(D_head)

    q_lens = q_cu_seqlens[:, 1:] - q_cu_seqlens[:, :-1]
    k_lens = k_cu_seqlens[:, 1:] - k_cu_seqlens[:, :-1]
    max_q_len = max(q_lens.max().item(), 1)
    max_k_len = max(k_lens.max().item(), 1)
    
    grid_dq = (M, B * H, triton.cdiv(max_q_len, BLOCK_Q))
    bwd_kernel_dq_mha_v4[grid_dq](
        Q_sorted, K_sorted, V_native, dO_native, LSE, Delta_native, dQ_sorted,
        q_cu_seqlens, k_cu_seqlens,
        q_sorted_indices, k_sorted_indices, sm_scale,
        Q_sorted.stride(0), Q_sorted.stride(1), Q_sorted.stride(2), Q_sorted.stride(3),
        K_sorted.stride(0), K_sorted.stride(1), K_sorted.stride(2), K_sorted.stride(3),
        V_native.stride(0), V_native.stride(1), V_native.stride(2), V_native.stride(3),
        dO_native.stride(0), dO_native.stride(1), dO_native.stride(2), dO_native.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        Delta_native.stride(0), Delta_native.stride(1), Delta_native.stride(2),
        q_cu_seqlens.stride(0), k_cu_seqlens.stride(0),
        q_sorted_indices.stride(0), q_sorted_indices.stride(1),
        k_sorted_indices.stride(0), k_sorted_indices.stride(1),
        actual_d=D_head, M=M, H=H, BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D, num_stages=2
    )

    grid_dk_dv = (M, B * H, triton.cdiv(max_k_len, BLOCK_K))
    bwd_kernel_dk_dv_mha_v4[grid_dk_dv](
        Q_sorted, K_sorted, V_native, dO_native, LSE, Delta_native, dK_sorted, dV_native,
        q_cu_seqlens, k_cu_seqlens,
        q_sorted_indices, k_sorted_indices, sm_scale,
        Q_sorted.stride(0), Q_sorted.stride(1), Q_sorted.stride(2), Q_sorted.stride(3),
        K_sorted.stride(0), K_sorted.stride(1), K_sorted.stride(2), K_sorted.stride(3),
        V_native.stride(0), V_native.stride(1), V_native.stride(2), V_native.stride(3),
        dO_native.stride(0), dO_native.stride(1), dO_native.stride(2), dO_native.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        Delta_native.stride(0), Delta_native.stride(1), Delta_native.stride(2),
        q_cu_seqlens.stride(0), k_cu_seqlens.stride(0),
        q_sorted_indices.stride(0), q_sorted_indices.stride(1),
        k_sorted_indices.stride(0), k_sorted_indices.stride(1),
        actual_d=D_head, M=M, H=H, BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D, num_stages=2
    )

    return dQ_sorted, dK_sorted, dV_native


def compute_routing_and_cu_seqlens(X, Router_weight, M):
    B, N, _ = X.shape
    logits = torch.matmul(X, Router_weight.transpose(0, 1))
    cluster_ids_orig = torch.argmax(logits, dim=-1).to(torch.int32)
    sorted_indices = torch.argsort(cluster_ids_orig, dim=-1).to(torch.int32)
    cu_seqlens = torch.zeros((B, M + 1), dtype=torch.int32, device=X.device)
    counts = F.one_hot(cluster_ids_orig.to(torch.int64), num_classes=M).sum(dim=1) 
    cu_seqlens[:, 1:] = torch.cumsum(counts, dim=1).to(torch.int32)
    return cluster_ids_orig, sorted_indices, cu_seqlens


class FinalCrossMoEMultiHeadAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X_q, X_kv, router_q, router_k, experts_q, experts_k, w_v_weight, H=8, num_splits=1):
        B, N_q, _ = X_q.shape
        _, N_kv, _ = X_kv.shape
        D_out_head = experts_q.shape[-1]
        M = experts_q.shape[0]

        V = F.linear(X_kv, w_v_weight).view(B, N_kv, H, D_out_head)

        q_ids_orig, q_sorted_idx, q_cu_seqlens = compute_routing_and_cu_seqlens(X_q, router_q, M)
        k_ids_orig, k_sorted_idx, k_cu_seqlens = compute_routing_and_cu_seqlens(X_kv, router_k, M)
        
        q_sorted = triton_moe_project_mha_v4(X_q, experts_q, q_sorted_idx, q_cu_seqlens, M, H)
        k_sorted = triton_moe_project_mha_v4(X_kv, experts_k, k_sorted_idx, k_cu_seqlens, M, H)
        
        out, LSE = batched_flash_decoding_mha_v3(
            q_sorted, k_sorted, V, 
            q_ids_orig, q_sorted_idx, k_sorted_idx, 
            k_cu_seqlens, M, num_splits=num_splits
        )

        ctx.save_for_backward(
            X_q, X_kv, experts_q, experts_k, w_v_weight, 
            q_ids_orig, k_ids_orig,
            q_sorted_idx, k_sorted_idx, 
            q_cu_seqlens, k_cu_seqlens,
            q_sorted, k_sorted, V, out, LSE
        )
        ctx.M = M
        ctx.H = H
        ctx.D_in_head = experts_q.shape[-2]
        
        return out.view(B, N_q, -1)

    @staticmethod
    def backward(ctx, grad_output):
        X_q, X_kv, experts_q, experts_k, w_v_weight, \
        q_cluster_ids_orig, k_cluster_ids_orig, \
        q_sorted_indices, k_sorted_indices, \
        q_cu_seqlens, k_cu_seqlens, \
        q_sorted, k_sorted, V_native, out_native, LSE = ctx.saved_tensors
        
        grad_output = grad_output.to(X_q.dtype)
        B, N_q, _ = X_q.shape
        _, N_kv, _ = X_kv.shape
        M = ctx.M
        H = ctx.H
        D_in_head = ctx.D_in_head
        D_out_head = experts_q.shape[-1]
        
        dO_native = grad_output.contiguous().view(B, N_q, H, D_out_head)
        out_native = out_native.contiguous()

        dQ_sorted, dK_sorted, dV_native = triton_attention_backward_v4(
            q_sorted, k_sorted, V_native, out_native, dO_native, 
            LSE, q_cu_seqlens, k_cu_seqlens, 
            q_sorted_indices, k_sorted_indices, M, H
        )

        grad_X_q = torch.zeros_like(X_q)
        grad_X_kv = torch.zeros_like(X_kv)
        
        # dv_flat = dV_native.view(-1, H * D_out_head)
        # X_kv_flat = X_kv.view(-1, H * D_in_head)
        # grad_w_v_weight = torch.matmul(dv_flat.t(), X_kv_flat)
        # grad_X_kv += torch.matmul(dv_flat, w_v_weight).view(B, N_kv, -1)

        dv_flat = dV_native.view(-1, H * D_out_head).to(torch.float32)
        X_kv_flat = X_kv.view(-1, H * D_in_head).to(torch.float32)
        w_v_weight_fp32 = w_v_weight.to(torch.float32)
        
        grad_w_v_weight = torch.matmul(dv_flat.t(), X_kv_flat).to(X_kv.dtype)
        grad_X_kv += torch.matmul(dv_flat, w_v_weight_fp32).to(X_kv.dtype).view(B, N_kv, -1)

        grad_experts_q = triton_moe_project_backward_v4(
            X_q.view(B, N_q, H, D_in_head), dQ_sorted, experts_q, 
            grad_X_q.view(B, N_q, H, D_in_head), 
            q_sorted_indices, q_cu_seqlens, M, H
        )

        grad_experts_k = triton_moe_project_backward_v4(
            X_kv.view(B, N_kv, H, D_in_head), dK_sorted, experts_k, 
            grad_X_kv.view(B, N_kv, H, D_in_head),
            k_sorted_indices, k_cu_seqlens, M, H
        )

        grad_router_q = None 
        grad_router_k = None

        return grad_X_q, grad_X_kv, grad_router_q, grad_router_k, grad_experts_q, grad_experts_k, \
              grad_w_v_weight, None, None


class FinalCrossMoEMultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, num_clusters):
        super().__init__()
        self.H = num_heads
        d_in_head = d_in // num_heads
        d_out_head = d_out // num_heads
        self.M = num_clusters
        self.router_q = nn.Parameter(torch.randn(num_clusters, d_in) * (d_in ** -0.5))
        self.router_k = nn.Parameter(torch.randn(num_clusters, d_in) * (d_in ** -0.5))
        self.experts_q = nn.Parameter(torch.randn(num_clusters, num_heads, d_in_head, d_out_head) * (d_in_head ** -0.5))
        self.experts_k = nn.Parameter(torch.randn(num_clusters, num_heads, d_in_head, d_out_head) * (d_in_head ** -0.5))
        self.w_v = nn.Linear(d_in, d_out, bias=False)

    def forward(self, X_q, X_kv, num_splits=1):
        return FinalCrossMoEMultiHeadAttentionFunc.apply(
            X_q, X_kv, self.router_q, self.router_k, self.experts_q, self.experts_k, self.w_v.weight, \
                self.H, num_splits
        )


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

        q = q.transpose(1, 2)  
        k = k.transpose(1, 2)  
        v = v.transpose(1, 2)  

        mask = (c_q.unsqueeze(-1) == c_k.unsqueeze(1)).unsqueeze(1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
        
        invalid_rows = ~mask.any(dim=-1, keepdim=True)
        safe_mask = mask.clone()
        safe_mask = safe_mask.masked_fill(invalid_rows.expand_as(safe_mask), True)

        scores = scores.masked_fill(~safe_mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = attn.masked_fill(invalid_rows, 0.0)
        out = torch.matmul(attn, v)

        return out.transpose(1, 2).contiguous().view(B, N_q, -1)


def check_grad(name, grad_torch, grad_triton, atol_bwd=1e-2):
    diff = (grad_torch - grad_triton).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    passed = max_err < (atol_bwd * 10 if "W_expert" in name else atol_bwd)
    print(f"[反向] {name:<12} 梯度对齐 : {'✅ 成功' if passed else '❌ 失败'} (Max Err: {max_err:.5f}, Mean Err: {mean_err:.6f})")


def run_full_verification_and_benchmark():
    B = 16       
    N_q = 96 
    N_kv = 512 * 16
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