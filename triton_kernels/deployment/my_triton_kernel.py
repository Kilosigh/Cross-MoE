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

    # 1. 精准定位当前 Cluster 的边界
    start_n = tl.load(cu_seqlens + b_idx * stride_cu_b + cluster_id)
    end_n = tl.load(cu_seqlens + b_idx * stride_cu_b + cluster_id + 1)

    curr_n = start_n + n_block_idx * BLOCK_N
    if curr_n >= end_n: 
        return # 空闲 Block 瞬间退核

    offs_n = curr_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < end_n

    # 2. 间接寻址：获取真实的 Token 物理索引
    idx_ptrs = sorted_indices + b_idx * stride_idx_b + offs_n * stride_idx_n
    src_idx = tl.load(idx_ptrs, mask=mask_n, other=0)

    offs_din = tl.arange(0, BLOCK_DIN)
    offs_dout = tl.arange(0, BLOCK_DOUT)
    mask_din = offs_din < actual_d_in
    mask_dout = offs_dout < actual_d_out

    # 3. 2D 矩阵加载 (X 和 Expert 权重)
    x_ptrs = X + b_idx * stride_xb + src_idx[:, None] * stride_xn + h_idx * stride_xh + offs_din[None, :] * stride_xd
    x = tl.load(x_ptrs, mask=mask_n[:, None] & mask_din[None, :], other=0.0)

    w_ptrs = Experts + cluster_id * stride_em + h_idx * stride_eh + offs_din[:, None] * stride_ed_in + offs_dout[None, :] * stride_ed_out
    w = tl.load(w_ptrs, mask=mask_din[:, None] & mask_dout[None, :], other=0.0)

    # 4. 🔥 召唤 Tensor Core 轰炸 (Matrix Multiplication)
    out = tl.dot(x.to(tl.float16), w.to(tl.float16))

    # 5. 存入连续的 Out 张量中 (为后续的 Attention 铺平道路)
    out_ptrs = Out + b_idx * stride_ob + offs_n[:, None] * stride_on + h_idx * stride_oh + offs_dout[None, :] * stride_od
    tl.store(out_ptrs, out, mask=mask_n[:, None] & mask_dout[None, :])


def triton_moe_project_mha_v4(X, Experts, sorted_indices, cu_seqlens, M, H):
    B, N, D_in = X.shape
    _, _, D_in_head, D_out_head = Experts.shape
    X_view = X.view(B, N, H, D_in_head)
    out = torch.empty((B, N, H, D_out_head), dtype=X.dtype, device=X.device)

    BLOCK_DIN = triton.next_power_of_2(D_in_head)
    BLOCK_DOUT = triton.next_power_of_2(D_out_head)
    BLOCK_N = 64  # 每 64 个 Token 打包成一个 Block

    # 动态计算最长的 Cluster，决定 Z 轴的网格深度
    lens = cu_seqlens[:, 1:] - cu_seqlens[:, :-1]
    max_len = max(lens.max().item(), 1)
    max_blocks = triton.cdiv(max_len, BLOCK_N)

    # Grid 数量从 78万 暴降至 ~5万！
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

    counts = F.one_hot(cluster_ids.to(torch.int64), num_classes=M).sum(dim=1) # 假设 cluster_ids 是 2D (B, N)
    cu_seqlens[:, 1:] = torch.cumsum(counts, dim=1)
    
    # for b in range(B):
    #     cu_seqlens[b, 1:] = torch.cumsum(torch.bincount(cluster_ids[b], minlength=M), dim=0)

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
        BLOCK_DIN=BLOCK_DIN, BLOCK_DOUT=BLOCK_DOUT, num_warps=2, num_stages=2
    )
    return out, cluster_ids, cu_seqlens, sorted_indices


@triton.jit
def batched_flash_decoding_mha_phase1_v3(
    Q, K, V, mid_acc, mid_m, mid_l,
    q_cu_seqlens, k_cu_seqlens, 
    q_sorted_indices, k_sorted_indices, # 传入原始索引
    sm_scale,
    stride_qb, stride_qn, stride_qh, stride_qd,
    stride_kb, stride_kn, stride_kh, stride_kd,
    stride_vb, stride_vn, stride_vh, stride_vd,
    stride_mid_acc_b, stride_mid_acc_h, stride_mid_acc_q, stride_mid_acc_s, stride_mid_acc_d,
    stride_mid_m_b, stride_mid_m_h, stride_mid_m_q, stride_mid_m_s,
    stride_mid_l_b, stride_mid_l_h, stride_mid_l_q, stride_mid_l_s,
    stride_q_cu_b, stride_k_cu_b,
    stride_q_idx_b, stride_q_idx_n, # 索引的 stride
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
    if q_base_idx >= q_end:
        return

    offs_q_local = tl.arange(0, BLOCK_Q)
    offs_q = q_base_idx + offs_q_local
    mask_q = offs_q < q_end
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < actual_d


    q_ptrs = Q + b_idx * stride_qb + h_idx * stride_qh + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_q[:, None] & mask_d[None, :], other=0.0) * sm_scale

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

            # -------- 修正 K 的读取 --------
            # 🔥 K 也是排序好的，直接用连续内存坐标 offs_k 读取！
            k_ptrs = K + b_idx * stride_kb + h_idx * stride_kh + offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=mask_k[:, None] & mask_d[None, :], other=0.0)

            qk = tl.dot(q.to(tl.float16), tl.trans(k.to(tl.float16)))
            qk = tl.where(mask_q[:, None] & mask_k[None, :], qk, -float('inf'))

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.math.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)

            alpha = tl.math.exp(m_i - m_ij)
            acc = acc * alpha[:, None]

            # -------- 修正 V 的读取 --------
            # 🔥 V 是 Zero-Copy 的原生态张量，全场只有它必须使用 k_idx 间接寻址！
            k_idx = tl.load(k_sorted_indices + b_idx * stride_k_idx_b + offs_k * stride_k_idx_n, mask=mask_k, other=0)
            
            v_ptrs = V + b_idx * stride_vb + h_idx * stride_vh + k_idx[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=mask_k[:, None] & mask_d[None, :], other=0.0)

            acc += tl.dot(p.to(tl.float16), v.to(tl.float16))

    # 中间变量（mid_acc等）仍然按照逻辑排序存储，方便 Phase 2 规约
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

    # 🔥 获取真实 Q 索引，准备将结果直接写回原位，彻底消灭 Unsort！
    q_idx = tl.load(q_sorted_indices + b_idx * stride_q_idx_b + offs_q * stride_q_idx_n, mask=mask_q, other=0)

    lse = tl.where(is_empty, -float('inf'), m_global + tl.math.log(l_global))
    lse_ptrs = LSE + b_idx * stride_lse_b + h_idx * stride_lse_h + q_idx * stride_lse_n
    tl.store(lse_ptrs, lse, mask=mask_q)

    acc_global = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)
    for s in range(NUM_SPLITS):
        w = tl.math.exp(tl.load(mid_m + b_idx * stride_mid_m_b + h_idx * stride_mid_m_h + offs_q * stride_mid_m_q + s * stride_mid_m_s, mask=mask_q, other=-float('inf')) - m_global_safe)
        acc_ptrs = mid_acc + b_idx * stride_mid_acc_b + h_idx * stride_mid_acc_h + offs_q[:, None] * stride_mid_acc_q + s * stride_mid_acc_s + offs_d[None, :] * stride_mid_acc_d
        acc_local = tl.load(acc_ptrs, mask=mask_q[:, None] & mask_d[None, :], other=0.0) 
        acc_global += acc_local * w[:, None]

    # 直接使用 q_idx 写回到最终 Output 的物理位置
    # out_ptrs = Out + b_idx * stride_ob + h_idx * stride_oh + q_idx[:, None] * stride_on + offs_d[None, :] * stride_od
    # tl.store(out_ptrs, acc_global / l_global[:, None], mask=mask_q[:, None] & mask_d[None, :])
    out_val = acc_global / l_global[:, None]
    out_val = tl.where(is_empty[:, None], 0.0, out_val)

    out_ptrs = Out + b_idx * stride_ob + h_idx * stride_oh + q_idx[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, out_val, mask=mask_q[:, None] & mask_d[None, :])


def batched_flash_decoding_mha_v3(Q, K, V, q_cluster_ids_orig, q_sorted_indices, k_sorted_indices, \
                                   k_cu_seqlens, M, num_splits=1):
    B, N_q, H, D = Q.shape
    mid_acc = torch.empty((B, H, N_q, num_splits, D), dtype=torch.float32, device=Q.device)
    mid_m = torch.empty((B, H, N_q, num_splits), dtype=torch.float32, device=Q.device)
    mid_l = torch.empty((B, H, N_q, num_splits), dtype=torch.float32, device=Q.device)
    
    # 原位输出张量
    out = torch.zeros_like(Q)
    LSE = torch.zeros((B, N_q, H), dtype=torch.float32, device=Q.device) 
    
    # 计算 q_cu_seqlens（继续使用 F.one_hot 因为你在 T4 上测出来更快）
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

    # 如果关闭了 Split-K (num_splits=1)，Phase 2 仍然需要跑，因为它负责把 acc 最终除以 l_global 并利用 q_idx 写回 Out
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


# =====================================================================
# 2. 反向传播 (Backward Kernels Multi-Head 改造)
# =====================================================================
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

    # 定位当前 Q Block 的边界
    q_start = tl.load(q_cu_seqlens + b_idx * stride_q_cu_b + cluster_id)
    q_end = tl.load(q_cu_seqlens + b_idx * stride_q_cu_b + cluster_id + 1)
    curr_q = q_start + q_block_idx * BLOCK_Q
    if curr_q >= q_end: return

    offs_q = curr_q + tl.arange(0, BLOCK_Q)
    q_mask = offs_q < q_end
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < actual_d

    # 🔥 间接寻址：获取真实的 Q 原生物理索引
    q_idx = tl.load(q_sorted_indices + b_idx * stride_q_idx_b + offs_q * stride_q_idx_n, mask=q_mask, other=0)

    # Q_sorted 是物理连续的
    q_ptrs = Q_sorted + b_idx * stride_qb + h_idx * stride_qh + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=q_mask[:, None] & mask_d[None, :], other=0.0)

    # dO_native, LSE, Delta 是物理离散的，必须使用 q_idx 读取
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

            # 🔥 间接寻址：获取真实的 K 原生物理索引
            k_idx = tl.load(k_sorted_indices + b_idx * stride_k_idx_b + offs_k * stride_k_idx_n, mask=k_mask, other=0)

            # K_sorted 是连续的，V_native 是离散的
            k_ptrs = K_sorted + b_idx * stride_kb + h_idx * stride_kh + offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=k_mask[:, None] & mask_d[None, :], other=0.0)
            
            v_ptrs = V_native + b_idx * stride_vb + h_idx * stride_vh + k_idx[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=k_mask[:, None] & mask_d[None, :], other=0.0)

            qk = tl.dot(q, tl.trans(k)) * sm_scale
            p = tl.math.exp(qk - lse[:, None])
            p = tl.where(q_mask[:, None] & k_mask[None, :], p, 0.0)

            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - delta[:, None]) * sm_scale

            # dq_acc += tl.dot(ds.to(q.dtype), k)
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

            dv_acc += tl.dot(pt.to(do.dtype), do.to(tl.float16))

            dp_t = tl.dot(v, tl.trans(do))
            ds_t = pt * (dp_t - delta[None, :]) * sm_scale

            dk_acc += tl.dot(ds_t.to(q.dtype), q.to(tl.float16))

    dk_ptrs = dK_sorted + b_idx * stride_kb + h_idx * stride_kh + offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd
    tl.store(dk_ptrs, dk_acc.to(k.dtype), mask=k_mask[:, None] & mask_d[None, :])

    # 🔥 神来之笔：利用 k_idx 将算好的梯度直接写回到 V 的原生物理地址中！
    # 因为每个 Token 属于唯一 Cluster，这里没有任何资源冲突，无需 atomic！
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
    actual_din, actual_dout, B: tl.constexpr, M: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_DIN: tl.constexpr, BLOCK_DOUT: tl.constexpr
):
    # 网格设计为 (M, H)，每个 Block 处理一个专家对应的一个多头
    cluster_id = tl.program_id(0)
    h_idx = tl.program_id(1)

    offs_din = tl.arange(0, BLOCK_DIN)
    offs_dout = tl.arange(0, BLOCK_DOUT)
    mask_din = offs_din < actual_din
    mask_dout = offs_dout < actual_dout

    # 1. 加载 Expert 权重 (W)
    w_ptrs = Experts + cluster_id * stride_em + h_idx * stride_eh + offs_din[:, None] * stride_edin + offs_dout[None, :] * stride_edout
    w = tl.load(w_ptrs, mask=mask_din[:, None] & mask_dout[None, :], other=0.0)

    # 在 SRAM 中分配一个 [BLOCK_DIN, BLOCK_DOUT] 的累加器，用于计算 W 的梯度
    dw_acc = tl.zeros([BLOCK_DIN, BLOCK_DOUT], dtype=tl.float32)

    # 2. 遍历所有的 Batch
    for b_idx in range(B):
        start_n = tl.load(cu_seqlens + b_idx * stride_cub + cluster_id)
        end_n = tl.load(cu_seqlens + b_idx * stride_cub + cluster_id + 1)

        # 遍历该 Batch 下属于该 Cluster 的所有 Token
        for curr_n in range(start_n, end_n, BLOCK_N):
            offs_n = curr_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < end_n

            # 获取真实索引
            idx_ptrs = sorted_indices + b_idx * stride_idxb + offs_n * stride_idxn
            src_idx = tl.load(idx_ptrs, mask=mask_n, other=0)

            # 间接寻址读入原生 X
            x_ptrs = X_native + b_idx * stride_xb + src_idx[:, None] * stride_xn + h_idx * stride_xh + offs_din[None, :] * stride_xdin
            x = tl.load(x_ptrs, mask=mask_n[:, None] & mask_din[None, :], other=0.0)

            # 连续读取排序后的梯度 dX_out (例如 dQ_sorted)
            dxo_ptrs = dX_out_sorted + b_idx * stride_dxob + offs_n[:, None] * stride_dxon + h_idx * stride_dxoh + offs_dout[None, :] * stride_dxodout
            dxo = tl.load(dxo_ptrs, mask=mask_n[:, None] & mask_dout[None, :], other=0.0)

            # 🔥 dW = X^T @ dX_out (累加到 SRAM 中)
            dw_acc += tl.dot(tl.trans(x.to(tl.float16)), dxo.to(tl.float16))

            # 🔥 dX_in = dX_out @ W^T
            dxin = tl.dot(dxo.to(tl.float16), tl.trans(w.to(tl.float16)))

            # 间接寻址，将 dX_in 的梯度直接写回到原生物理地址！
            dxin_ptrs = dX_native_out + b_idx * stride_xb + src_idx[:, None] * stride_xn + h_idx * stride_xh + offs_din[None, :] * stride_xdin
            # 因为这里 dX_native_out 已经被初始化为 0，并且每个 Token 严格属于一个 Cluster，
            # 这里绝对不会发生竞争写（Write Conflict），所以可以直接覆盖写入！
            existing_dx = tl.load(dxin_ptrs, mask=mask_n[:, None] & mask_din[None, :], other=0.0)
            
            # 🔥 修复：累加上当前的 dxin 再存回去
            tl.store(dxin_ptrs, existing_dx + dxin.to(x.dtype), mask=mask_n[:, None] & mask_din[None, :])

    # 3. 循环结束，将这个 Expert 的权重梯度写回全局显存
    dw_ptrs = dExperts_out + cluster_id * stride_em + h_idx * stride_eh + offs_din[:, None] * stride_edin + offs_dout[None, :] * stride_edout
    tl.store(dw_ptrs, dw_acc.to(w.dtype), mask=mask_din[:, None] & mask_dout[None, :])


def triton_moe_project_backward_v4(X_native, dX_out_sorted, Experts, dX_native_out, sorted_indices, cu_seqlens, M, H):
    B, N, H_dim, D_in_head = X_native.shape
    _, _, D_in_head, D_out_head = Experts.shape
    
    dExperts = torch.empty_like(Experts)
    
    BLOCK_DIN = triton.next_power_of_2(D_in_head)
    BLOCK_DOUT = triton.next_power_of_2(D_out_head)
    BLOCK_N = 32

    # Grid 只有 (M, H)，通过内部循环处理 B 和 N，彻底规避 W 的原子加锁
    grid = (M, H)
    
    bwd_expert_project_kernel_v4[grid](
        X_native, dX_out_sorted, Experts,
        dX_native_out, dExperts,
        sorted_indices, cu_seqlens,
        X_native.stride(0), X_native.stride(1), X_native.stride(2), X_native.stride(3),
        dX_out_sorted.stride(0), dX_out_sorted.stride(1), dX_out_sorted.stride(2), dX_out_sorted.stride(3),
        Experts.stride(0), Experts.stride(1), Experts.stride(2), Experts.stride(3),
        sorted_indices.stride(0), sorted_indices.stride(1), cu_seqlens.stride(0),
        actual_din=D_in_head, actual_dout=D_out_head, B=B, M=M,
        BLOCK_N=BLOCK_N, BLOCK_DIN=BLOCK_DIN, BLOCK_DOUT=BLOCK_DOUT, num_warps=2
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
    
    # dV 现在直接开辟原生形状
    dV_native = torch.zeros_like(V_native)
    sm_scale = 1.0 / (D_head ** 0.5)

    # Delta 的计算完全在原生物理维度进行
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
    """
    只计算路由、排序索引和 cu_seqlens。
    不涉及对 X 本身的 Gathering，彻底切断内存搬运墙。
    """
    B, N, _ = X.shape
    
    # 1. 路由 Logits 计算并获取 Cluster IDs
    logits = torch.matmul(X, Router_weight.transpose(0, 1))
    cluster_ids_orig = torch.argmax(logits, dim=-1).to(torch.int32) # [B, N]
    
    # 2. 获取排序索引 (这是零拷贝间接寻址的核心钥匙)
    sorted_indices = torch.argsort(cluster_ids_orig, dim=-1).to(torch.int32) # [B, N]
    
    # 3. 计算 cu_seqlens
    cu_seqlens = torch.zeros((B, M + 1), dtype=torch.int32, device=X.device)
    # 使用 F.one_hot 避免 T4 上的原子累加冲突
    counts = F.one_hot(cluster_ids_orig.to(torch.int64), num_classes=M).sum(dim=1) 
    cu_seqlens[:, 1:] = torch.cumsum(counts, dim=1).to(torch.int32)
    
    return cluster_ids_orig, sorted_indices, cu_seqlens


# =====================================================================
# 3. 核心 PyTorch 封装 (Autograd & Module)
# =====================================================================
class FinalCrossMoEMultiHeadAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X_q, X_kv, router_q, router_k, experts_q, experts_k, w_v_weight, H=8, num_splits=1):
        B, N_q, _ = X_q.shape
        _, N_kv, _ = X_kv.shape
        D_out_head = experts_q.shape[-1]
        M = experts_q.shape[0]

        # ---------------------------------------------------------
        # 2. 准备 V (不重排！) 
        # ---------------------------------------------------------
        # 直接执行线性投影，保持物理位置不变，彻底干掉 Stage 2 的 Gather 耗时
        V = F.linear(X_kv, w_v_weight).view(B, N_kv, H, D_out_head)


        q_ids_orig, q_sorted_idx, q_cu_seqlens = compute_routing_and_cu_seqlens(X_q, router_q, M)
        k_ids_orig, k_sorted_idx, k_cu_seqlens = compute_routing_and_cu_seqlens(X_kv, router_k, M)
        
        # 使用全新的 V4 Kernel 进行间接寻址 + Tensor Core 投影
        q_sorted = triton_moe_project_mha_v4(X_q, experts_q, q_sorted_idx, q_cu_seqlens, M, H)
        k_sorted = triton_moe_project_mha_v4(X_kv, experts_k, k_sorted_idx, k_cu_seqlens, M, H)
        # ---------------------------------------------------------
        # 4. 调用零拷贝 (v3) Triton 核心
        # ---------------------------------------------------------
        # Q 和 K 传 sorted 进去，V 传原生进去，靠 k_sorted_indices 去捞 V
        out, LSE = batched_flash_decoding_mha_v3(
            q_sorted, k_sorted, V, 
            q_ids_orig, k_ids_orig, k_sorted_idx, 
            k_cu_seqlens,
            M, num_splits=num_splits
        )

        # 🔥 只需保存原生状态和索引信息
        ctx.save_for_backward(
            X_q, X_kv, experts_q, experts_k, w_v_weight, 
            q_ids_orig, k_ids_orig,
            q_sorted_idx, k_sorted_idx, 
            q_cu_seqlens, k_cu_seqlens,
            q_sorted, k_sorted, V, out, LSE
        )
        # ctx.batch_indices_q = torch.arange(B, device=X_q.device).unsqueeze(1).expand(B, N_q)
        # ctx.batch_indices_kv = torch.arange(B, device=X_kv.device).unsqueeze(1).expand(B, N_kv)
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
        
        # 1. 原生 dO
        dO_native = grad_output.contiguous().view(B, N_q, H, D_out_head)
        out_native = out_native.contiguous()

        # 2. 注意力反向 (V4版)
        # 输出的 dQ 和 dK 是 sorted 状态，dV 已经是原生状态！
        dQ_sorted, dK_sorted, dV_native = triton_attention_backward_v4(
            q_sorted, k_sorted, V_native, out_native, dO_native, 
            LSE, q_cu_seqlens, k_cu_seqlens, 
            q_sorted_indices, k_sorted_indices, M, H
        )

        # 初始化最终将返回的 X 梯度 (原生形状)
        grad_X_q = torch.zeros_like(X_q)
        grad_X_kv = torch.zeros_like(X_kv)
        
        # 3. Value 反向 (dV_native 已经是原生的了，直接矩阵乘)
        dv_flat = dV_native.view(-1, H * D_out_head)
        X_kv_flat = X_kv.view(-1, H * D_in_head)
        grad_w_v_weight = torch.matmul(dv_flat.t(), X_kv_flat)
        grad_X_kv += torch.matmul(dv_flat, w_v_weight).view(B, N_kv, -1)

        # 4. 专家层反向 (V4版)
        # 将 dQ_sorted 灌入，它会在 Triton 里算出 dW_q，并直接把输入梯度写回原生 grad_X_q 中
        grad_experts_q = triton_moe_project_backward_v4(
            X_q.view(B, N_q, H, D_in_head), dQ_sorted, experts_q, 
            grad_X_q.view(B, N_q, H, D_in_head), 
            q_sorted_indices, q_cu_seqlens, M, H
        )

        grad_experts_k = triton_moe_project_backward_v4(
            X_kv.view(B, N_kv, H, D_in_head), dK_sorted, experts_k, 
            grad_X_kv.view(B, N_kv, H, D_in_head),  # 注意：这里直接把梯度叠加到 grad_X_kv 的原生张量上
            k_sorted_indices, k_cu_seqlens, M, H
        )

        # Q/K Router 的梯度逻辑如果需要的话，按照之前的标准 PyTorch AutoGrad 处理即可
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
        self.experts_q = nn.Parameter(torch.randn(\
            num_clusters, num_heads, d_in_head, d_out_head) * (d_in_head ** -0.5))
        self.experts_k = nn.Parameter(torch.randn(\
            num_clusters, num_heads, d_in_head, d_out_head) * (d_in_head ** -0.5))
        self.w_v = nn.Linear(d_in, d_out, bias=False)

    def forward(self, X_q, X_kv, num_splits=1):
        return FinalCrossMoEMultiHeadAttentionFunc.apply(
            X_q, X_kv, self.router_q, self.router_k, self.experts_q, self.experts_k, self.w_v.weight, \
                self.H, num_splits
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

        # 1. 调整维度以适应标准的注意力计算形状: (B, H, SeqLen, D)
        q = q.transpose(1, 2)  # [B, H, N_q, d_out_head]
        k = k.transpose(1, 2)  # [B, H, N_k, d_out_head]
        v = v.transpose(1, 2)  # [B, H, N_k, d_out_head]

        # 2. 构建聚类掩码 (Cluster Mask)
        # c_q: [B, N_q] -> [B, N_q, 1]
        # c_k: [B, N_k] -> [B, 1, N_k]
        # mask shape: [B, 1, N_q, N_k]，额外增加的 1 是为了与 H (Heads) 维度广播
        mask = (c_q.unsqueeze(-1) == c_k.unsqueeze(1)).unsqueeze(1)

        # 3. 计算 Attention Scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
        
        # 🔥 Pytorch Safe Masking: 找出那些没有任何合规 K 的“死尸 Query”
        invalid_rows = ~mask.any(dim=-1, keepdim=True)
        
        # 欺骗 Softmax：如果是无效行，我们就把它对应的整行 mask 临时强行变为 True，
        # 这样 Softmax 看到的就是全 0 而不是全 -inf，输出均匀分布 1/N，完美避开 NaN
        safe_mask = mask.clone()
        safe_mask = safe_mask.masked_fill(invalid_rows.expand_as(safe_mask), True)

        scores = scores.masked_fill(~safe_mask, float('-inf'))
        
        # 4. 计算 Softmax (现在绝对安全，不会有 NaN)
        attn = torch.softmax(scores, dim=-1)
        
        # 5. 暴力抹杀：算完后，把刚才欺骗 Softmax 的无效行注意力权重强行设为 0
        # 这会物理阻断无效行的梯度向后传导 (梯度也会变 0)
        attn = attn.masked_fill(invalid_rows, 0.0)

        # 6. 乘以 Value 得到最终结果
        out = torch.matmul(attn, v)

        # 6. 还原形状并展平
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


import torch.cuda.nvtx as nvtx

def run_nsight_profiling():
    B, N_q, N_kv, H, D_IN, D_OUT, M = 4, 6, 2048, 6, 768, 768, 4
    device = torch.device('cuda')
    
    triton_model = FinalCrossMoEMultiHeadAttention(D_IN, D_OUT, H, M).to(device).half()
    X_q = torch.randn((B, N_q, D_IN), dtype=torch.float16, device=device, requires_grad=True)
    X_kv = torch.randn((B, N_kv, D_IN), dtype=torch.float16, device=device, requires_grad=True)
    dO = torch.randn((B, N_q, D_OUT), dtype=torch.float16, device=device)

    # WARMUP (极其重要：让 Triton 完成 JIT 编译，不要把编译时间算进 Profile 里)
    for _ in range(3):
        out = triton_model(X_q, X_kv)
        out.backward(dO, retain_graph=True)
    torch.cuda.synchronize()

    # 开始精确捕获
    torch.cuda.cudart().cudaProfilerStart()
    
    nvtx.range_push("Triton_MoE_MHA_Forward")
    out = triton_model(X_q, X_kv)
    torch.cuda.synchronize()
    nvtx.range_pop()

    nvtx.range_push("Triton_MoE_MHA_Backward")
    out.backward(dO, retain_graph=True)
    torch.cuda.synchronize()
    nvtx.range_pop()

    torch.cuda.cudart().cudaProfilerStop()


def profile_forward_breakdown():
    B = 16       
    N_kv = 512 * 16
    H = 6
    D_IN = 768
    D_OUT = 768
    M = 4         
    device = torch.device('cuda')

    # 测试两个不同的 N_q
    for N_q in [96, 192]:
        print(f"\n" + "="*50)
        print(f"🚀 开始拆解性能瓶颈 (N_q = {N_q})")
        print("="*50)

        X_q = torch.randn((B, N_q, D_IN), dtype=torch.float16, device=device)
        X_kv = torch.randn((B, N_kv, D_IN), dtype=torch.float16, device=device)
        
        triton_model = FinalCrossMoEMultiHeadAttention(D_IN, D_OUT, H, M).to(device).half()

        batch_indices_q = torch.arange(B, device=X_q.device).unsqueeze(1).expand(B, N_q)
        batch_indices_kv = torch.arange(B, device=X_kv.device).unsqueeze(1).expand(B, N_kv)

        # -------------------------------------------------------------------
        # 阶段 1: 路由与排序 (Gather)
        # -------------------------------------------------------------------
        def stage1_route_and_sort():
            q_sorted, q_ids, _, q_idx = triton_moe_router_and_project_mha(X_q, triton_model.router_q, triton_model.experts_q, H)
            k_sorted, _, k_lens, k_idx = triton_moe_router_and_project_mha(X_kv, triton_model.router_k, triton_model.experts_k, H)
            return q_sorted, q_ids, q_idx, k_sorted, k_lens, k_idx

        # 预运行以获取中间变量，供后续阶段使用
        q_sorted, q_ids_orig, q_sorted_idx, k_sorted, k_cu_seqlens, k_sorted_idx = stage1_route_and_sort()

        # -------------------------------------------------------------------
        # 阶段 2: V 的投影与重排 (Gather)
        # -------------------------------------------------------------------
        def stage2_v_proj_and_sort():
            V = F.linear(X_kv, triton_model.w_v.weight).view(B, N_kv, H, D_OUT // H)
            v_sorted = V[batch_indices_kv, k_sorted_idx]
            q_ids_sorted = q_ids_orig[batch_indices_q, q_sorted_idx]
            return v_sorted, q_ids_sorted

        v_sorted, q_ids_sorted = stage2_v_proj_and_sort()

        # -------------------------------------------------------------------
        # 阶段 3: Triton 核心 Kernel (Phase 1+2)
        # -------------------------------------------------------------------
        def stage3_triton_kernel():
            # 确保你已经将代码中的 num_splits 改为了 1
            return batched_flash_decoding_mha_v3(q_sorted, k_sorted, v_sorted, q_ids_sorted, k_cu_seqlens, M, num_splits=1)

        out_sorted, _ = stage3_triton_kernel()

        # -------------------------------------------------------------------
        # 阶段 4: 逆排序 (Scatter/Unsort)
        # -------------------------------------------------------------------
        def stage4_unsort():
            q_unsort_idx = torch.argsort(q_sorted_idx, dim=-1)
            final_out = out_sorted[batch_indices_q, q_unsort_idx]
            return final_out

        # -------------------------------------------------------------------
        # 精确计时
        # -------------------------------------------------------------------
        quantiles = [0.5, 0.2, 0.8]
        ms1, _, _ = triton.testing.do_bench(stage1_route_and_sort, quantiles=quantiles)
        ms2, _, _ = triton.testing.do_bench(stage2_v_proj_and_sort, quantiles=quantiles)
        ms3, _, _ = triton.testing.do_bench(stage3_triton_kernel, quantiles=quantiles)
        ms4, _, _ = triton.testing.do_bench(stage4_unsort, quantiles=quantiles)

        total_ms = ms1 + ms2 + ms3 + ms4
        
        print(f"{'阶段 (Stage)':<30} | {'耗时 (ms)':<15} | {'占比 (%)'}")
        print("-" * 65)
        print(f"{'1. 路由与专家投影 (Router+Gather)':<30} | {ms1:<15.4f} | {ms1/total_ms*100:.2f}%")
        print(f"{'2. V 投影与重排 (V-Proj+Gather)':<30} | {ms2:<15.4f} | {ms2/total_ms*100:.2f}%")
        print(f"{'3. Triton 注意力核 (Attention)':<30} | {ms3:<15.4f} | {ms3/total_ms*100:.2f}%")
        print(f"{'4. 逆排序恢复 (Unsort)':<30} | {ms4:<15.4f} | {ms4/total_ms*100:.2f}%")
        print("-" * 65)
        print(f"{'合计累加耗时 (Total)':<30} | {total_ms:<15.4f} | 100.00%")



def profile_forward_breakdown_v3():
    B = 16       
    N_kv = 512 * 16 # 8192
    H = 6
    D_IN = 768
    D_OUT = 768
    M = 4         
    device = torch.device('cuda')

    # 测试两个不同的 N_q
    for N_q in [5, 96, 192]:
        print(f"\n" + "="*50)
        print(f"🚀 开始拆解性能瓶颈 (V3 零拷贝架构, N_q = {N_q})")
        print("="*50)

        X_q = torch.randn((B, N_q, D_IN), dtype=torch.float16, device=device)
        X_kv = torch.randn((B, N_kv, D_IN), dtype=torch.float16, device=device)
        
        triton_model = FinalCrossMoEMultiHeadAttention(D_IN, D_OUT, H, M).to(device).half()

        # -------------------------------------------------------------------
        # 阶段 1: 路由计算与 Q/K 的物理专家投影 
        # (暂时保留 Q/K 重排，因为 Expert_W 依然需要物理连续以便于 einsum 计算)
        # -------------------------------------------------------------------
        def stage1_route_and_project_qk():
            q_ids_orig, q_sorted_idx, q_cu_seqlens = compute_routing_and_cu_seqlens(X_q, triton_model.router_q, M)
            k_ids_orig, k_sorted_idx, k_cu_seqlens = compute_routing_and_cu_seqlens(X_kv, triton_model.router_k, M)
            
            # 使用全新的 V4 Kernel 进行间接寻址 + Tensor Core 投影
            q_sorted = triton_moe_project_mha_v4(X_q, triton_model.experts_q, q_sorted_idx, q_cu_seqlens, M, H)
            k_sorted = triton_moe_project_mha_v4(X_kv, triton_model.experts_k, k_sorted_idx, k_cu_seqlens, M, H)
            
            return q_ids_orig, q_sorted_idx, q_cu_seqlens, k_ids_orig, k_sorted_idx, k_cu_seqlens, q_sorted, k_sorted

        q_ids_orig, q_sorted_idx, q_cu_seqlens, k_ids_orig, k_sorted_idx, k_cu_seqlens, q_sorted, k_sorted = stage1_route_and_project_qk()

        # -------------------------------------------------------------------
        # 阶段 2: 纯净 V 投影 (ZERO-COPY! 没有任何索引搬运和 Gather 操作)
        # -------------------------------------------------------------------
        def stage2_v_proj_only():
            V = F.linear(X_kv, triton_model.w_v.weight).view(B, N_kv, H, D_OUT // H)
            return V

        V_native = stage2_v_proj_only()

        # -------------------------------------------------------------------
        # 阶段 3: Triton 核心 Kernel (间接寻址)
        # -------------------------------------------------------------------
        def stage3_triton_kernel_v3():
            out, LSE = batched_flash_decoding_mha_v3(
                q_sorted, k_sorted, V_native, 
                q_ids_orig, q_sorted_idx, k_sorted_idx, k_cu_seqlens, # <-- 注意这里加上了 k_cu_seqlens
                M, num_splits=1
            )
            return out

        out_final = stage3_triton_kernel_v3()

        # -------------------------------------------------------------------
        # 精确计时
        # -------------------------------------------------------------------
        quantiles = [0.5, 0.2, 0.8]
        ms1, _, _ = triton.testing.do_bench(stage1_route_and_project_qk, quantiles=quantiles)
        ms2, _, _ = triton.testing.do_bench(stage2_v_proj_only, quantiles=quantiles)
        ms3, _, _ = triton.testing.do_bench(stage3_triton_kernel_v3, quantiles=quantiles)
        
        # 第 4 阶段 (Unsort) 已经被 v3 Kernel 的原位写回彻底消灭
        ms4 = 0.0

        total_ms = ms1 + ms2 + ms3 + ms4
        
        print(f"{'阶段 (Stage)':<30} | {'耗时 (ms)':<15} | {'占比 (%)'}")
        print("-" * 65)
        print(f"{'1. 路由与 Q/K 投影':<30} | {ms1:<15.4f} | {ms1/total_ms*100:.2f}%")
        print(f"{'2. V 纯投影 (Zero-Copy)':<30} | {ms2:<15.4f} | {ms2/total_ms*100:.2f}%")
        print(f"{'3. Triton 注意力核 (间接寻址)':<30} | {ms3:<15.4f} | {ms3/total_ms*100:.2f}%")
        print(f"{'4. 逆排序恢复 (已被彻底消灭)':<30} | {ms4:<15.4f} | {ms4/total_ms*100:.2f}%")
        print("-" * 65)
        print(f"{'合计累加耗时 (Total)':<30} | {total_ms:<15.4f} | 100.00%")

if __name__ == '__main__':
    # profile_forward_breakdown_v3()
    run_full_verification_and_benchmark()


# if __name__ == '__main__':
#     run_nsight_profiling()


# if __name__ == '__main__':
    # profile_forward_breakdown()


    