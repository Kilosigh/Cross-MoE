import torch
import torch.nn as nn
import triton
import triton.language as tl

# =====================================================================
# 1. Triton Kernel: 融合路由收集与多头专家映射 (Multi-Head)
# =====================================================================
@triton.jit
def fused_gather_expert_mha_kernel(
    X, Experts, Out,
    sorted_indices, cluster_ids,
    stride_xb, stride_xn, stride_xh, stride_xd,
    stride_em, stride_eh, stride_ed_in, stride_ed_out,
    stride_ob, stride_on, stride_oh, stride_od,
    stride_idx_b, stride_idx_n,
    stride_cid_b, stride_cid_n,
    actual_d_in, actual_d_out, H,
    BLOCK_DIN: tl.constexpr, BLOCK_DOUT: tl.constexpr
):
    b_idx = tl.program_id(0)
    out_token_idx = tl.program_id(1)
    pid_z = tl.program_id(2)  # 包含 H 和 D_out 块
    
    num_d_blocks = tl.cdiv(actual_d_out, BLOCK_DOUT)
    h_idx = pid_z // num_d_blocks
    out_d_idx = pid_z % num_d_blocks

    src_token_idx = tl.load(sorted_indices + b_idx * stride_idx_b + out_token_idx * stride_idx_n)
    
    # [核心] 所有头共享一个 cluster_id
    c_id = tl.load(cluster_ids + b_idx * stride_cid_b + src_token_idx * stride_cid_n)

    offs_din = tl.arange(0, BLOCK_DIN)
    offs_dout = out_d_idx * BLOCK_DOUT + tl.arange(0, BLOCK_DOUT)

    mask_din = offs_din < actual_d_in
    mask_dout = offs_dout < actual_d_out

    # 加载当前 Token 的特定 Head 切片
    x_ptrs = X + b_idx * stride_xb + src_token_idx * stride_xn + h_idx * stride_xh + offs_din * stride_xd
    x = tl.load(x_ptrs, mask=mask_din, other=0.0) 

    # 加载当前 Head 专属的专家子权重
    w_ptrs = Experts + c_id * stride_em + h_idx * stride_eh + offs_din[:, None] * stride_ed_in + offs_dout[None, :] * stride_ed_out
    w = tl.load(w_ptrs, mask=mask_din[:, None] & mask_dout[None, :], other=0.0)

    out = tl.sum(x[:, None] * w, axis=0)

    out_ptrs = Out + b_idx * stride_ob + out_token_idx * stride_on + h_idx * stride_oh + offs_dout * stride_od
    tl.store(out_ptrs, out, mask=mask_dout)


def triton_moe_router_and_project_mha(X, Router, Experts, H):
    B, N, D_in = X.shape
    M, _, D_in_head, D_out_head = Experts.shape
    
    # 1. 路由共享：使用完整的 X 决定 cluster
    logits = torch.matmul(X, Router.transpose(0, 1))
    cluster_ids = torch.argmax(logits, dim=-1).to(torch.int32)
    
    sorted_indices = torch.argsort(cluster_ids, dim=-1).to(torch.int32)
    cu_seqlens = torch.zeros((B, M + 1), dtype=torch.int32, device=X.device)
    for b in range(B):
        cu_seqlens[b, 1:] = torch.cumsum(torch.bincount(cluster_ids[b], minlength=M), dim=0)

    # 2. 特征切分为多头
    X_view = X.view(B, N, H, D_in_head)
    out = torch.empty((B, N, H, D_out_head), dtype=X.dtype, device=X.device)
    
    BLOCK_DIN = triton.next_power_of_2(D_in_head)
    BLOCK_DOUT = 32 if D_out_head > 16 else 16 
    
    grid = (B, N, H * triton.cdiv(D_out_head, BLOCK_DOUT))
    fused_gather_expert_mha_kernel[grid](
        X_view, Experts, out,
        sorted_indices, cluster_ids,
        X_view.stride(0), X_view.stride(1), X_view.stride(2), X_view.stride(3),
        Experts.stride(0), Experts.stride(1), Experts.stride(2), Experts.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        sorted_indices.stride(0), sorted_indices.stride(1),
        cluster_ids.stride(0), cluster_ids.stride(1),
        D_in_head, D_out_head, H,
        BLOCK_DIN=BLOCK_DIN, BLOCK_DOUT=BLOCK_DOUT,
        num_warps=4, num_stages=2
    )
    return out, cluster_ids, cu_seqlens, sorted_indices

# =====================================================================
# 2. Triton Kernel: Batched Split-K MHA Clustered Attention
# =====================================================================
@triton.jit
def batched_flash_decoding_mha_phase1(
    Q, K, V,
    mid_acc, mid_m, mid_l,
    q_cluster_ids, k_cu_seqlens, sm_scale,
    stride_qb, stride_qn, stride_qh, stride_qd,
    stride_kb, stride_kn, stride_kh, stride_kd,
    stride_vb, stride_vn, stride_vh, stride_vd,
    stride_mid_acc_b, stride_mid_acc_h, stride_mid_acc_q, stride_mid_acc_s, stride_mid_acc_d,
    stride_mid_m_b, stride_mid_m_h, stride_mid_m_q, stride_mid_m_s,
    stride_mid_l_b, stride_mid_l_h, stride_mid_l_q, stride_mid_l_s,
    stride_q_cid_b, stride_q_cid_n,
    stride_k_cu_b, stride_k_cu_m,
    actual_d, H,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    q_idx = tl.program_id(0)
    split_idx = tl.program_id(1)
    pid_bh = tl.program_id(2)  # B * H
    
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    num_splits = tl.num_programs(1)

    # Cluster ID 不分 Head，只跟 batch 和 token 有关
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
        # [核心] 引入 Head 偏移量
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

    mid_acc_ptrs = mid_acc + b_idx * stride_mid_acc_b + h_idx * stride_mid_acc_h + q_idx * stride_mid_acc_q + split_idx * stride_mid_acc_s + offs_d * stride_mid_acc_d
    mid_m_ptr = mid_m + b_idx * stride_mid_m_b + h_idx * stride_mid_m_h + q_idx * stride_mid_m_q + split_idx * stride_mid_m_s
    mid_l_ptr = mid_l + b_idx * stride_mid_l_b + h_idx * stride_mid_l_h + q_idx * stride_mid_l_q + split_idx * stride_mid_l_s

    tl.store(mid_acc_ptrs, acc, mask=mask_d)
    tl.store(mid_m_ptr, m_i)
    tl.store(mid_l_ptr, l_i)


@triton.jit
def batched_flash_decoding_mha_phase2(
    mid_acc, mid_m, mid_l, Out, LSE, 
    stride_mid_acc_b, stride_mid_acc_h, stride_mid_acc_q, stride_mid_acc_s, stride_mid_acc_d,
    stride_mid_m_b, stride_mid_m_h, stride_mid_m_q, stride_mid_m_s,
    stride_mid_l_b, stride_mid_l_h, stride_mid_l_q, stride_mid_l_s,
    stride_ob, stride_on, stride_oh, stride_od,
    stride_lse_b, stride_lse_n, stride_lse_h,    
    actual_d, H,
    NUM_SPLITS: tl.constexpr, BLOCK_D: tl.constexpr
):
    q_idx = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    
    offs_d = tl.arange(0, BLOCK_D)
    offs_s = tl.arange(0, NUM_SPLITS)
    mask_d = offs_d < actual_d 

    mid_m_ptrs = mid_m + b_idx * stride_mid_m_b + h_idx * stride_mid_m_h + q_idx * stride_mid_m_q + offs_s * stride_mid_m_s
    m_locals = tl.load(mid_m_ptrs)
    m_global = tl.max(m_locals, 0)

    mid_l_ptrs = mid_l + b_idx * stride_mid_l_b + h_idx * stride_mid_l_h + q_idx * stride_mid_l_q + offs_s * stride_mid_l_s
    l_locals = tl.load(mid_l_ptrs)
    weights = tl.math.exp(m_locals - m_global)
    l_global = tl.sum(l_locals * weights, 0)

    lse_global = m_global + tl.math.log(l_global)
    lse_ptr = LSE + b_idx * stride_lse_b + q_idx * stride_lse_n + h_idx * stride_lse_h
    tl.store(lse_ptr, lse_global)

    acc_global = tl.zeros([BLOCK_D], dtype=tl.float32)
    
    for s in range(NUM_SPLITS):
        w = tl.load(mid_m + b_idx * stride_mid_m_b + h_idx * stride_mid_m_h + q_idx * stride_mid_m_q + s * stride_mid_m_s)
        w = tl.math.exp(w - m_global)
        
        mid_acc_ptrs = mid_acc + b_idx * stride_mid_acc_b + h_idx * stride_mid_acc_h + q_idx * stride_mid_acc_q + s * stride_mid_acc_s + offs_d * stride_mid_acc_d
        acc_local = tl.load(mid_acc_ptrs, mask=mask_d, other=0.0) 
        acc_global += acc_local * w

    out = acc_global / l_global
    out_ptrs = Out + b_idx * stride_ob + q_idx * stride_on + h_idx * stride_oh + offs_d * stride_od
    tl.store(out_ptrs, out, mask=mask_d)


def batched_flash_decoding_mha(q, k, v, q_cluster_ids, k_cu_seqlens, num_splits=16):
    B, N_q, H, D = q.shape
    mid_acc = torch.empty((B, H, N_q, num_splits, D), dtype=torch.float32, device=q.device)
    mid_m = torch.empty((B, H, N_q, num_splits), dtype=torch.float32, device=q.device)
    mid_l = torch.empty((B, H, N_q, num_splits), dtype=torch.float32, device=q.device)
    
    out = torch.empty_like(q)
    LSE = torch.empty((B, N_q, H), dtype=torch.float32, device=q.device) 
    
    sm_scale = 1.0 / (D ** 0.5)
    BLOCK_D = triton.next_power_of_2(D)

    grid_1 = (N_q, num_splits, B * H)
    batched_flash_decoding_mha_phase1[grid_1](
        q, k, v, mid_acc, mid_m, mid_l, q_cluster_ids, k_cu_seqlens, sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        mid_acc.stride(0), mid_acc.stride(1), mid_acc.stride(2), mid_acc.stride(3), mid_acc.stride(4),
        mid_m.stride(0), mid_m.stride(1), mid_m.stride(2), mid_m.stride(3),
        mid_l.stride(0), mid_l.stride(1), mid_l.stride(2), mid_l.stride(3),
        q_cluster_ids.stride(0), q_cluster_ids.stride(1),
        k_cu_seqlens.stride(0), k_cu_seqlens.stride(1),
        actual_d=D, H=H,
        BLOCK_N=16, BLOCK_D=BLOCK_D, num_warps=4
    )

    grid_2 = (N_q, B * H)
    batched_flash_decoding_mha_phase2[grid_2](
        mid_acc, mid_m, mid_l, out, LSE,
        mid_acc.stride(0), mid_acc.stride(1), mid_acc.stride(2), mid_acc.stride(3), mid_acc.stride(4),
        mid_m.stride(0), mid_m.stride(1), mid_m.stride(2), mid_m.stride(3),
        mid_l.stride(0), mid_l.stride(1), mid_l.stride(2), mid_l.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        actual_d=D, H=H,
        NUM_SPLITS=num_splits, BLOCK_D=BLOCK_D, num_warps=4
    )
    
    return out, LSE

# =====================================================================
# 终极封装: 端到端 PyTorch MHA Module (支持 Q 与 K/V 长度解耦)
# =====================================================================
class CrossMoEMultiHeadAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, num_heads: int, num_clusters: int, num_splits: int = 16):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.H = num_heads
        self.M = num_clusters
        self.num_splits = num_splits
        
        self.d_in_head = d_in // num_heads
        self.d_out_head = d_out // num_heads

        # 路由共享 
        self.router_q = nn.Parameter(torch.randn(self.M, d_in) * (d_in ** -0.5))
        self.router_k = nn.Parameter(torch.randn(self.M, d_in) * (d_in ** -0.5))

        # 专家按 Head 拆分
        self.experts_q = nn.Parameter(torch.randn(self.M, self.H, self.d_in_head, self.d_out_head) * (self.d_in_head ** -0.5))
        self.experts_k = nn.Parameter(torch.randn(self.M, self.H, self.d_in_head, self.d_out_head) * (self.d_in_head ** -0.5))
        
        self.w_v = nn.Linear(d_in, d_out, bias=False)

    def forward(self, X_q: torch.Tensor, X_kv: torch.Tensor):
        """
        X_q:  [Batch, N_q, D_in]
        X_kv: [Batch, N_k, D_in]
        返回: Output [Batch, N_q, D_out]
        """
        B, N_q, _ = X_q.shape
        _, N_k, _ = X_kv.shape
        
        # 分别为 Q 和 K/V 生成 batch_indices，因为长度不同了
        batch_indices_q = torch.arange(B, device=X_q.device).unsqueeze(1).expand(B, N_q)
        batch_indices_k = torch.arange(B, device=X_kv.device).unsqueeze(1).expand(B, N_k)

        # -------------------------------------------------------------
        # Phase 1: MoE Routing & Projection 独立处理 Q 和 K/V
        # -------------------------------------------------------------
        q_sorted, q_cluster_ids, _, q_sorted_indices = triton_moe_router_and_project_mha(
            X_q, self.router_q, self.experts_q, self.H
        )
        
        k_sorted, _, k_cu_seqlens, k_sorted_indices = triton_moe_router_and_project_mha(
            X_kv, self.router_k, self.experts_k, self.H
        )

        # -------------------------------------------------------------
        # Phase 2: Value Projection (使用 X_kv 的长度)
        # -------------------------------------------------------------
        V = self.w_v(X_kv).view(B, N_k, self.H, self.d_out_head)
        v_sorted = V[batch_indices_k, k_sorted_indices]

        # -------------------------------------------------------------
        # Phase 3: Split-K MHA Clustered Attention
        # -------------------------------------------------------------
        # 注意这里必须使用 batch_indices_q，因为 q_cluster_ids 的长度是 N_q
        q_cluster_ids_sorted = q_cluster_ids[batch_indices_q, q_sorted_indices]
        
        out_sorted, _ = batched_flash_decoding_mha(
            q_sorted, k_sorted, v_sorted, 
            q_cluster_ids_sorted, k_cu_seqlens, 
            num_splits=self.num_splits
        )

        # -------------------------------------------------------------
        # Phase 4: Unsort & Flatten 恢复物理顺序 (按照 Q 的长度恢复)
        # -------------------------------------------------------------
        q_unsort_indices = torch.argsort(q_sorted_indices, dim=-1)
        final_out = out_sorted[batch_indices_q, q_unsort_indices]

        return final_out.view(B, N_q, self.d_out)


# =====================================================================
# 原生 PyTorch 对照组 (用于严格精度校验)
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

        # 1. 路由 (共享)
        c_q = torch.argmax(torch.matmul(X_q, self.router_q.transpose(0, 1)), dim=-1)
        c_k = torch.argmax(torch.matmul(X_kv, self.router_k.transpose(0, 1)), dim=-1)

        # 2. 多头专家映射 (einsum)
        q = torch.zeros((B, N_q, self.H, self.d_out_head), device=X_q.device, dtype=X_q.dtype)
        k = torch.zeros((B, N_k, self.H, self.d_out_head), device=X_q.device, dtype=X_q.dtype)
        
        for m in range(self.M):
            # Q
            mask_q = (c_q == m) 
            if mask_q.any():
                X_m_q = X_q[mask_q].view(-1, self.H, self.d_in_head) 
                W_m_q = self.experts_q[m] # [H, Din_H, Dout_H]
                q[mask_q] = torch.einsum('thd,hdo->tho', X_m_q, W_m_q)
            # K
            mask_k = (c_k == m)
            if mask_k.any():
                X_m_k = X_kv[mask_k].view(-1, self.H, self.d_in_head)
                W_m_k = self.experts_k[m]
                k[mask_k] = torch.einsum('thd,hdo->tho', X_m_k, W_m_k)

        # 3. Value 投影
        v = self.w_v(X_kv).view(B, N_k, self.H, self.d_out_head)

        # 4. Attention (独立 Head)
        for b in range(B):
            for i in range(N_q):
                cluster = c_q[b, i]
                mask = (c_k[b] == cluster)
                if not mask.any(): continue
                
                # 提取当前 Token 的 Q, 维度转换以适应 bmm: [H, 1, D_head]
                q_i = q[b, i].unsqueeze(1) 
                # 提取对应 Cluster 的 K 和 V: [H, Seq_len, D_head]
                k_local = k[b, mask].transpose(0, 1) 
                v_local = v[b, mask].transpose(0, 1) 
                
                # BMM 在 Head 维进行批次乘法
                scores = torch.bmm(q_i, k_local.transpose(1, 2)) * sm_scale
                attn = torch.softmax(scores, dim=-1)
                out[b, i] = torch.bmm(attn, v_local).squeeze(1) # 写回 [H, D_head]
                
        return out.view(B, N_q, -1)


# =====================================================================
# 性能压测与校验
# =====================================================================
# =====================================================================
# 性能压测与校验 (异构长度版)
# =====================================================================
def run_benchmark():
    B = 4          
    N_q = 256       # Q 的长度
    N_k = 1024      # K/V 的长度 (例如 Cross-Attention 场景)
    D_IN = 512
    D_OUT = 512
    H = 8           
    M = 4           
    num_splits = 16 
    device = torch.device('cuda')

    print(f"初始化环境... [B={B}, N_q={N_q}, N_k={N_k}, H={H}, D_in={D_IN}, D_out={D_OUT}, M={M}]")
    
    # 构建长度独立的 X_q 和 X_kv
    X_q = torch.randn((B, N_q, D_IN), dtype=torch.float32, device=device)
    X_kv = torch.randn((B, N_k, D_IN), dtype=torch.float32, device=device)

    # 1. 实例化模型
    triton_model = CrossMoEMultiHeadAttention(D_IN, D_OUT, H, M, num_splits).to(device)
    torch_model = TorchCrossMoEMultiHeadAttention(D_IN, D_OUT, H, M).to(device)

    # 2. 强制对齐权重
    with torch.no_grad():
        torch_model.router_q.copy_(triton_model.router_q)
        torch_model.router_k.copy_(triton_model.router_k)
        torch_model.experts_q.copy_(triton_model.experts_q)
        torch_model.experts_k.copy_(triton_model.experts_k)
        torch_model.w_v.weight.copy_(triton_model.w_v.weight)

    triton_model.eval()
    torch_model.eval()

    # 3. 端到端精度校验
    print("正在进行解耦长度精度校验...")
    with torch.no_grad():
        # 这里传入拆分后的 X_q 和 X_kv
        out_torch = torch_model(X_q, X_kv)
        out_triton = triton_model(X_q, X_kv)
        
    assert torch.allclose(out_torch, out_triton, atol=1e-3), "端到端解耦长度精度校验失败！"
    print(f"✅ 精度校验通过！输出 Shape 为: {out_triton.shape} (Triton MHA 版本与 PyTorch 严格一致)\n")


if __name__ == '__main__':
    run_benchmark()