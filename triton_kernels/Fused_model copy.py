import torch
import torch.nn as nn
import triton
import triton.language as tl

# =====================================================================
# 1. Triton Kernel: 融合路由收集与专家映射 (修复版)
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
    out_token_idx = tl.program_id(1)  
    out_d_idx = tl.program_id(2)      

    src_token_idx = tl.load(sorted_indices + b_idx * stride_idx_b + out_token_idx * stride_idx_n)
    c_id = tl.load(cluster_ids + b_idx * stride_cid_b + src_token_idx * stride_cid_n)

    offs_din = tl.arange(0, BLOCK_DIN)
    offs_dout = out_d_idx * BLOCK_DOUT + tl.arange(0, BLOCK_DOUT)

    # 加载 Token (1D向量) 和 Expert 权重 (2D矩阵)
    x_ptrs = X + b_idx * stride_xb + src_token_idx * stride_xn + offs_din * stride_xd
    x = tl.load(x_ptrs) 

    w_ptrs = Experts + c_id * stride_em + offs_din[:, None] * stride_ed_in + offs_dout[None, :] * stride_ed_out
    w = tl.load(w_ptrs)

    # 修复 TensorCore 限制: 使用 1D 广播乘法 + 归约求和
    out = tl.sum(x[:, None] * w, axis=0)

    out_ptrs = Out + b_idx * stride_ob + out_token_idx * stride_on + offs_dout * stride_od
    tl.store(out_ptrs, out)

def triton_moe_router_and_project(X, Router, Experts):
    B, N, D_in = X.shape
    M, _, D_out = Experts.shape
    
    # 1. 计算路由得分 (此处使用 PyTorch 原生算子，因其对密集型矩阵乘优化极好)
    logits = torch.matmul(X, Router.transpose(0, 1))
    cluster_ids = torch.argmax(logits, dim=-1).to(torch.int32)
    
    # 2. 生成排序索引与偏移量
    sorted_indices = torch.argsort(cluster_ids, dim=-1).to(torch.int32)
    cu_seqlens = torch.zeros((B, M + 1), dtype=torch.int32, device=X.device)
    for b in range(B):
        cu_seqlens[b, 1:] = torch.cumsum(torch.bincount(cluster_ids[b], minlength=M), dim=0)

    # 3. 启动 Triton Kernel 进行零中间显存的重排与专家计算
    out = torch.empty((B, N, D_out), dtype=X.dtype, device=X.device)
    BLOCK_DIN = D_in
    BLOCK_DOUT = 16 # 控制 SRAM 占用
    
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
    return out, cluster_ids, cu_seqlens, sorted_indices

# =====================================================================
# 2. Triton Kernel: Batched Split-K Clustered Attention (Flash-Decoding)
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
    b_idx = tl.program_id(2)
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

    if chunk_start < chunk_end:
        offs_d = tl.arange(0, BLOCK_D)
        q_ptrs = Q + b_idx * stride_qb + q_idx * stride_qn + offs_d * stride_qd
        q = tl.load(q_ptrs) * sm_scale

        for start_n in range(chunk_start, chunk_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k_mask = offs_n < chunk_end

            k_ptrs = K + b_idx * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)

            qk = tl.sum(q[None, :] * k, axis=1)
            qk = tl.where(k_mask, qk, -float('inf'))

            m_ij = tl.maximum(m_i, tl.max(qk, 0))
            p = tl.math.exp(qk - m_ij)
            l_ij = tl.sum(p, 0)

            alpha = tl.math.exp(m_i - m_ij)
            acc = acc * alpha

            v_ptrs = V + b_idx * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
            acc += tl.sum(p[:, None] * v, axis=0)

            m_i = m_ij
            l_i = l_i * alpha + l_ij

    mid_acc_ptrs = mid_acc + b_idx * stride_mid_acc_b + q_idx * stride_mid_acc_q + split_idx * stride_mid_acc_s + tl.arange(0, BLOCK_D) * stride_mid_acc_d
    mid_m_ptr = mid_m + b_idx * stride_mid_m_b + q_idx * stride_mid_m_q + split_idx * stride_mid_m_s
    mid_l_ptr = mid_l + b_idx * stride_mid_l_b + q_idx * stride_mid_l_q + split_idx * stride_mid_l_s

    tl.store(mid_acc_ptrs, acc)
    tl.store(mid_m_ptr, m_i)
    tl.store(mid_l_ptr, l_i)

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
    b_idx = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_D)
    offs_s = tl.arange(0, NUM_SPLITS)

    mid_m_ptrs = mid_m + b_idx * stride_mid_m_b + q_idx * stride_mid_m_q + offs_s * stride_mid_m_s
    m_locals = tl.load(mid_m_ptrs)
    m_global = tl.max(m_locals, 0)

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

def batched_flash_decoding_attention(q, k, v, q_cluster_ids, k_cu_seqlens, num_splits=16):
    B, N_q, D = q.shape
    mid_acc = torch.empty((B, N_q, num_splits, D), dtype=torch.float32, device=q.device)
    mid_m = torch.empty((B, N_q, num_splits), dtype=torch.float32, device=q.device)
    mid_l = torch.empty((B, N_q, num_splits), dtype=torch.float32, device=q.device)
    out = torch.empty_like(q)
    sm_scale = 1.0 / (D ** 0.5)

    grid_1 = (N_q, num_splits, B)
    batched_flash_decoding_phase1_kernel[grid_1](
        q, k, v, mid_acc, mid_m, mid_l, q_cluster_ids, k_cu_seqlens, sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        mid_acc.stride(0), mid_acc.stride(1), mid_acc.stride(2), mid_acc.stride(3),
        mid_m.stride(0), mid_m.stride(1), mid_m.stride(2),
        mid_l.stride(0), mid_l.stride(1), mid_l.stride(2),
        q_cluster_ids.stride(0), q_cluster_ids.stride(1),
        k_cu_seqlens.stride(0), k_cu_seqlens.stride(1),
        BLOCK_N=16, BLOCK_D=D, num_warps=4
    )

    grid_2 = (N_q, B)
    batched_flash_decoding_phase2_kernel[grid_2](
        mid_acc, mid_m, mid_l, out,
        mid_acc.stride(0), mid_acc.stride(1), mid_acc.stride(2), mid_acc.stride(3),
        mid_m.stride(0), mid_m.stride(1), mid_m.stride(2),
        mid_l.stride(0), mid_l.stride(1), mid_l.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        NUM_SPLITS=num_splits, BLOCK_D=D, num_warps=4
    )
    return out

# =====================================================================
# 3. 终极封装: 端到端 PyTorch Module
# =====================================================================
class CrossMoEClusteredAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, num_clusters: int, num_splits: int = 16):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.M = num_clusters
        self.num_splits = num_splits

        # 初始化 Router 参数 (FP32)
        self.router_q = nn.Parameter(torch.randn(self.M, d_in) * (d_in ** -0.5))
        self.router_k = nn.Parameter(torch.randn(self.M, d_in) * (d_in ** -0.5))

        # 初始化 Expert 参数 (FP32)
        self.experts_q = nn.Parameter(torch.randn(self.M, d_in, d_out) * (d_in ** -0.5))
        self.experts_k = nn.Parameter(torch.randn(self.M, d_in, d_out) * (d_in ** -0.5))
        
        # 标准的 Value 投影
        self.w_v = nn.Linear(d_in, d_out, bias=False)

    def forward(self, X: torch.Tensor):
        """
        X: [Batch, Seq_Len, D_in]
        返回: 经过聚类注意力计算后的 Output [Batch, Seq_Len, D_out]
        """
        B, N, _ = X.shape
        batch_indices = torch.arange(B, device=X.device).unsqueeze(1).expand(B, N)

        # -------------------------------------------------------------
        # Phase 1: MoE 路由与重排序映射 (Q 和 K)
        # -------------------------------------------------------------
        # 生成排序后的 Q，以及 Q 的原始归属簇
        q_sorted, q_cluster_ids_orig, _, q_sorted_indices = triton_moe_router_and_project(
            X, self.router_q, self.experts_q
        )
        
        # 生成排序后的 K，以及用于界定 K 簇边界的 cu_seqlens
        k_sorted, _, k_cu_seqlens, k_sorted_indices = triton_moe_router_and_project(
            X, self.router_k, self.experts_k
        )

        # -------------------------------------------------------------
        # Phase 2: Value 映射与内存对齐
        # -------------------------------------------------------------
        # V 使用标准线性层计算，但必须按照 K 的路由结果进行重排，以保证底层连续
        V = self.w_v(X)
        v_sorted = V[batch_indices, k_sorted_indices]

        # -------------------------------------------------------------
        # Phase 3: Split-K Clustered Attention 计算
        # -------------------------------------------------------------
        # 注意：Q 已经是排好序的了，所以我们喂给 Attention 的 q_cluster_ids 
        # 必须是按照 q_sorted_indices 重排过的，这样 kernel 才能找对对应的 K/V
        q_cluster_ids_sorted = q_cluster_ids_orig[batch_indices, q_sorted_indices]
        
        out_sorted = batched_flash_decoding_attention(
            q_sorted, k_sorted, v_sorted, 
            q_cluster_ids_sorted, k_cu_seqlens, 
            num_splits=self.num_splits
        )

        # -------------------------------------------------------------
        # Phase 4: 逆排序还原 (Unsort)
        # -------------------------------------------------------------
        # 将结果按照 Q 最初的顺序还原回去，保证输出序列的物理意义与输入 X 一致
        q_unsort_indices = torch.argsort(q_sorted_indices, dim=-1)
        final_out = out_sorted[batch_indices, q_unsort_indices]

        return final_out
    

# =====================================================================
# 原生 PyTorch 对照组 (严格遵循 Algorithm 1)
# =====================================================================
class TorchCrossMoEClusteredAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, num_clusters: int):
        super().__init__()
        self.M = num_clusters
        self.router_q = nn.Parameter(torch.randn(num_clusters, d_in) * (d_in ** -0.5))
        self.router_k = nn.Parameter(torch.randn(num_clusters, d_in) * (d_in ** -0.5))
        self.experts_q = nn.Parameter(torch.randn(num_clusters, d_in, d_out) * (d_in ** -0.5))
        self.experts_k = nn.Parameter(torch.randn(num_clusters, d_in, d_out) * (d_in ** -0.5))
        self.w_v = nn.Linear(d_in, d_out, bias=False)

    def forward(self, X: torch.Tensor):
            B, N, _ = X.shape
            d_out = self.experts_q.shape[-1]
            out = torch.zeros((B, N, d_out), device=X.device, dtype=X.dtype)
            sm_scale = 1.0 / (d_out ** 0.5)

            # -------------------------------------------------------------
            # Phase 1: MoE Routing & Projection
            # -------------------------------------------------------------
            logits_q = torch.matmul(X, self.router_q.transpose(0, 1))
            c_q = torch.argmax(logits_q, dim=-1) # [B, N]
            
            logits_k = torch.matmul(X, self.router_k.transpose(0, 1))
            c_k = torch.argmax(logits_k, dim=-1) # [B, N]

            # 显存安全的 Q 生成
            q = torch.zeros((B, N, d_out), device=X.device, dtype=X.dtype)
            for m in range(self.M):
                mask_q = (c_q == m) 
                if mask_q.any():
                    X_m_q = X[mask_q] 
                    W_m_q = self.experts_q[m] 
                    q[mask_q] = torch.matmul(X_m_q, W_m_q)
            
            # 显存安全的 K 生成 (修复处)
            k = torch.zeros((B, N, d_out), device=X.device, dtype=X.dtype)
            for m in range(self.M):
                mask_k = (c_k == m) 
                if mask_k.any():
                    X_m_k = X[mask_k] 
                    W_m_k = self.experts_k[m] 
                    k[mask_k] = torch.matmul(X_m_k, W_m_k)

            # -------------------------------------------------------------
            # Phase 2: Value Projection
            # -------------------------------------------------------------
            v = self.w_v(X)

            # -------------------------------------------------------------
            # Phase 3: Bucket Construction & Computation (串行之源)
            # -------------------------------------------------------------
            for b in range(B):
                for i in range(N):
                    cluster = c_q[b, i]
                    mask = (c_k[b] == cluster)
                    
                    if not mask.any():
                        continue
                    
                    # 动态长度截取
                    k_local = k[b, mask]
                    v_local = v[b, mask]
                    
                    scores = torch.matmul(q[b, i:i+1], k_local.transpose(0, 1)) * sm_scale
                    attn = torch.softmax(scores, dim=-1)
                    out[b, i:i+1] = torch.matmul(attn, v_local)
                    
            return out


# =====================================================================
# 性能压测与校验
# =====================================================================
def run_benchmark():
    B = 32          # Batch Size
    N = 512        # Sequence Length
    D_IN = 512
    D_OUT = 512
    M = 4          # Clusters 数量
    num_splits = 16 # Split-K 并发度
    device = torch.device('cuda')

    print(f"初始化环境... [B={B}, N={N}, D_in={D_IN}, D_out={D_OUT}, M={M}]")
    X = torch.randn((B, N, D_IN), dtype=torch.float32, device=device)

    # 1. 实例化模型
    triton_model = CrossMoEClusteredAttention(D_IN, D_OUT, M, num_splits).to(device)
    torch_model = TorchCrossMoEClusteredAttention(D_IN, D_OUT, M).to(device)

    # 2. 强制对齐权重，确保精度对比公平
    with torch.no_grad():
        torch_model.router_q.copy_(triton_model.router_q)
        torch_model.router_k.copy_(triton_model.router_k)
        torch_model.experts_q.copy_(triton_model.experts_q)
        torch_model.experts_k.copy_(triton_model.experts_k)
        torch_model.w_v.weight.copy_(triton_model.w_v.weight)

    triton_model.eval()
    torch_model.eval()

    # 3. 端到端精度校验
    print("正在进行精度校验...")
    with torch.no_grad():
        out_torch = torch_model(X)
        out_triton = triton_model(X)
        
    assert torch.allclose(out_torch, out_triton, atol=1e-3), "端到端精度校验失败！"
    print("✅ 精度校验通过！Triton 版本输出与原生 PyTorch 完全一致。\n")

    # 4. 分阶段微基准测试 (Micro-benchmarking)
    print("-" * 50)
    print(f"{'Phase (阶段)':<30} | {'PyTorch (ms)':<15} | {'Triton (ms)':<15}")
    print("-" * 50)
    
    quantiles = [0.5, 0.2, 0.8]

    # --- Phase 1: 路由与重排 ---
    def torch_phase1():
        logits_q = torch.matmul(X, torch_model.router_q.transpose(0, 1))
        c_q = torch.argmax(logits_q, dim=-1)
        # 使用显存安全的循环写法替换原本的 OOM 写法
        q = torch.zeros((B, N, D_OUT), device=X.device, dtype=X.dtype)
        for m in range(M):
            mask_q = (c_q == m)
            if mask_q.any():
                q[mask_q] = torch.matmul(X[mask_q], torch_model.experts_q[m])
        return q

    def triton_phase1():
        return triton_moe_router_and_project(X, triton_model.router_q, triton_model.experts_q)

    t_torch_p1, _, _ = triton.testing.do_bench(torch_phase1, quantiles=quantiles)
    t_triton_p1, _, _ = triton.testing.do_bench(triton_phase1, quantiles=quantiles)
    print(f"{'1. MoE Routing & Projection':<30} | {t_torch_p1:<15.4f} | {t_triton_p1:<15.4f}")

    # --- Phase 2: Value 映射 ---
    # (此阶段两者逻辑基本一致，仅 Triton 多了打乱开销，通常极小)
    k_sorted, _, k_cu_seqlens, k_sorted_indices = triton_phase1()
    q_sorted, q_cluster_ids_orig, _, q_sorted_indices = triton_moe_router_and_project(X, triton_model.router_q, triton_model.experts_q)
    batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
    
    def torch_phase2():
        return torch_model.w_v(X)
        
    def triton_phase2():
        V = triton_model.w_v(X)
        return V[batch_indices, k_sorted_indices]

    t_torch_p2, _, _ = triton.testing.do_bench(torch_phase2, quantiles=quantiles)
    t_triton_p2, _, _ = triton.testing.do_bench(triton_phase2, quantiles=quantiles)
    print(f"{'2. Value Projection & Align':<30} | {t_torch_p2:<15.4f} | {t_triton_p2:<15.4f}")

    # --- Phase 3 & 4: Clustered Attention + Unsort ---
    q_torch = torch_phase1()
    k_torch = torch.zeros((B, N, D_OUT), device=X.device, dtype=X.dtype)
    v_torch = torch_phase2()
    c_q_torch = torch.argmax(torch.matmul(X, torch_model.router_q.transpose(0, 1)), dim=-1)
    c_k_torch = torch.argmax(torch.matmul(X, torch_model.router_k.transpose(0, 1)), dim=-1)
    
    for m in range(M):
        mask_k = (c_k_torch == m)
        if mask_k.any():
            k_torch[mask_k] = torch.matmul(X[mask_k], torch_model.experts_k[m])


    def torch_phase3():
        # 极度缓慢的 Python 双重 for 循环
        out = torch.zeros_like(q_torch)
        sm_scale = 1.0 / (D_OUT ** 0.5)
        for b in range(B):
            for i in range(N):
                mask = (c_k_torch[b] == c_q_torch[b, i])
                if mask.any():
                    attn = torch.softmax(torch.matmul(q_torch[b, i:i+1], k_torch[b, mask].transpose(0, 1)) * sm_scale, dim=-1)
                    out[b, i:i+1] = torch.matmul(attn, v_torch[b, mask])
        return out

    def triton_phase3_4():
        v_sorted = triton_phase2()
        q_cluster_ids_sorted = q_cluster_ids_orig[batch_indices, q_sorted_indices]
        out_sorted = batched_flash_decoding_attention(q_sorted, k_sorted, v_sorted, q_cluster_ids_sorted, k_cu_seqlens, num_splits=num_splits)
        q_unsort_indices = torch.argsort(q_sorted_indices, dim=-1)
        return out_sorted[batch_indices, q_unsort_indices]

    t_torch_p3, _, _ = triton.testing.do_bench(torch_phase3, quantiles=quantiles)
    t_triton_p3, _, _ = triton.testing.do_bench(triton_phase3_4, quantiles=quantiles)
    print(f"{'3. Attention & Final Unsort':<30} | {t_torch_p3:<15.4f} | {t_triton_p3:<15.4f}")

    print("-" * 50)
    print(f"{'Total End-to-End':<30} | {t_torch_p1 + t_torch_p2 + t_torch_p3:<15.4f} | {t_triton_p1 + t_triton_p2 + t_triton_p3:<15.4f}")
    speedup = (t_torch_p1 + t_torch_p2 + t_torch_p3) / (t_triton_p1 + t_triton_p2 + t_triton_p3)
    print(f"🔥 Triton 整体加速比: {speedup:.2f}x")

if __name__ == '__main__':
    run_benchmark()