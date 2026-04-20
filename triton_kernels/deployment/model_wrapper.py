import torch
import time

# 假设这是你的 Triton Kernel 包装函数 (你需要将其替换为你的实际调用)
# from my_triton_kernel import fused_moe_attn_triton

def dummy_triton_attn(q, k, v):
    # 这里暂时用原生的模拟一下，等你接入真正的 Triton kernel
    # 实际上这里应该是： return fused_moe_attn_triton(q, k, v)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)

class MoEAttnEngine:
    def __init__(self, device='cuda', dtype=torch.float16):
        self.device = device
        self.dtype = dtype

    @torch.inference_mode()
    def forward_native(self, q, k, v):
        """原生 PyTorch 引擎 (模拟回退/基线)"""
        # 计算注意力分数 L x N
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        # 计算输出
        out = torch.matmul(attn_weights, v)
        return out

    @torch.inference_mode()
    def forward_triton(self, q, k, v):
        """Triton 优化引擎"""
        # 调用你的 Triton kernel
        out = dummy_triton_attn(q, k, v)
        return out