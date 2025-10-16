import torch

# batch_size=4, seq_len=5
mask = torch.tensor([
    [True,  False, True,  False, False],  # batch 0: 有True值
    [False, False, False, False, False],  # batch 1: 全False
    [True,  True,  False, False, False],  # batch 2: 有True值  
    [False, False, False, False, False]   # batch 3: 全False
])

print(mask.any(dim=1))  # tensor([True, False, True, False])
print(torch.where(mask.any(dim=1)))  # tensor([0, 2])

mask = torch.tensor([0,  1, 0,  1, 0])

print(mask.nonzero(as_tuple=False))

import torch

# 让我用一个具体的例子来解释这个语法
def explain_advanced_indexing():
    # 假设我们有以下情况
    batch_size = 2
    seq_len_q = 5  # 查询序列长度
    seq_len_k = 6  # 键序列长度
    
    # 创建一个注意力权重矩阵 [batch_size, seq_len_q, seq_len_k]
    attention_weights = torch.zeros(batch_size, seq_len_q, seq_len_k)
    print("原始attention_weights形状:", attention_weights.shape)
    print("原始矩阵 (batch=0):")
    print(attention_weights[0])
    print()
    
    # 假设在某个簇中，我们有以下查询和键的索引
    b = 0  # 当前batch
    q_indices = torch.tensor([1, 3, 4])  # 形状: [3]
    k_indices = torch.tensor([0, 2, 5])  # 形状: [3]
    attn_probs = torch.tensor([[0.4, 0.3, 0.3],    # 形状: [3, 3]
                               [0.2, 0.5, 0.3],
                               [0.1, 0.1, 0.8]])
    
    print("q_indices:", q_indices, "形状:", q_indices.shape)
    print("k_indices:", k_indices, "形状:", k_indices.shape)
    print("attn_probs形状:", attn_probs.shape)
    print("attn_probs:")
    print(attn_probs)
    print()
    
    # 关键语法解释: q_indices[:, None] 和 k_indices[None, :]
    q_expanded = q_indices[:, None]  # 形状从 [3] 变成 [3, 1]
    k_expanded = k_indices[None, :]  # 形状从 [3] 变成 [1, 3]
    
    print("q_indices[:, None] (添加新的列维度):")
    print(q_expanded, "形状:", q_expanded.shape)
    print()
    print("k_indices[None, :] (添加新的行维度):")
    print(k_expanded, "形状:", k_expanded.shape)
    print()
    
    # Broadcasting机制解释
    # 当我们使用 attention_weights[b, q_expanded, k_expanded] 时
    # PyTorch会对 q_expanded [3,1] 和 k_expanded [1,3] 进行广播
    # 结果是选择所有 (q_idx, k_idx) 的组合
    
    print("广播后的索引组合 (所有query-key对):")
    for i in range(len(q_indices)):
        for j in range(len(k_indices)):
            q_idx = q_indices[i].item()
            k_idx = k_indices[j].item()
            prob = attn_probs[i, j].item()
            print(f"  位置 ({q_idx}, {k_idx}) <- 权重 {prob:.1f}")
    print()
    
    # 执行实际的赋值操作
    attention_weights[b, q_indices[:, None], k_indices[None, :]] = attn_probs
    
    print("赋值后的attention_weights (batch=0):")
    print(attention_weights[0])
    print()
    
    # 验证结果
    print("验证几个具体位置:")
    print(f"位置 (1,0): {attention_weights[0, 1, 0].item():.1f} (应该是0.4)")
    print(f"位置 (3,2): {attention_weights[0, 3, 2].item():.1f} (应该是0.5)")
    print(f"位置 (4,5): {attention_weights[0, 4, 5].item():.1f} (应该是0.8)")
    print(f"位置 (0,0): {attention_weights[0, 0, 0].item():.1f} (应该是0.0，未被赋值)")

# 进一步的语法解释
def syntax_breakdown():
    print("\n" + "="*60)
    print("语法详细分解:")
    print("="*60)
    
    print("1. q_indices[:, None] 的作用:")
    print("   - 原始: [a, b, c]  形状: [3]")
    print("   - 结果: [[a],      形状: [3, 1]")
    print("           [b],")
    print("           [c]]")
    print()
    
    print("2. k_indices[None, :] 的作用:")
    print("   - 原始: [x, y, z]  形状: [3]")
    print("   - 结果: [[x, y, z]] 形状: [1, 3]")
    print()
    
    print("3. Broadcasting机制:")
    print("   - [3, 1] 和 [1, 3] 广播后变成 [3, 3]")
    print("   - [[a],    和  [[x, y, z]]  =>  选择位置:")
    print("     [b],                          (a,x) (a,y) (a,z)")
    print("     [c]]                          (b,x) (b,y) (b,z)")
    print("                                   (c,x) (c,y) (c,z)")
    print()
    
    print("4. 等价的传统写法 (效率较低):")
    print("   for i, q_idx in enumerate(q_indices):")
    print("       for j, k_idx in enumerate(k_indices):")
    print("           attention_weights[b, q_idx, k_idx] = attn_probs[i, j]")

if __name__ == "__main__":
    explain_advanced_indexing()
    syntax_breakdown()