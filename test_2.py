import torch
import torch.nn as nn

def demonstrate_expert_assignment():
    """
    演示专家分配机制中mask、where、nonzero的工作原理
    """
    # 设置参数
    batch_size = 2
    seq_len_q = 4  # 查询序列长度
    seq_len_k = 3  # 键序列长度
    d_model = 8    # 特征维度
    M = 3          # 专家数量
    
    # 模拟数据
    torch.manual_seed(42)
    device = 'cpu'
    
    # 创建输入张量 (Q和K拼接后的结果)
    x = torch.randn(batch_size, seq_len_q + seq_len_k, d_model)
    print("输入张量 x 的形状:", x.shape)
    print("=" * 80)
    
    # 模拟专家分配结果
    # assignments[b, i] 表示第b个batch中第i个token被分配给哪个专家
    assignments = torch.tensor([
        [0, 1, 0, 2, 1, 2, 0],  # batch 0: 前4个是Q，后3个是K
        [1, 1, 2, 0, 2, 0, 1]   # batch 1: 前4个是Q，后3个是K
    ])
    print("专家分配 assignments:")
    print(assignments)
    print("  - 每行代表一个batch")
    print("  - 前4个位置是Query tokens，后3个是Key tokens")
    print("  - 数值表示分配给哪个专家(0, 1, 2)")
    print("=" * 80)
    
    # 创建简单的专家网络
    experts_Q = nn.ModuleList([
        nn.Linear(d_model, d_model) for _ in range(M)
    ])
    
    # 初始化输出张量
    x_transformed = torch.zeros_like(x)
    
    print("\n开始处理Query tokens (前seq_len_q个):")
    print("-" * 80)
    
    for m in range(M):
        print(f"\n处理专家 {m}:")
        
        # ========== 关键步骤1: 创建mask ==========
        # mask的两个条件：
        # 1. (assignments == m): 找出分配给专家m的tokens
        # 2. (torch.arange(...) < seq_len_q): 确保是Query tokens（前4个）
        
        # 创建位置索引
        position_indices = torch.arange(seq_len_q + seq_len_k, device=device)
        print(f"  位置索引: {position_indices.tolist()}")
        
        # 条件1: 哪些token分配给了专家m
        expert_mask = (assignments == m)
        print(f"  专家{m}的分配mask (assignments == {m}):")
        print(f"    {expert_mask}")
        
        # 条件2: 哪些是Query tokens（位置 < seq_len_q）
        query_mask = position_indices < seq_len_q
        print(f"  Query位置mask (position < {seq_len_q}):")
        print(f"    {query_mask.tolist()}")
        
        # 组合条件：既分配给专家m，又是Query token
        mask = expert_mask & query_mask.unsqueeze(0)  # 广播到batch维度
        print(f"  最终mask (专家{m} & Query):")
        print(f"    {mask}")
        
        # ========== 关键步骤2: torch.where和mask.any(dim=1) ==========
        # mask.any(dim=1) 检查每个batch是否有至少一个True
        has_tokens = mask.any(dim=1)
        print(f"  每个batch是否有token: {has_tokens.tolist()}")
        
        # torch.where找出哪些batch有token需要处理
        active_batches = torch.where(has_tokens)[0]
        # print(f"  需要处理的batch索引: {torch.where(has_tokens)}")
        print(f"  需要处理的batch索引: {active_batches.tolist()}")
        
        # ========== 关键步骤3: nonzero获取具体位置 ==========
        for b in active_batches:
            # nonzero返回所有True值的位置
            indices = mask[b].nonzero(as_tuple=True)[0]
            print(f"  1  Batch {b.item()}中的token位置: {mask[b].nonzero(as_tuple=False)[0]}")
            print(f"  2  Batch {b.item()}中的token位置: {indices.tolist()}")
            
            # 应用专家变换
            if len(indices) > 0:
                # 取出对应位置的tokens
                tokens_to_transform = x[b, indices]
                print(f"      输入形状: {tokens_to_transform.shape}")
                
                # 应用专家网络
                transformed = experts_Q[m](tokens_to_transform)
                print(f"      输出形状: {transformed.shape}")
                
                # 将结果写回对应位置
                x_transformed[b, indices] = transformed
                print(f"      已将变换结果写入位置 {indices.tolist()}")
    
    print("\n" + "=" * 80)
    print("处理完成！")
    print(f"最终 x_transformed 形状: {x_transformed.shape}")
    
    # 验证：检查哪些位置被处理了
    print("\n验证结果:")
    processed_mask = (x_transformed != 0).any(dim=-1)
    for b in range(batch_size):
        print(f"Batch {b} 被处理的位置: {processed_mask[b].nonzero().squeeze().tolist()}")
    
    return x_transformed, assignments


def step_by_step_explanation():
    """
    逐步解释mask操作的细节
    """
    print("\n" + "=" * 80)
    print("深入理解Mask操作")
    print("=" * 80)
    
    # 简化示例
    assignments = torch.tensor([[0, 1, 0, 2, 1]])  # 1个batch，5个tokens
    seq_len_q = 3  # 前3个是Query
    m = 0  # 查看专家0
    
    print(f"assignments: {assignments[0].tolist()}")
    print(f"  解释: token 0分配给专家0, token 1分配给专家1, token 2分配给专家0...")
    print(f"\nseq_len_q = {seq_len_q} (前3个是Query tokens)")
    print(f"当前处理专家 m = {m}")
    
    # Step 1: 创建位置索引
    position = torch.arange(5)
    print(f"\n1. 位置索引: {position.tolist()}")
    
    # Step 2: 找出分配给专家0的tokens
    expert_mask = (assignments == m)
    print(f"\n2. 专家{m}的mask: {expert_mask[0].tolist()}")
    print(f"   解释: True表示该位置分配给专家{m}")
    
    # Step 3: 找出Query tokens
    query_mask = position < seq_len_q
    print(f"\n3. Query mask: {query_mask.tolist()}")
    print(f"   解释: True表示该位置是Query token (位置 < {seq_len_q})")
    
    # Step 4: 组合条件
    final_mask = expert_mask & query_mask
    print(f"\n4. 最终mask: {final_mask[0].tolist()}")
    print(f"   解释: True表示该位置既是Query token，又分配给专家{m}")
    
    # Step 5: 找出True的位置
    indices = final_mask[0].nonzero(as_tuple=True)[0]
    print(f"\n5. nonzero结果: {indices.tolist()}")
    print(f"   解释: 位置0和位置2满足条件（是Query且分配给专家0）")
    
    # 可视化
    print("\n可视化总结:")
    print("-" * 50)
    print("位置:      ", list(range(5)))
    print("Token类型: ", ["Q", "Q", "Q", "K", "K"])
    print("分配专家:  ", assignments[0].tolist())
    print(f"专家{m}处理:", ["✓" if i in indices else " " for i in range(5)])


# 运行演示
if __name__ == "__main__":
    # 运行主要演示
    x_transformed, assignments = demonstrate_expert_assignment()
    
    # 运行详细解释
    step_by_step_explanation()