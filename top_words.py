import re

def text_to_markdown_table(input_text, dataset_names=None):
    """
    将文本转换为Markdown表格
    
    参数:
    input_text (str): 输入的文本数据
    dataset_names (list): 长度为9的数据集名称列表，用于填充第一列
    
    返回:
    str: Markdown格式的表格代码
    """
    # 分割输入文本为行
    lines = input_text.strip().split('\n')
    
    # 处理每一行数据
    processed_rows = []
    for line in lines:
        # 使用正则表达式提取所有项目（移除数字序号）
        items = re.findall(r'\d+\.\s*(.*?)(?=\s+\d+\.|$)', line)
        
        # 确保每行有10个元素（不足的用空字符串填充）
        items = items + [''] * (10 - len(items)) if len(items) < 10 else items[:10]
        
        # 添加到处理后的行列表
        processed_rows.append(items)
    
    # 验证数据集名称参数
    if dataset_names is None:
        dataset_names = [''] * 9  # 默认为空
    elif len(dataset_names) < 9:
        # 不足9个时用空字符串填充
        dataset_names = dataset_names + [''] * (9 - len(dataset_names))
    elif len(dataset_names) > 9:
        # 超过9个时截断
        dataset_names = dataset_names[:9]
    
    # 创建Markdown表格
    markdown_lines = [
        "| Dataset Name | 1st | 2ed | 3rd | 4th | 5th | 6th | 7th | 8th | 9th | 10th |",
        "|--------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|----------|"
    ]
    
    for i, row in enumerate(processed_rows):
        # 获取当前行的数据集名称（如果存在）
        name = dataset_names[i] if i < len(dataset_names) else ''
        
        # 构建表格行
        markdown_line = f"| {name.ljust(13)} | " + " | ".join(row) + " |"
        markdown_lines.append(markdown_line)
    
    return "\n".join(markdown_lines)

# 示例使用
if __name__ == "__main__":
    input_text = """
1. source 2. predictions 3. agricultural 4. mount 5. chicken 6. official 7. com 8. market 9. ##gate 10. end
1. source 2. ##imate 3. as 4. ant 5. available 6. prompt
1. policy 2. states 3. japan 4. bilateral 5. 1994 6. current 7. january 8. ##do 9. deficit 10. org
1. prompt 2. source 3. prices 4. predictions 5. oil 6. gov 7. about 8. crude 9. facts 10. 02
1. new 2. air 3. york 4. state 5. source 6. quality 7. has 8. environmental 9. follows 10. gov
1. influenza 2. source 3. virus 4. states 5. gov 6. ##m 7. cdc 8. follows 9. pub 10. that
1. response 2. data 3. states 4. register 5. ##liest 6. ##bula 7. three 8. oklahoma 9. gov 10. federal
1. unemployment 2. rate 3. source 4. unemployed 5. 4 6. ##s 7. us 8. b 9. based 10. november
1. future 2. volume 3. source 4. 05 5. prompt 6. ##mo 7. greenhouse 8. lifetime 9. traffic 10. provides
"""

    input_text = """
1. prompt 2. 2016 3. according 4. agricultural 5. predictions 6. start 7. per 8. source 9. follows 10. ##w
1. ed 2. based 3. 48 4. near 5. across 6. conditions 7. ##dent
1. trade 2. predictions 3. source 4. future 5. restrictions 6. org 7. japan 8. states 9. gov 10. reduce
1. source 2. predictions 3. oil 4. prices 5. gov 6. crude 7. natural 8. price 9. 2021 10. rising
1. air 2. quality 3. source 4. new 5. york 6. gov 7. epa 8. environmental 9. state 10. water
1. based 2. information 3. predictions 4. influenza 5. ##bi 6. source 7. ni 8. infection 9. various 10. states
1. follows 2. help 3. ##a 4. federal 5. source 6. state 7. key 8. 5 9. ##lian 10. earthquakes
1. ni 2. ep 3. gov 4. 6 5. unemployed 6. information 7. 12 8. source 9. unable 10. ##ls
1. source 2. volume 3. gov 4. united 5. tr 6. com 7. high 8. data 9. vehicle 10. michigan
"""

    # 示例数据集名称（长度为9的列表）
    dataset_names = ["Agriculture", "Climate", "Economy", "Energy", "Environment", "Public Health", "Security", "SocialGood", "Traffic"]
    
    markdown_table = text_to_markdown_table(input_text, dataset_names)
    print(markdown_table)