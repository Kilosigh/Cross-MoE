import os

def generate_pixel_perfect_architecture():
    """彻底剥离强制宽度，利用纯数学计算实现绝对对齐的终极架构图"""
    
    dot_code = """
digraph SystemArchitecture {
    rankdir=TB;
    splines=none; 
    nodesep=0.5;
    ranksep=0.3;  
    fontname="SimHei";

    // 消除 node 默认的 margin
    node [shape=none, margin=0, fontname="SimHei"];

    // ==========================================
    // 1. 表现层
    // 注意：绝对不能给 TABLE 加 WIDTH 属性，纯靠内部 TD 的原生宽度撑开
    // ==========================================
    Presentation [label=<
      <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="10" CELLPADDING="12" BGCOLOR="#e8f4f8" STYLE="ROUNDED">
        <TR>
          <TD BGCOLOR="#e2e8f0" WIDTH="150" STYLE="ROUNDED" ALIGN="CENTER" VALIGN="MIDDLE"><B>表现层</B></TD>
          <TD BGCOLOR="white" STYLE="ROUNDED" BORDER="1" COLOR="#94a3b8" WIDTH="210">UI (Web可视化前端)</TD>
          <TD BGCOLOR="white" STYLE="ROUNDED" BORDER="1" COLOR="#94a3b8" WIDTH="210">数据资产与图表渲染</TD>
          <TD BGCOLOR="white" STYLE="ROUNDED" BORDER="1" COLOR="#94a3b8" WIDTH="210">外部因素冲击模拟表单</TD>
          <TD BGCOLOR="white" STYLE="ROUNDED" BORDER="1" COLOR="#94a3b8" WIDTH="210">多模态注意力热图呈现</TD>
        </TR>
      </TABLE>
    >];

    // ==========================================
    // 2. 业务调度与控制层
    // ==========================================
    Logic [label=<
      <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="10" CELLPADDING="12" BGCOLOR="#f0fdf4" STYLE="ROUNDED">
        <TR>
          <TD ROWSPAN="2" BGCOLOR="#e2e8f0" WIDTH="150" STYLE="ROUNDED" ALIGN="CENTER" VALIGN="MIDDLE">
            <B>业务调度与</B><BR ALIGN="CENTER"/>
            <FONT POINT-SIZE="8"> </FONT><BR ALIGN="CENTER"/>
            <B>控制层</B>
          </TD>
          <TD COLSPAN="2" BGCOLOR="white" STYLE="ROUNDED" BORDER="1" COLOR="#86efac" WIDTH="870">API网关 (统一请求分发)</TD>
        </TR>
        <TR>
          <TD BGCOLOR="white" STYLE="ROUNDED" BORDER="1" COLOR="#94a3b8" WIDTH="430">数据对齐与预处理控制器</TD>
          <TD BGCOLOR="white" STYLE="ROUNDED" BORDER="1" COLOR="#94a3b8" WIDTH="430">异步推理任务队列</TD>
        </TR>
      </TABLE>
    >];

    // ==========================================
    // 3. 模型推理层
    // ==========================================
    Inference [label=<
      <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="10" CELLPADDING="12" BGCOLOR="#faf5ff" STYLE="ROUNDED">
        <TR>
          <TD ROWSPAN="3" BGCOLOR="#e2e8f0" WIDTH="150" STYLE="ROUNDED" ALIGN="CENTER" VALIGN="MIDDLE"><B>模型推理层</B></TD>
          <TD BGCOLOR="white" STYLE="ROUNDED" BORDER="1" COLOR="#94a3b8" WIDTH="430">PyTorch推理引擎</TD>
          <TD BGCOLOR="white" STYLE="ROUNDED" BORDER="1" COLOR="#94a3b8" WIDTH="430">LLM文本特征提取器</TD>
        </TR>
        <TR>
          <TD COLSPAN="2" BGCOLOR="#f3e8ff" STYLE="ROUNDED" BORDER="1" COLOR="#d8b4fe" WIDTH="870">Cross-MoE 多模态注意力融合网络</TD>
        </TR>
        <TR>
          <TD COLSPAN="2" BGCOLOR="#f1f5f9" STYLE="ROUNDED" BORDER="1" COLOR="#94a3b8" WIDTH="870">底层 Triton 自定义算子 (GPU Kernel)</TD>
        </TR>
      </TABLE>
    >];

    // ==========================================
    // 4. 数据存储层
    // ==========================================
    Data [label=<
      <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="10" CELLPADDING="12" BGCOLOR="#fff7ed" STYLE="ROUNDED">
        <TR>
          <TD BGCOLOR="#e2e8f0" WIDTH="150" STYLE="ROUNDED" ALIGN="CENTER" VALIGN="MIDDLE">
            <B>数据存储与</B><BR ALIGN="CENTER"/>
            <FONT POINT-SIZE="8"> </FONT><BR ALIGN="CENTER"/>
            <B>访问层</B>
          </TD>
          <TD BGCOLOR="white" STYLE="ROUNDED" BORDER="1" COLOR="#94a3b8" WIDTH="430">关系型数据库元数据管理</TD>
          <TD BGCOLOR="white" STYLE="ROUNDED" BORDER="1" COLOR="#94a3b8" WIDTH="430">列式二进制存储 (多模态对齐特征)</TD>
        </TR>
      </TABLE>
    >];

    // 强制堆叠顺序
    Presentation -> Logic -> Inference -> Data [style=invis, weight=10];
}
"""
    
    dot_filename = "pixel_perfect_architecture.dot"
    png_filename = "pixel_perfect_architecture.png"
    
    with open(dot_filename, "w", encoding="utf-8") as f:
        f.write(dot_code)

    print(f"原生精准代码已生成: {dot_filename}")
    print("正在渲染...")
    
    exit_code = os.system(f"dot -Tpng {dot_filename} -o {png_filename}")
    
    if exit_code == 0:
        print(f"渲染成功！请查看: {png_filename}。剥离了错误的伸缩约束后，现在的右侧组件绝对是像刀切一样的方阵。")
    else:
        print("渲染失败。请检查 Graphviz 环境。")

if __name__ == "__main__":
    generate_pixel_perfect_architecture()