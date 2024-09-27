import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取 CSV 文件
row_data = pd.read_csv('./files/RowLatency.csv', header=None, names=['batch_size', 'length', 'comm_latency_row', 'comp_latency_row'])
col_data = pd.read_csv('./files/ColLatency.csv', header=None, names=['batch_size', 'length', 'comm_latency_col', 'comp_latency_col'])

# 合并 RowLatency 和 ColLatency 数据
merged_data = pd.merge(row_data, col_data, on=['batch_size', 'length'])

# 计算 row 的总延迟
merged_data['col_total_latency'] = merged_data['comm_latency_col'] + merged_data['comp_latency_col']

# 取出 batch size = 1, length = 16 时的 row 总延迟作为归一化基准
normalization_factor = merged_data.loc[(merged_data['batch_size'] == 1) & (merged_data['length'] == 64), 'col_total_latency'].values[0]

# 将 row 和 col 的延迟归一化为相对于 batch size = 1, length = 16 时 row 总延迟的倍数
merged_data['comm_latency_row_norm'] = merged_data['comm_latency_row'] / normalization_factor
merged_data['comp_latency_row_norm'] = merged_data['comp_latency_row'] / normalization_factor
merged_data['comm_latency_col_norm'] = merged_data['comm_latency_col'] / normalization_factor
merged_data['comp_latency_col_norm'] = merged_data['comp_latency_col'] / normalization_factor

# 绘制归一化后的延迟图
def plot_normalized_latency(merged_data):
    # 生成不同 batch size 和 length 的标签
    labels = [f'BS: {bs}, Len: {length}' for bs, length in zip(merged_data['batch_size'], merged_data['length'])]

    # 各种延迟
    comm_latency_row = merged_data['comm_latency_row_norm']
    comp_latency_row = merged_data['comp_latency_row_norm']
    comm_latency_col = merged_data['comm_latency_col_norm']
    comp_latency_col = merged_data['comp_latency_col_norm']

    bar_width = 0.35  # 柱状图的宽度
    index = np.arange(len(labels))  # 标签的位置

    fig, ax = plt.subplots(figsize=(6, 4))

    # 设置颜色
    row_color = '#D2691E'  # Row 的颜色
    row_color2 = '#B8860B'  # Col 的颜色（类似 Row 的颜色）
    col_color = '#BC8F8F'  # 另一个类似的颜色
    col_color2 = '#8B4513'  # 另一个类似的颜色

    # 绘制 row 的堆叠柱状图
    row_comm_bars = ax.bar(index + bar_width / 2, comm_latency_row, bar_width, label='Communication (Row)', color=row_color, alpha=0.7)
    row_comp_bars = ax.bar(index + bar_width / 2, comp_latency_row, bar_width, bottom=comm_latency_row, label='Computation (Row)', color=row_color2)

    # 绘制 col 的堆叠柱状图
    col_comm_bars = ax.bar(index - bar_width / 2, comm_latency_col, bar_width, label='Communication (Col)', color=col_color, alpha=0.7)
    col_comp_bars = ax.bar(index - bar_width / 2, comp_latency_col, bar_width, bottom=comm_latency_col, label='Computation (Col)', color=col_color2)

    # 添加标签、标题和自定义的 x 轴标签
    #ax.set_xlabel('Batch Size and Length',fontsize=14)
    ax.set_ylabel('Normalized Performance',fontsize=14)
    # 设置对数刻度
    ax.set_yscale('log')

    ax.set_xticks(index)
    ax.set_xticklabels(labels, rotation=45, ha="center",fontsize=12)
    
    ax.legend(loc='center', bbox_to_anchor=(0.5, 1.12), ncol=2, prop={'size': 13}, frameon=False)
    

    fig.tight_layout()
    plt.show()
    plt.savefig('Compare.png', dpi=500)

# 调用绘图函数
plot_normalized_latency(merged_data)