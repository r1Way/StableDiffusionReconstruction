import matplotlib.pyplot as plt
import numpy as np

# 定义网络结构的参数
rois = ['V1', 'V2', 'V3', 'V4', 'V8', 'FFC', 'VMV1', 'VMV2', 'VMV3', 'VVC']
roi_out_dim = 20
gcn_hidden1 = 64
gcn_hidden2 = 128
fc1 = 512
fc2 = 1024
out_dim = 6400


# 绘制网络结构
def draw_network(rois, roi_out_dim, gcn_hidden1, gcn_hidden2, fc1, fc2, out_dim):
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制每个脑区的CNN层
    for i, roi in enumerate(rois):
        ax.text(0, i, f'CNN_{roi}\n({roi_out_dim}D)', fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

    # 绘制图卷积层
    ax.text(1, len(rois) / 2, f'GCNConv1\n({gcn_hidden1}D)', fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
    ax.text(2, len(rois) / 2, f'GCNConv2\n({gcn_hidden2}D)', fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

    # 绘制全连接层
    ax.text(3, len(rois) / 2, f'FC1\n({fc1}D)', fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
    ax.text(4, len(rois) / 2, f'FC2\n({fc2}D)', fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
    ax.text(5, len(rois) / 2, f'Output\n({out_dim}D)', fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

    # 绘制连接线
    for i in range(len(rois)):
        ax.plot([0, 1], [i, len(rois) / 2], 'k-')
    ax.plot([1, 2], [len(rois) / 2, len(rois) / 2], 'k-')
    ax.plot([2, 3], [len(rois) / 2, len(rois) / 2], 'k-')
    ax.plot([3, 4], [len(rois) / 2, len(rois) / 2], 'k-')
    ax.plot([4, 5], [len(rois) / 2, len(rois) / 2], 'k-')

    # 设置坐标轴范围
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, len(rois) + 1)

    # 隐藏坐标轴
    ax.axis('off')

    # 显示图形
    plt.show()


# 调用函数绘制网络结构
draw_network(rois, roi_out_dim, gcn_hidden1, gcn_hidden2, fc1, fc2, out_dim)