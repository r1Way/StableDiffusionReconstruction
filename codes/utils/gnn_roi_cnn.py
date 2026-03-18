# 本脚本实现：每个脑区先单独卷积（CNN），输出20维向量，然后所有脑区特征拼接后进行图卷积，最后通过若干全连接层输出6400维目标。
# 用法示例：
# python gnn_roi_cnn.py --subject subj01 --target init_latent --test

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import argparse
import time
from sklearn.metrics import mean_squared_error
import random
from himalaya.scoring import correlation_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class ROICNN(nn.Module):
    """
    对每个脑区单独做卷积（MLP），输出20维向量
    输入: [batch_size, n_voxels]
    输出: [batch_size, 20]
    """

    def __init__(self, in_dim, out_dim=20):
        super().__init__()
        # MLP结构，自动适应不同体素数量
        self.net = nn.Sequential(
            nn.Linear(in_dim, max(32, in_dim // 4)),  # 动态隐藏层
            nn.ReLU(),
            nn.Linear(max(32, in_dim // 4), out_dim)
        )

    def forward(self, x):
        # x: [batch_size, n_voxels]
        return self.net(x)


class FullModel(nn.Module):
    """
    1. 对每个脑区单独卷积（MLP），输出20维
    2. 拼接为[10, 20]，作为图节点特征
    3. 图卷积+池化
    4. 多层全连接输出6400维
    """

    # roi_in_dims 每个脑区的体素数量列表。例如 [500, 800, 600, ...]
    # dropout=0.3 Dropout 随机失活比例，用于防止过拟合。
    def __init__(self, roi_in_dims, roi_out_dim=20, gcn_hidden1=64, gcn_hidden2=128, fc1=512, fc2=1024, out_dim=6400,
                 dropout=0.3):
        super().__init__()
        self.n_rois = len(roi_in_dims)
        self.roi_in_dims = roi_in_dims  # 新增：保存每个脑区体素数量
        # 每个脑区一个MLP
        self.roi_cnns = nn.ModuleList([ROICNN(in_dim, roi_out_dim) for in_dim in roi_in_dims])
        # 图卷积层
        self.gcn1 = GCNConv(roi_out_dim, gcn_hidden1)
        self.gcn2 = GCNConv(gcn_hidden1, gcn_hidden2)
        self.dropout = dropout
        # 全连接层
        self.fc1 = nn.Linear(gcn_hidden2, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc_out = nn.Linear(fc2, out_dim)

    def forward(self, x, edge_index, batch):
        # batch.x为list[10],list[i]为维度为(batch_size*第i个roi的体素数量)的tensor，装有fMRI数据，可能就是这里出了问题。

        # n_voxels_i 表示第 i 个脑区（ROI）包含的体素（voxels）数量
        # x: list of [n_voxels_i] tensors, batch_size=batch.max()+1, n_rois=len(x) // batch_size
        # x是一个list，包含所有样本的所有脑区的体素数据，每个元素 shape 不同
        batch_size = batch.max().item() + 1
        n_rois = self.n_rois  # 脑区数量
        roi_feats = []
        # 遍历每个样本
        for b in range(batch_size):
            roi_feat_sample = []
            # 遍历每个脑区
            for i in range(n_rois):
                # 修正：正确提取当前 batch 的第 b 个样本的第 i 个脑区体素数据
                roi_x = x[i][b * self.roi_cnns[i].net[0].in_features: (b + 1) * self.roi_cnns[i].net[0].in_features]
                # 或者更通用写法（推荐）：
                # roi_x = x[i][b*roi_in_dims[i] : (b+1)*roi_in_dims[i]]
                roi_x = roi_x.view(1, -1)  # [1, n_voxels_i]
                roi_feat = self.roi_cnns[i](roi_x)  # [1, roi_out_dim] 用该脑区的 MLP 提取 20 维特征。
                roi_feat_sample.append(roi_feat)
            roi_feat_sample = torch.cat(roi_feat_sample, dim=0)  # [n_rois, roi_out_dim]
            roi_feats.append(roi_feat_sample)
            # 把当前样本所有脑区的特征拼成一个二维张量
        node_feats = torch.cat(roi_feats, dim=0)  # [batch_size*n_rois, roi_out_dim]
        # 把所有样本的所有脑区特征拼成一个大张量，作为图神经网络的节点特征输入。
        # 图卷积
        x = self.gcn1(node_feats, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # 池化
        x = global_mean_pool(x, batch)
        # 全连接
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc_out(x)
        return x


def build_edge_index(adj):
    """
    根据邻接矩阵构建edge_index
    """
    idx = np.where(adj > 0)
    edge_index = np.vstack(idx)
    return torch.tensor(edge_index, dtype=torch.long)


def build_custom_adj(rois):
    """
    构建自定义邻接矩阵：
    - 三个连通集
      [V1]
      [V2,V3,V4]
      ['V8','FFC','VMV1','VMV2','VMV3','VVC']
    - 连通集内部边权为3
    - 不同连通集之间边权为1
    - 自己到自己为0
    """
    n = len(rois)
    adj = np.ones((n, n), dtype=np.float32)
    np.fill_diagonal(adj, 0)
    # 连通集定义
    group1 = ['V1']
    group2 = ['V2', 'V3', 'V4']
    group3 = ['V8', 'FFC', 'VMV1', 'VMV2', 'VMV3', 'VVC']
    groups = [group1, group2, group3]
    # 连通集内部边权为3
    for group in groups:
        idxs = [rois.index(r) for r in group]
        for i in idxs:
            for j in idxs:
                if i != j:
                    adj[i, j] = 3
    np.fill_diagonal(adj, 0)
    return adj


def set_random_seeds(seed=42):
    """
    设置随机种子，保证可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True, help="subject name")
    parser.add_argument("--mridir", type=str, default='../../mrifeat/', help="fMRI feature dir")
    parser.add_argument("--target", type=str, required=True, help="预测目标，如c")
    parser.add_argument("--decoded_dir", type=str, default='../../decoded', help="预测结果保存目录")
    parser.add_argument("--test", action="store_true", help="测试模式，仅训练1个epoch，batch_size=4")
    args = parser.parse_args()

    subject = args.subject
    mridir = args.mridir
    target = args.target
    decoded_dir = args.decoded_dir
    rois = ['V1', 'V2', 'V3', 'V4', 'V8', 'FFC', 'VMV1', 'VMV2', 'VMV3', 'VVC']

    # 构建自定义邻接矩阵
    adj = build_custom_adj(rois)
    edge_index = build_edge_index(adj)

    # 读取每个脑区的原始体素数据 shape
    roi_in_dims = []  # list[10]，统计各个脑区的体素数量
    for roi in rois:
        betas_path = os.path.join(mridir, subject, f"{subject}_{roi}_betas_tr.npy")
        X = np.load(betas_path)  # [试次，roi体素数]
        roi_in_dims.append(X.shape[1])

    # 读取训练集
    X_list = []
    for roi in rois:
        betas_path = os.path.join(mridir, subject, f"{subject}_{roi}_betas_tr.npy")
        X = np.load(betas_path)  # shape: [n_train, n_voxels]
        X_list.append(X)
    X_train = np.concatenate(X_list, axis=1)  # [n_train, sum_voxels]
    n_train = X_train.shape[0]  # 24980
    # 对训练集做z-score标准化
    X_train_mean = X_train.mean(axis=0, keepdims=True)
    X_train_std = X_train.std(axis=0, keepdims=True) + 1e-6
    X_train = (X_train - X_train_mean) / X_train_std
    # 读取标签
    Y_train_path = os.path.join('../../nsdfeat/subjfeat/', f"{subject}_each_{target}_tr.npy")
    Y_train = np.load(Y_train_path).astype("float32")
    # out_dim = 6400
    out_dim = Y_train.shape[1] if Y_train.ndim == 2 else Y_train.reshape(n_train, -1).shape[1]

    # 读取测试集
    X_list_te = []  # list[10] list[i]为 ndarray[982,第i个roi的体素数量]
    for roi in rois:
        betas_te_path = os.path.join(mridir, subject, f"{subject}_{roi}_betas_ave_te.npy")
        X_te = np.load(betas_te_path)  # shape: [n_test, n_voxels]
        X_list_te.append(X_te)
    X_test = np.concatenate(X_list_te, axis=1)  # [n_test, sum_voxels]
    n_test = X_test.shape[0]  # 982
    # 对测试集做z-score标准化（用训练集均值和方差）
    X_test = (X_test - X_train_mean) / X_train_std
    Y_test_path = os.path.join('../../nsdfeat/subjfeat/', f"{subject}_ave_{target}_te.npy")
    Y_test = np.load(Y_test_path).astype("float32")

    # 查看 river
    y_train_mean = Y_train.mean()
    y_train_std = Y_train.std()
    y_test_mean = Y_test.mean()
    y_test_std = Y_test.std()
    # 支持--test参数，减少数据量用于网络结构检查
    if args.test:
        print("测试模式：仅使用前20个样本和标签")
        for i in range(len(X_list)):
            X_list[i] = X_list[i][:20]
        for i in range(len(X_list_te)):
            X_list_te[i] = X_list_te[i][:8]
        Y_train = Y_train[:20]
        Y_test = Y_test[:8]
        n_train = 20
        n_test = 8

    # 构建DataLoader
    # 训练集
    train_data_list = []
    for i in range(n_train):
        roi_xs = []  # list[10] list[i]为第i个roi的体素数据
        start = 0
        for dim in roi_in_dims:  # roi_in_dims = [4308, 2916, 2096, 1311, 330, 747, 324, 245, 203, 482]
            # 分割每个脑区的体素数据
            roi_xs.append(torch.tensor(X_train[i, start:start + dim], dtype=torch.float32))
            start += dim
        # roi_xs: list of [n_rois, n_voxels_i]，每个样本一个list
        y = torch.tensor(Y_train[i], dtype=torch.float32)  # y 为6400维
        # batch: [n_rois]，每个脑区属于同一个样本
        batch = torch.full((len(roi_xs),), i, dtype=torch.long)
        data = Data(x=roi_xs, edge_index=edge_index, y=y, batch=batch)
        train_data_list.append(data)

    # 测试集
    test_data_list = []
    for i in range(n_test):
        roi_xs = []
        start = 0
        for dim in roi_in_dims:
            roi_xs.append(torch.tensor(X_test[i, start:start + dim], dtype=torch.float32))
            start += dim
        y = torch.tensor(Y_test[i], dtype=torch.float32)
        batch = torch.full((len(roi_xs),), i, dtype=torch.long)
        data = Data(x=roi_xs, edge_index=edge_index, y=y, batch=batch)
        test_data_list.append(data)
    batch_size = 4 if args.test else 32
    train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=False)

    # 设置随机种子
    set_random_seeds(42)
    print("随机种子已设置为42")

    # 构建模型
    model = FullModel(
        roi_in_dims=roi_in_dims,
        roi_out_dim=20,
        gcn_hidden1=64,
        gcn_hidden2=128,
        fc1=512,
        fc2=1024,
        out_dim=out_dim,
        dropout=0.3
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # 训练
    model.train()
    n_epochs = 1 if args.test else 10
    for epoch in range(n_epochs):
        epoch_start = time.time()
        total_loss = 0
        print(f"Epoch {epoch + 1}/{n_epochs} 开始")
        for batch_idx, batch in enumerate(train_loader):
            batch_start = time.time()
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.batch)
            if batch.y.ndim == 1:
                batch_y = batch.y.view(-1, out_dim)
            else:
                batch_y = batch.y
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_end = time.time()
            print(
                f"  Batch {batch_idx + 1}/{len(train_loader)} Loss: {loss.item():.4f} 用时: {batch_end - batch_start:.2f}s")
        epoch_end = time.time()
        print(f"Epoch {epoch + 1} 总Loss: {total_loss / len(train_loader):.4f} 用时: {epoch_end - epoch_start:.2f}s")

        # 每个epoch后对训练集评估
        model.eval()
        train_preds = []
        train_trues = []
        with torch.no_grad():
            for batch in train_loader:
                pred = model(batch.x, batch.edge_index, batch.batch)
                train_preds.append(pred.cpu().numpy())
                if batch.y.ndim == 1:
                    train_trues.append(batch.y.cpu().numpy().reshape(-1, out_dim))
                else:
                    train_trues.append(batch.y.cpu().numpy())
        train_preds = np.vstack(train_preds)
        train_trues = np.vstack(train_trues)
        train_mse = mean_squared_error(train_trues, train_preds)
        train_corr = np.corrcoef(train_trues.flatten(), train_preds.flatten())[0, 1]
        train_rs = correlation_score(train_trues.T, train_preds.T)
        train_rs_mean = np.mean(train_rs)
        print(
            f"[Train] Epoch {epoch + 1}: MSE={train_mse:.4f}, 相关系数={train_corr:.4f}, correlation_score均值={train_rs_mean:.4f}")
        model.train()

    # 测试集预测与评估
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch in test_loader:
            pred = model(batch.x, batch.edge_index, batch.batch)
            preds.append(pred.cpu().numpy())
            if batch.y.ndim == 1:
                trues.append(batch.y.cpu().numpy().reshape(-1, out_dim))
            else:
                trues.append(batch.y.cpu().numpy())
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    mse = mean_squared_error(trues, preds)
    corr = np.corrcoef(trues.flatten(), preds.flatten())[0, 1]
    rs = correlation_score(trues.T, preds.T)
    rs_mean = np.mean(rs)
    print(f"测试集 MSE: {mse:.4f}, 相关系数: {corr:.4f}, correlation_score均值: {rs_mean:.4f}")

    # 保存测试集预测结果
    save_test_path = os.path.join(decoded_dir, subject, f"{subject}_roi_cnn_gnn_scores_{target}_test.npy")
    np.save(save_test_path, preds)
    print(f"测试集预测结果已保存到: {save_test_path}")

    # 保存模型
    model_path = os.path.join(decoded_dir, subject, f"{subject}_roi_cnn_gnn_model_{target}.pt")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"训练好的模型已保存到: {model_path}")


if __name__ == "__main__":
    main()