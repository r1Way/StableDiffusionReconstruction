# 本脚本实现：每个脑区先单独卷积（CNN），输出20维向量，然后所有脑区特征拼接后进行图卷积，最后通过若干全连接层输出6400维目标。
# 支持GPU加速
# 用法示例：
# python gnn_roi_cnn2.py --subject subj01 --target init_latent --test

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
import csv
import gc


class ROICNN(nn.Module):
    def __init__(self, in_dim, out_dim=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, max(32, in_dim // 4)),
            nn.ReLU(),
            nn.Linear(max(32, in_dim // 4), out_dim)
        )

    def forward(self, x):
        return self.net(x)


class FullModel(nn.Module):
    def __init__(self, roi_in_dims, roi_out_dim=20, gcn_hidden1=64, gcn_hidden2=128, fc1=512, fc2=1024, out_dim=6400,
                 dropout=0.3):
        super().__init__()
        self.n_rois = len(roi_in_dims)
        self.roi_in_dims = roi_in_dims
        self.roi_cnns = nn.ModuleList([ROICNN(in_dim, roi_out_dim) for in_dim in roi_in_dims])
        self.gcn1 = GCNConv(roi_out_dim, gcn_hidden1)
        self.gcn2 = GCNConv(gcn_hidden1, gcn_hidden2)
        self.dropout = dropout
        self.fc1 = nn.Linear(gcn_hidden2, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc_out = nn.Linear(fc2, out_dim)

    def forward(self, x, edge_index, batch):
        batch_size = batch.max().item() + 1
        n_rois = self.n_rois
        roi_feats = []
        for b in range(batch_size):
            roi_feat_sample = []
            for i in range(n_rois):
                roi_x = x[i][b * self.roi_in_dims[i]: (b + 1) * self.roi_in_dims[i]]
                roi_x = roi_x.view(1, -1)
                roi_feat = self.roi_cnns[i](roi_x)
                roi_feat_sample.append(roi_feat)
            roi_feat_sample = torch.cat(roi_feat_sample, dim=0)
            roi_feats.append(roi_feat_sample)
        node_feats = torch.cat(roi_feats, dim=0)
        x = self.gcn1(node_feats, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc_out(x)
        return x


def build_edge_index(adj):
    idx = np.where(adj > 0)
    edge_index = np.vstack(idx)
    return torch.tensor(edge_index, dtype=torch.long)


def build_custom_adj(rois):
    n = len(rois)
    adj = np.ones((n, n), dtype=np.float32)
    np.fill_diagonal(adj, 0)
    group1 = ['V1']
    group2 = ['V2', 'V3', 'V4']
    group3 = ['V8', 'FFC', 'VMV1', 'VMV2', 'VMV3', 'VVC']
    groups = [group1, group2, group3]
    for group in groups:
        idxs = [rois.index(r) for r in group]
        for i in idxs:
            for j in idxs:
                if i != j:
                    adj[i, j] = 3
    np.fill_diagonal(adj, 0)
    return adj


def set_random_seeds(seed=42):
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
    parser.add_argument("--roi_out_dim", type=int, default=20, help="ROICNN输出维度")
    parser.add_argument("--test", action="store_true", help="测试模式，仅训练1个epoch，batch_size=4")
    args = parser.parse_args()

    subject = args.subject
    mridir = args.mridir
    target = args.target
    decoded_dir = args.decoded_dir
    roi_out_dim = args.roi_out_dim
    rois = ['V1', 'V2', 'V3', 'V4', 'V8', 'FFC', 'VMV1', 'VMV2', 'VMV3', 'VVC']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    adj = build_custom_adj(rois)
    edge_index = build_edge_index(adj).to(device)

    roi_in_dims = []
    for roi in rois:
        betas_path = os.path.join(mridir, subject, f"{subject}_{roi}_betas_tr.npy")
        X = np.load(betas_path)
        roi_in_dims.append(X.shape[1])

    X_list = []
    for roi in rois:
        betas_path = os.path.join(mridir, subject, f"{subject}_{roi}_betas_tr.npy")
        X = np.load(betas_path)
        X_list.append(X)
    X_train = np.concatenate(X_list, axis=1)
    n_train = X_train.shape[0]
    X_train_mean = X_train.mean(axis=0, keepdims=True)
    X_train_std = X_train.std(axis=0, keepdims=True) + 1e-6
    X_train = (X_train - X_train_mean) / X_train_std
    Y_train_path = os.path.join('../../nsdfeat/subjfeat/', f"{subject}_each_{target}_tr.npy")
    Y_train = np.load(Y_train_path).astype("float32")
    out_dim = Y_train.shape[1] if Y_train.ndim == 2 else Y_train.reshape(n_train, -1).shape[1]

    X_list_te = []
    for roi in rois:
        betas_te_path = os.path.join(mridir, subject, f"{subject}_{roi}_betas_ave_te.npy")
        X_te = np.load(betas_te_path)
        X_list_te.append(X_te)
    X_test = np.concatenate(X_list_te, axis=1)
    n_test = X_test.shape[0]
    X_test = (X_test - X_train_mean) / X_train_std
    Y_test_path = os.path.join('../../nsdfeat/subjfeat/', f"{subject}_ave_{target}_te.npy")
    Y_test = np.load(Y_test_path).astype("float32")

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

    train_data_list = []
    for i in range(n_train):
        roi_xs = []
        start = 0
        for dim in roi_in_dims:
            roi_xs.append(torch.tensor(X_train[i, start:start + dim], dtype=torch.float32))
            start += dim
        y = torch.tensor(Y_train[i], dtype=torch.float32)
        batch = torch.full((len(roi_xs),), i, dtype=torch.long)
        data = Data(x=roi_xs, edge_index=edge_index, y=y, batch=batch)
        train_data_list.append(data)
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

    set_random_seeds(42)
    print("随机种子已设置为42")

    model = FullModel(
        roi_in_dims=roi_in_dims,
        roi_out_dim=roi_out_dim,
        gcn_hidden1=64,
        gcn_hidden2=128,
        fc1=512,
        fc2=1024,
        out_dim=out_dim,
        dropout=0.3
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    n_epochs = 1 if args.test else 10
    train_csv_path = os.path.join(decoded_dir, subject,
                                  f"{subject}_roi_cnn_gnn2_train_metrics_{target}_outdim{roi_out_dim}.csv")
    if not os.path.exists(os.path.dirname(train_csv_path)):
        os.makedirs(os.path.dirname(train_csv_path), exist_ok=True)
    if not os.path.exists(train_csv_path):
        with open(train_csv_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "mse", "corr", "correlation_score_mean"])
    for epoch in range(n_epochs):
        epoch_start = time.time()
        total_loss = 0
        print(f"Epoch {epoch + 1}/{n_epochs} 开始")
        for batch_idx, batch in enumerate(train_loader):
            batch_start = time.time()
            optimizer.zero_grad()
            # 将数据转移到GPU
            batch_x = [roi_x.to(device) for roi_x in batch.x]
            batch_edge_index = batch.edge_index.to(device)
            batch_batch = batch.batch.to(device)
            batch_y = batch.y.to(device)
            pred = model(batch_x, batch_edge_index, batch_batch)
            if batch_y.ndim == 1:
                batch_y = batch_y.view(-1, out_dim)
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
        # 清理不需要的变量和内存
        gc.collect()
        torch.cuda.empty_cache()
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
        # 追加写入csv
        with open(train_csv_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_mse, train_corr, train_rs_mean])
        # 清理评估产生的大数组
        del train_preds, train_trues
        gc.collect()
        torch.cuda.empty_cache()
        model.train()

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

    save_test_path = os.path.join(decoded_dir, subject,
                                  f"{subject}_roi_cnn_gnn2_scores_{target}_outdim{roi_out_dim}_test.npy")
    np.save(save_test_path, preds)
    print(f"测试集预测结果已保存到: {save_test_path}")

    model_path = os.path.join(decoded_dir, subject, f"{subject}_roi_cnn_gnn2_model_{target}_outdim{roi_out_dim}.pt")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"训练好的模型已保存到: {model_path}")


if __name__ == "__main__":
    main()
