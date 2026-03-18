# 本脚本用于基于10个脑区的PCA特征和邻接矩阵，构建图神经网络进行预测。
# 网络结构：2层GCN+全局池化+2层全连接，输出与mlp2.py一致。
# 用法示例：
# python gnn_predict.py --subject subj01 --target init_latent

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
import joblib
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class GNNModel(nn.Module):
    def __init__(self, in_dim, hidden1=64, hidden2=128, fc1=1024, out_dim=6400, dropout=0.3):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, hidden1)
        self.gcn2 = GCNConv(hidden1, hidden2)
        self.dropout = dropout
        self.fc1 = nn.Linear(hidden2, fc1)
        self.fc2 = nn.Linear(fc1, out_dim)

    def forward(self, x, edge_index, batch):
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)  # [batch_size, hidden2]
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

def build_edge_index(adj):
    # 构建无向图的edge_index
    idx = np.where(adj > 0)
    edge_index = np.vstack(idx)
    return torch.tensor(edge_index, dtype=torch.long)

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
    parser.add_argument("--n_components", type=int, default=20, help="PCA主成分数")
    parser.add_argument("--target", type=str, required=True, help="预测目标，如c")
    parser.add_argument("--decoded_dir", type=str, default='../../decoded', help="预测结果保存目录")
    parser.add_argument("--test", action="store_true", help="测试模式，仅训练1个epoch，batch_size=4")
    args = parser.parse_args()

    subject = args.subject
    mridir = args.mridir
    n_components = args.n_components
    target = args.target
    decoded_dir = args.decoded_dir
    rois = ['V1','V2','V3','V4','V8','FFC','VMV1','VMV2','VMV3','VVC']

    # 读取邻接矩阵
    adj_path = os.path.join(mridir, subject, f"{subject}_pca_adj_matrix.npy")
    adj = np.load(adj_path)
    edge_index = build_edge_index(adj)

    # 读取训练集
    X_list = []
    for roi in rois:
        pca_path = os.path.join(mridir, subject, f"{subject}_{roi}_betas_tr_PCA{n_components}.npy")
        X = np.load(pca_path)  # shape: [n_train, n_components]
        X_list.append(X)
    X_train = np.stack(X_list, axis=1)  # [n_train, 10, n_components]
    Y_train_path = os.path.join('../../nsdfeat/subjfeat/', f"{subject}_each_{target}_tr.npy")
    Y_train = np.load(Y_train_path).astype("float32")
    n_train = X_train.shape[0]
    out_dim = Y_train.shape[1] if Y_train.ndim == 2 else Y_train.reshape(n_train, -1).shape[1]

    # 读取测试集
    X_list_te = []
    for roi in rois:
        pca_path_te = os.path.join(mridir, subject, f"{subject}_{roi}_betas_ave_te_PCA{n_components}.npy")
        X_te = np.load(pca_path_te)  # shape: [n_test, n_components]
        X_list_te.append(X_te)
    X_test = np.stack(X_list_te, axis=1)  # [n_test, 10, n_components]
    Y_test_path = os.path.join('../../nsdfeat/subjfeat/', f"{subject}_ave_{target}_te.npy")
    Y_test = np.load(Y_test_path).astype("float32")
    n_test = X_test.shape[0]

    # 构建训练集和测试集 DataLoader
    train_data_list = []
    for i in range(n_train):
        x = torch.tensor(X_train[i], dtype=torch.float32)  # [10, n_components]
        y = torch.tensor(Y_train[i], dtype=torch.float32)  # [out_dim]
        # 不要传入 batch
        data = Data(x=x, edge_index=edge_index, y=y)
        train_data_list.append(data)
    test_data_list = []
    for i in range(n_test):
        x = torch.tensor(X_test[i], dtype=torch.float32)
        y = torch.tensor(Y_test[i], dtype=torch.float32)
        data = Data(x=x, edge_index=edge_index, y=y)
        test_data_list.append(data)
    batch_size = 4 if args.test else 32
    train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=False)

    # 设置随机种子
    set_random_seeds(42)
    print("随机种子已设置为42")

    # 构建模型
    model = GNNModel(in_dim=n_components, out_dim=out_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # 新增：用于保存loss
    all_epoch_loss = []
    all_batch_loss = []

    # 训练
    model.train()
    n_epochs = 1 if args.test else 10
    for epoch in range(n_epochs):
        epoch_start = time.time()
        total_loss = 0
        batch_losses = []
        print(f"Epoch {epoch+1}/{n_epochs} 开始")
        for batch_idx, batch in enumerate(train_loader):
            batch_start = time.time()
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.batch)  # [batch_size, out_dim]
            if batch.y.ndim == 1:
                batch_y = batch.y.view(-1, out_dim)
            else:
                batch_y = batch.y
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_losses.append(loss.item())
            batch_end = time.time()
            print(f"  Batch {batch_idx+1}/{len(train_loader)} Loss: {loss.item():.4f} 用时: {batch_end-batch_start:.2f}s")
        epoch_end = time.time()
        avg_loss = total_loss / len(train_loader)
        all_epoch_loss.append(avg_loss)
        all_batch_loss.append(batch_losses)
        print(f"Epoch {epoch+1} 总Loss: {avg_loss:.4f} 用时: {epoch_end-epoch_start:.2f}s")

    # 保存loss曲线数据
    loss_save_dir = os.path.join(decoded_dir, subject)
    os.makedirs(loss_save_dir, exist_ok=True)
    np.save(os.path.join(loss_save_dir, f"{subject}_gnn_epoch_loss_{target}.npy"), np.array(all_epoch_loss))
    np.save(os.path.join(loss_save_dir, f"{subject}_gnn_batch_loss_{target}.npy"), np.array(all_batch_loss, dtype=object))
    print(f"每个epoch和batch的loss已保存到: {loss_save_dir}")

    # 测试集预测与评估
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch in test_loader:
            pred = model(batch.x, batch.edge_index, batch.batch)
            if batch.y.ndim == 1:
                trues.append(batch.y.cpu().numpy().reshape(-1, out_dim))
            else:
                trues.append(batch.y.cpu().numpy())
            preds.append(pred.cpu().numpy())
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    mse = mean_squared_error(trues, preds)
    corr = np.corrcoef(trues.flatten(), preds.flatten())[0, 1]
    # correlation_score 按照特征维度评估
    rs = correlation_score(trues.T, preds.T)
    rs_mean = np.mean(rs)
    print(f"测试集 MSE: {mse:.4f}, 相关系数: {corr:.4f}, correlation_score均值: {rs_mean:.4f}")

    # 保存测试集预测结果
    save_test_path = os.path.join(decoded_dir, subject, f"{subject}_gnn_scores_{target}_test.npy")
    np.save(save_test_path, preds)
    print(f"测试集预测结果已保存到: {save_test_path}")

    # 保存模型
    model_path = os.path.join(decoded_dir, subject, f"{subject}_gnn_model_{target}.pt")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"训练好的模型已保存到: {model_path}")

    # 训练集预测（可选）
    train_preds = []
    train_trues = []
    with torch.no_grad():
        for batch in train_loader:
            pred = model(batch.x, batch.edge_index, batch.batch)
            if batch.y.ndim == 1:
                train_trues.append(batch.y.cpu().numpy().reshape(-1, out_dim))
            else:
                train_trues.append(batch.y.cpu().numpy())
            train_preds.append(pred.cpu().numpy())
    train_preds = np.vstack(train_preds)
    train_trues = np.vstack(train_trues)
    train_mse = mean_squared_error(train_trues, train_preds)
    train_corr = np.corrcoef(train_trues.flatten(), train_preds.flatten())[0, 1]
    train_rs = correlation_score(train_trues.T, train_preds.T)
    train_rs_mean = np.mean(train_rs)
    print(f"训练集 MSE: {train_mse:.4f}, 相关系数: {train_corr:.4f}, correlation_score均值: {train_rs_mean:.4f}")

    save_train_path = os.path.join(decoded_dir, subject, f"{subject}_gnn_scores_{target}_train.npy")
    np.save(save_train_path, train_preds)
    print(f"训练集预测结果已保存到: {save_train_path}")

if __name__ == "__main__":
    main()
