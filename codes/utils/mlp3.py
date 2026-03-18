# 本脚本用于基于10个脑区的主成分（PCA特征）进行MLP预测。
# 输入特征为每个脑区的PCA特征拼接，输出与mlp2.py一致。
# 用法示例：
# python mlp3.py --subject subj01 --target init_latent

import argparse, os
import numpy as np
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def correlation_score(y_true, y_pred):
    # 按列计算皮尔逊相关系数
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rs = []
    for i in range(y_true.shape[1]):
        r = np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1]
        rs.append(r)
    return np.array(rs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True, help="subject name")
    parser.add_argument("--target", type=str, required=True, help="预测目标，如c")
    parser.add_argument("--mridir", type=str, default='../../mrifeat/', help="fMRI feature dir")
    parser.add_argument("--n_components", type=int, default=20, help="PCA主成分数")
    parser.add_argument("--mlp_hidden", type=int, default=128, help="MLP隐藏层单元数")
    parser.add_argument("--mlp_max_iter", type=int, default=300, help="MLP最大迭代次数")
    parser.add_argument("--mlp_alpha", type=float, default=0.0001, help="MLP正则化参数")
    parser.add_argument("--mlp_activation", type=str, default="relu", help="MLP激活函数")
    parser.add_argument("--mlp_solver", type=str, default="adam", help="MLP优化器")
    args = parser.parse_args()

    subject = args.subject
    target = args.target
    mridir = args.mridir
    n_components = args.n_components
    rois = ['V1','V2','V3','V4','V8','FFC','VMV1','VMV2','VMV3','VVC']

    # 读取每个脑区的PCA特征
    X_list = []
    for roi in rois:
        pca_path = os.path.join(mridir, subject, f"{subject}_{roi}_betas_tr_PCA{n_components}.npy")
        X = np.load(pca_path)
        X_list.append(X)
    X = np.hstack(X_list)  # [n_samples, 10*n_components]

    # 读取测试集PCA特征
    X_te_list = []
    for roi in rois:
        pca_path = os.path.join(mridir, subject, f"{subject}_{roi}_betas_ave_te_PCA{n_components}.npy")
        X_te = np.load(pca_path)
        X_te_list.append(X_te)
    X_te = np.hstack(X_te_list)

    # 读取目标特征
    featdir = '../../nsdfeat/subjfeat/'
    Y = np.load(f'{featdir}/{subject}_each_{target}_tr.npy').astype("float32")
    Y_te = np.load(f'{featdir}/{subject}_ave_{target}_te.npy').astype("float32")

    print(f'X {X.shape}, Y {Y.shape}, X_te {X_te.shape}, Y_te {Y_te.shape}')

    # 构建MLP模型
    mlp = MLPRegressor(
        hidden_layer_sizes=(args.mlp_hidden,),
        max_iter=args.mlp_max_iter,
        alpha=args.mlp_alpha,
        activation=args.mlp_activation,
        solver=args.mlp_solver,
        random_state=42,
    )
    pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        mlp,
    )

    # 训练
    pipeline.fit(X, Y)
    model_path = os.path.join('../../decoded', subject, f"{subject}_pca_mlp_pipeline_{target}.joblib")
    joblib.dump(pipeline, model_path)
    print(f'Pipeline model saved to: {model_path}')

    # 预测
    scores = pipeline.predict(X_te)
    np.save(os.path.join('../../decoded', subject, f"{subject}_pca_mlp_scores_{target}.npy"), scores)
    print(f'预测结果已保存到: ../../decoded/{subject}/')

    # 计算预测准确度
    rs = correlation_score(Y_te, scores)
    print(f'预测准确度均值: {np.mean(rs):.4f}')

if __name__ == "__main__":
    main()
