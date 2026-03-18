# 本脚本用于自定义每个脑区的PCA主成分数量，拼接后直接进行Ridge回归预测。
# 用法示例：
# python ridge_custom_pca.py --subject subj01 --target init_latent

import argparse, os
import numpy as np
import joblib
import csv
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
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
    args = parser.parse_args()

    subject = args.subject
    target = args.target
    mridir = args.mridir

    # 在此处手动填写每个脑区的PCA主成分数量
    roi_pca_dict = {
        'V1': 1014,
        'V2': 686,
        'V3': 492,
        'V4': 308,
        'V8': 78,
        'FFC': 176,
        'VMV1': 77,
        'VMV2': 57,
        'VMV3': 48,
        'VVC': 66,
    }
    rois = list(roi_pca_dict.keys())

    # 读取每个脑区的原始训练数据，做PCA
    X_list = []
    pca_models = {}  # 保存每个ROI的PCA模型
    for roi in rois:
        betas_path = os.path.join(mridir, subject, f"{subject}_{roi}_betas_tr.npy")
        X_raw = np.load(betas_path)  # [n_samples, n_voxels]
        pca_num = roi_pca_dict[roi]
        pca = PCA(n_components=pca_num)
        X_pca = pca.fit_transform(X_raw)
        X_list.append(X_pca)
        pca_models[roi] = pca  # 保存训练集PCA模型
    X = np.hstack(X_list)  # [n_samples, sum_pca]

    # 测试集同理（使用训练集的PCA模型transform）
    X_te_list = []
    for roi in rois:
        betas_te_path = os.path.join(mridir, subject, f"{subject}_{roi}_betas_ave_te.npy")
        X_te_raw = np.load(betas_te_path)
        pca = pca_models[roi]
        X_te_pca = pca.transform(X_te_raw)
        X_te_list.append(X_te_pca)
    X_te = np.hstack(X_te_list)

    # 读取目标特征
    featdir = '../../nsdfeat/subjfeat/'
    Y = np.load(f'{featdir}/{subject}_each_{target}_tr.npy').astype("float32")
    Y_te = np.load(f'{featdir}/{subject}_ave_{target}_te.npy').astype("float32")

    print(f'X {X.shape}, Y {Y.shape}, X_te {X_te.shape}, Y_te {Y_te.shape}')

    # 构建Ridge模型
    alpha = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    ridge = RidgeCV(alphas=alpha)
    pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        ridge,
    )

    # 训练
    pipeline.fit(X, Y)
    model_path = os.path.join('../../decoded', subject, f"{subject}_custompca_ridge_pipeline_{target}.joblib")
    joblib.dump(pipeline, model_path)
    print(f'Pipeline model saved to: {model_path}')

    # 预测
    scores = pipeline.predict(X_te)
    np.save(os.path.join('../../decoded', subject, f"{subject}_custompca_ridge_scores_{target}.npy"), scores)
    print(f'预测结果已保存到: ../../decoded/{subject}/')

    # 计算预测准确度
    rs = correlation_score(Y_te, scores)
    acc_mean = np.mean(rs)
    print(f'预测准确度均值: {acc_mean:.4f}')

    # 保存结果到csv（追加模式）
    csv_path = os.path.join('../../decoded', 'custompca_ridge_results.csv')
    header = list(roi_pca_dict.keys()) + [f'{roi}_pca' for roi in roi_pca_dict.keys()] + ['accuracy']
    row = [subject] + [roi_pca_dict[roi] for roi in roi_pca_dict.keys()] + [acc_mean]

    # 检查文件是否存在，决定是否写表头
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['subject'] + [f'{roi}_pca' for roi in roi_pca_dict.keys()] + ['accuracy'])
        writer.writerow([subject] + [roi_pca_dict[roi] for roi in roi_pca_dict.keys()] + [acc_mean])

if __name__ == "__main__":
    main()
