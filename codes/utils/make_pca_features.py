# 本脚本用于对指定被试(subject)的多个脑区的fMRI数据进行主成分分析（PCA），
# 提取每个脑区的前20个主成分特征，并分别保存训练集和测试集的PCA特征文件。
# 用法示例：
# python make_pca_features.py --subject subj01

import os
import numpy as np
from sklearn.decomposition import PCA
import argparse

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True, help="subject name")
    parser.add_argument("--mridir", type=str, default='../../mrifeat/', help="fMRI feature dir")
    parser.add_argument("--n_components", type=int, default=20, help="PCA主成分数")
    parser.add_argument("--rois", type=str, nargs="+", default=['V1','V2','V3','V4','V8','FFC','VMV1','VMV2','VMV3','VVC'], help="ROI列表")
    args = parser.parse_args()

    subject = args.subject
    mridir = args.mridir
    n_components = args.n_components
    rois = args.rois

    savedir = os.path.join(mridir, subject)
    for roi in rois:
        # 构建训练集和测试集文件路径
        tr_path = os.path.join(savedir, f"{subject}_{roi}_betas_tr.npy")
        te_path = os.path.join(savedir, f"{subject}_{roi}_betas_ave_te.npy")
        if not os.path.exists(tr_path) or not os.path.exists(te_path):
            print(f"缺少文件: {tr_path} 或 {te_path}，跳过 {roi}")
            continue

        print(f"处理脑区: {roi}")
        X_tr = np.load(tr_path)
        X_te = np.load(te_path)

        # 对训练集做PCA，提取主成分
        pca = PCA(n_components=n_components)
        X_tr_pca = pca.fit_transform(X_tr)
        X_te_pca = pca.transform(X_te)

        # 保存PCA特征
        np.save(os.path.join(savedir, f"{subject}_{roi}_betas_tr_PCA{n_components}.npy"), X_tr_pca)
        np.save(os.path.join(savedir, f"{subject}_{roi}_betas_ave_te_PCA{n_components}.npy"), X_te_pca)
        # 可选：保存PCA模型参数
        # np.save(os.path.join(savedir, f"{subject}_{roi}_PCA{n_components}_components.npy"), pca.components_)
        # np.save(os.path.join(savedir, f"{subject}_{roi}_PCA{n_components}_explained_variance.npy"), pca.explained_variance_ratio_)

if __name__ == "__main__":
    main()
