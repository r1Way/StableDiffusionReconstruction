# 本脚本用于计算指定被试(subject)的10个脑区主成分均值时间序列的相关性邻接矩阵。
# 用法示例：
# python make_pca_adj_matrix.py --subject subj01

import os
import numpy as np

rois = ['V1','V2','V3','V4','V8','FFC','VMV1','VMV2','VMV3','VVC']

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True, help="subject name")
    parser.add_argument("--mridir", type=str, default='../../mrifeat/', help="fMRI feature dir")
    parser.add_argument("--n_components", type=int, default=20, help="PCA主成分数")
    args = parser.parse_args()

    subject = args.subject
    mridir = args.mridir
    n_components = args.n_components
    savedir = os.path.join(mridir, subject)

    # 读取每个脑区的PCA特征，取均值
    roi_means = []
    for roi in rois:
        pca_path = os.path.join(savedir, f"{subject}_{roi}_betas_tr_PCA{n_components}.npy")
        if not os.path.exists(pca_path):
            print(f"缺少文件: {pca_path}，跳过 {roi}")
            roi_means.append(None)
            continue
        X_pca = np.load(pca_path)  # shape: (n_samples, n_components)
        mean_ts = X_pca.mean(axis=1)  # 对主成分取均值，得到时间序列
        roi_means.append(mean_ts)

    # 构建邻接矩阵
    n_roi = len(rois)
    adj_matrix = np.zeros((n_roi, n_roi))
    for i in range(n_roi):
        for j in range(n_roi):
            if roi_means[i] is None or roi_means[j] is None:
                adj_matrix[i, j] = np.nan
            else:
                r = np.corrcoef(roi_means[i], roi_means[j])[0, 1]
                adj_matrix[i, j] = r

    # 保存邻接矩阵
    out_path = os.path.join(savedir, f"{subject}_pca_adj_matrix.npy")
    np.save(out_path, adj_matrix)
    print(f"邻接矩阵已保存到: {out_path}")
    print("邻接矩阵:")
    print(adj_matrix)

if __name__ == "__main__":
    main()
