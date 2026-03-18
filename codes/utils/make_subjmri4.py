# python make_subjmri4.py --subject subj01
import argparse
import os
import numpy as np
import pandas as pd
from nsd_access import NSDAccess
import scipy.io
import gc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )

    opt = parser.parse_args()
    subject = opt.subject
    atlasname = 'HCP_MMP1'  # 指定使用的脑区图谱名称

    nsda = NSDAccess('../../nsd/')
    nsd_expdesign = scipy.io.loadmat('../../nsd/nsddata/experiments/nsd/nsd_expdesign.mat')  # 读取实验设计矩阵

    # Note that most of nsd_expdesign indices are 1-base index!
    # This is why subtracting 1
    # 修改索引的起始位置
    sharedix = nsd_expdesign['sharedix'] - 1  # 共享的刺激图片索引

    behs = pd.DataFrame()  # 初始化一个空的 DataFrame，用于存储行为数据。

    # 将每个会话的数据拼接到一个behs中
    for i in range(1, 38):
        print('beh:', i)
        beh = nsda.read_behavior(subject=subject,
                                 session_index=i)
        behs = pd.concat((behs, beh))

    # Caution: 73KID is 1-based! https://cvnlab.slite.page/p/fRv4lz5V2F/Behavioral-data
    # 73KID 是每个视觉刺激（图片）的唯一标识符
    stims_unique = behs['73KID'].unique() - 1  # 获取不重复的刺激ID
    stims_all = behs['73KID'] - 1  # 获取所有刺激ID

    savedir = f'../../mrifeat/{subject}/'
    os.makedirs(savedir, exist_ok=True)

    # 定义要检查的文件路径
    stims_path = f'{savedir}/{subject}_stims.npy'
    stims_ave_path = f'{savedir}/{subject}_stims_ave.npy'
    betas_all_path = os.path.join(savedir, f'{subject}_betas_all.dat')

    # 预先定义 betas_all，方便后续使用
    betas_all = None

    if os.path.exists(stims_path) and os.path.exists(stims_ave_path) and os.path.exists(betas_all_path):
        print('所有文件已存在，直接加载.')
        stims_all = np.load(stims_path)
        stims_unique = np.load(stims_ave_path)
        # 从 stims_all 获取 total_samples，尝试一下，不行的话，就只能重新读一遍betas了，慢的一批，虽然是可接受的。
        total_samples = stims_all.shape[0]
        # 读取第一个会话的 beta_trial 来获取原始 shape
        beta_trial_shape = nsda.read_betas(subject=subject,
                                           session_index=1,
                                           trial_index=[],
                                           data_type='betas_fithrf_GLMdenoise_RR',
                                           data_format='func1pt8mm').shape
        # 加载内存映射文件时指定 shape
        betas_all = np.memmap(betas_all_path, dtype='int16', mode='r',
                              shape=(total_samples,) + beta_trial_shape[1:])
    else:
        print('betas内存映射文件缺失，开始拼接数据.')
        beta_trial = nsda.read_betas(subject=subject,
                                     session_index=1,
                                     trial_index=[],
                                     data_type='betas_fithrf_GLMdenoise_RR',
                                     data_format='func1pt8mm')
        shape = beta_trial.shape
        total_samples = shape[0]
        beta_trial_shape = shape
        for i in range(2, 38):
            print('beta_trial_count:', i)
            beta_trial = nsda.read_betas(subject=subject,
                                         session_index=i,
                                         trial_index=[],
                                         data_type='betas_fithrf_GLMdenoise_RR',
                                         data_format='func1pt8mm')
            total_samples += beta_trial.shape[0]
            del beta_trial
            gc.collect()

        # 创建内存映射文件
        betas_all = np.memmap(betas_all_path, dtype='int16', mode='w+',
                              shape=(total_samples,) + shape[1:])

        # 填充数据
        current_index = 0
        for i in range(1, 38):
            print('beta_trial:', i)
            beta_trial = nsda.read_betas(subject=subject,
                                         session_index=i,
                                         trial_index=[],
                                         data_type='betas_fithrf_GLMdenoise_RR',
                                         data_format='func1pt8mm')
            betas_all[current_index:current_index + beta_trial.shape[0]] = beta_trial
            current_index += beta_trial.shape[0]
            del beta_trial
            gc.collect()

        # 保存 stims_all 和 stims_unique
        if not os.path.exists(stims_path):
            np.save(stims_path, stims_all)
        if not os.path.exists(stims_ave_path):
            np.save(stims_ave_path, stims_unique)

    print('映射文件读取/生成结束')

    # 后续处理统一使用 betas_all
    # 读取指定被试者的脑区图谱数据。
    # 只保留指定的ROI
    # needed_rois = ['V1','V2','V3','V4','V5','V6','V7','V8','FFC','PIT','VMV1','VMV2','VMV3','VVC']
    needed_rois = ['ISP1', 'V3A', 'V3B', 'V6', 'V6A', 'V7']
    atlas = nsda.read_atlas_results(subject=subject, atlas=atlasname, data_format='func1pt8mm')
    print(f"Total number of ROIs: {len(atlas[1])}")
    for roi, val in atlas[1].items():
        if roi not in needed_rois:
            continue
        print(roi, val)
        if val == 0:
            print('SKIP')
            continue
        else:
            betas_roi = betas_all[:, atlas[0].transpose([2, 1, 0]) == val]

        print(betas_roi.shape)

        # Averaging for each stimulus 计算每个刺激的平均 fMRI 数据
        betas_roi_ave = []
        for stim in stims_unique:  # 遍历每个唯一的刺激索引，计算该刺激对应的 fMRI 数据的平均值。
            stim_mean = np.mean(betas_roi[stims_all == stim, :], axis=0)
            betas_roi_ave.append(stim_mean)
        betas_roi_ave = np.stack(betas_roi_ave)
        print(betas_roi_ave.shape)

        # Train/Test Split
        # ALLDATA
        betas_tr = []
        betas_te = []

        for idx, stim in enumerate(stims_all):
            if stim in sharedix:  # 共享的图片作为测试集
                betas_te.append(betas_roi[idx, :])
            else:
                betas_tr.append(betas_roi[idx, :])

        betas_tr = np.stack(betas_tr)
        betas_te = np.stack(betas_te)

        # AVERAGED DATA
        betas_ave_tr = []
        betas_ave_te = []
        for idx, stim in enumerate(stims_unique):
            if stim in sharedix:
                betas_ave_te.append(betas_roi_ave[idx, :])
            else:
                betas_ave_tr.append(betas_roi_ave[idx, :])
        betas_ave_tr = np.stack(betas_ave_tr)
        betas_ave_te = np.stack(betas_ave_te)

        # Save
        np.save(f'{savedir}/{subject}_{roi}_betas_tr.npy', betas_tr)
        np.save(f'{savedir}/{subject}_{roi}_betas_te.npy', betas_te)
        np.save(f'{savedir}/{subject}_{roi}_betas_ave_tr.npy', betas_ave_tr)
        np.save(f'{savedir}/{subject}_{roi}_betas_ave_te.npy', betas_ave_te)


if __name__ == "__main__":
    main()