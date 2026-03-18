# 忘记考虑内存映射了
import os
import numpy as np
import pandas as pd
from nsd_access import NSDAccess
import scipy.io
import argparse

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
    atlasname = 'HCP_MMP1' # 指定使用的脑区图谱名称
    
    nsda = NSDAccess('../../nsd/')
    nsd_expdesign = scipy.io.loadmat('../../nsd/nsddata/experiments/nsd/nsd_expdesign.mat')#读取实验设计矩阵
    sharedix = nsd_expdesign['sharedix'] -1 #共享的刺激图片索引

    behs = pd.DataFrame()# 初始化一个空的 DataFrame，用于存储行为数据。
    
    #将每个会话的数据拼接到一个behs中 
    for i in range(1,38):
        beh = nsda.read_behavior(subject=subject, 
                                session_index=i)
        behs = pd.concat((behs,beh))

    # Caution: 73KID is 1-based! https://cvnlab.slite.page/p/fRv4lz5V2F/Behavioral-data
    # 73KID 是每个视觉刺激（图片）的唯一标识符
    stims_unique = behs['73KID'].unique() - 1 # 获取不重复的刺激ID 
    stims_all = behs['73KID'] - 1 # 获取所有刺激ID

    savedir = f'../../mrifeat/{subject}/'
    os.makedirs(savedir, exist_ok=True)

    if not os.path.exists(f'{savedir}/{subject}_stims.npy'):
        np.save(f'{savedir}/{subject}_stims.npy',stims_all)
        np.save(f'{savedir}/{subject}_stims_ave.npy',stims_unique)


    # 读取每个会话，拼接fmri数据
    for i in range(1,38):
        print(i)
        beta_trial = nsda.read_betas(subject=subject, 
                                session_index=i, 
                                trial_index=[], # empty list as index means get all for this session
                                data_type='betas_fithrf_GLMdenoise_RR',
                                data_format='func1pt8mm')
        if i==1:
            betas_all = beta_trial
        else:
            betas_all = np.concatenate((betas_all,beta_trial),0)    
    
    # 只保留指定的ROI
    needed_rois = ['V1','V2','V3','V4','V5','V6','V7','V8','FFC','PIT','VMV1','VMV2','VMV3','VVC']
    # 读取指定被试者的脑区图谱数据。
    atlas = nsda.read_atlas_results(subject=subject, atlas=atlasname, data_format='func1pt8mm')
    for roi,val in atlas[1].items(): # atlas[1] - 体素标签数组
        if roi not in needed_rois:
            continue
        print(roi,val)
        if val == 0:
            print('SKIP')
            continue
        else:
            betas_roi = betas_all[:,atlas[0].transpose([2,1,0])==val] # atlas[0] - 体素标签数组

        print(betas_roi.shape)
        
        # Averaging for each stimulus 计算每个刺激的平均 fMRI 数据
        betas_roi_ave = []
        for stim in stims_unique: # 遍历每个唯一的刺激索引，计算该刺激对应的 fMRI 数据的平均值。
            stim_mean = np.mean(betas_roi[stims_all == stim,:],axis=0)
            betas_roi_ave.append(stim_mean)
        betas_roi_ave = np.stack(betas_roi_ave)
        print(betas_roi_ave.shape)
        
        # Train/Test Split
        # ALLDATA
        betas_tr = []
        betas_te = []

        for idx,stim in enumerate(stims_all):
            if stim in sharedix: # 共享的图片作为测试集
                betas_te.append(betas_roi[idx,:])
            else:
                betas_tr.append(betas_roi[idx,:])

        betas_tr = np.stack(betas_tr)
        betas_te = np.stack(betas_te)    
        
        # AVERAGED DATA        
        betas_ave_tr = []
        betas_ave_te = []
        for idx,stim in enumerate(stims_unique):
            if stim in sharedix:
                betas_ave_te.append(betas_roi_ave[idx,:])
            else:
                betas_ave_tr.append(betas_roi_ave[idx,:])
        betas_ave_tr = np.stack(betas_ave_tr)
        betas_ave_te = np.stack(betas_ave_te)    
        
        # Save
        np.save(f'{savedir}/{subject}_{roi}_betas_tr.npy',betas_tr)
        np.save(f'{savedir}/{subject}_{roi}_betas_te.npy',betas_te)
        np.save(f'{savedir}/{subject}_{roi}_betas_ave_tr.npy',betas_ave_tr)
        np.save(f'{savedir}/{subject}_{roi}_betas_ave_te.npy',betas_ave_te)


if __name__ == "__main__":
    main()
