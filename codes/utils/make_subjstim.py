'''
为指定被试（如 subj01）和指定特征（如 init_latent），
生成每个刺激（图片）对应的特征矩阵，并进行训练/测试集划分，
保存为 numpy 文件，供后续脑解码或重建分析使用。
'''
import numpy as np
import scipy.io
from tqdm import tqdm
import argparse
import os

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()

    # 添加特征名称参数，用于指定要提取的特征类型（如 init_latent 等）
    parser.add_argument(
        "--featname",
        type=str,
        default='',
        help="Target variable",
    )
    # 添加刺激使用方式参数，ave表示使用平均刺激，each表示使用每个刺激
    parser.add_argument(
        "--use_stim",
        type=str,
        default='',
        help="ave or each",
    )
    # 添加被试名称参数，支持完整数据的被试（subj01, subj02, subj05, subj07）
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )

    # 解析命令行参数
    opt = parser.parse_args()
    subject = opt.subject
    use_stim = opt.use_stim
    featname = opt.featname
    
    # 定义数据路径
    topdir = '../../nsdfeat/'  # 特征数据根目录
    savedir = f'{topdir}/subjfeat/'  # 保存被试特征的目录
    featdir = f'{topdir}/{featname}/'  # 指定特征类型的目录

    # 加载 NSD 实验设计文件，包含刺激索引信息
    nsd_expdesign = scipy.io.loadmat('../../nsd/nsddata/experiments/nsd/nsd_expdesign.mat')

    # 获取共享刺激索引（在训练和测试中都出现的刺激）
    # 注意：MATLAB 索引从1开始，Python从0开始，所以需要减1
    sharedix = nsd_expdesign['sharedix'] - 1 

    # 根据刺激使用方式加载对应的刺激索引数据
    if use_stim == 'ave':
        # 加载平均刺激索引
        stims = np.load(f'../../mrifeat/{subject}/{subject}_stims_ave.npy')
    else: # Each
        # 加载每个刺激的索引
        stims = np.load(f'../../mrifeat/{subject}/{subject}_stims.npy')
    
    # 初始化特征列表和训练/测试标签数组
    feats = []
    tr_idx = np.zeros(len(stims))  # 0表示测试集，1表示训练集

    # 遍历所有刺激，提取特征并标记训练/测试集
    for idx, s in tqdm(enumerate(stims)): 
        # 判断当前刺激是否为共享刺激（测试集）
        if s in sharedix:
            tr_idx[idx] = 0  # 共享刺激标记为测试集
        else:
            tr_idx[idx] = 1  # 非共享刺激标记为训练集
        
        # 加载对应刺激的特征文件（文件名格式为6位数字）
        feat = np.load(f'{featdir}/{s:06}.npy')
        feats.append(feat)

    # 将特征列表转换为numpy数组
    feats = np.stack(feats)    

    # 创建保存目录（如果不存在）
    os.makedirs(savedir, exist_ok=True)

    # 根据训练/测试标签分离特征数据
    feats_tr = feats[tr_idx==1,:]  # 训练集特征
    feats_te = feats[tr_idx==0,:]  # 测试集特征
    
    # 保存训练/测试索引标签
    np.save(f'../../mrifeat/{subject}/{subject}_stims_tridx.npy', tr_idx)

    # 保存训练集和测试集特征数据
    np.save(f'{savedir}/{subject}_{use_stim}_{featname}_tr.npy', feats_tr)
    np.save(f'{savedir}/{subject}_{use_stim}_{featname}_te.npy', feats_te)


if __name__ == "__main__":
    main()
