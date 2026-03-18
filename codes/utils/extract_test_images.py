'''
cd codes/utils/
python extract_test_images.py --subject subj01
# 或指定自定义输出目录
python extract_test_images.py --subject subj01 --output_dir ../../my_test_images
'''
import h5py
from PIL import Image
import scipy.io
import argparse, os
import numpy as np
from tqdm import tqdm
import sys
sys.path.append("../utils/")
from nsd_access.nsda import NSDAccess

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--subject",
        required=True,
        type=str,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../test_images",
        help="output directory for test images",
    )
    
    opt = parser.parse_args()
    subject = opt.subject
    output_dir = opt.output_dir
    
    # 创建输出目录
    subject_output_dir = os.path.join(output_dir, subject)
    os.makedirs(subject_output_dir, exist_ok=True)
    
    print(f"提取被试 {subject} 的测试集图片...")
    print(f"输出目录: {subject_output_dir}")
    
    # Load NSD information
    nsd_expdesign = scipy.io.loadmat('../../nsd/nsddata/experiments/nsd/nsd_expdesign.mat')
    
    # Note that most of them are 1-base index!
    # This is why I subtract 1
    sharedix = nsd_expdesign['sharedix'] - 1
    
    nsda = NSDAccess('../../nsd/')
    # 原始图片存储在NSD数据集的stimuli文件中
    # 路径：nsd/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5
    sf = h5py.File(nsda.stimuli_file, 'r')
    sdataset = sf.get('imgBrick')  # 包含73,000张原始刺激图片的数据集
    
    # 加载被试的刺激索引映射
    # 这个文件定义了哪些图片对应测试集的982张图片
    stims_ave = np.load(f'../../mrifeat/{subject}/{subject}_stims_ave.npy')
    
    tr_idx = np.zeros_like(stims_ave)
    for idx, s in enumerate(stims_ave):
        if s in sharedix:
            tr_idx[idx] = 0  # 测试集图片标记为0
        else:
            tr_idx[idx] = 1  # 训练集图片标记为1
    
    # 获取测试集图片索引
    test_indices = np.where(tr_idx == 0)[0]
    total_test_images = len(test_indices)
    
    print(f"找到 {total_test_images} 张测试集图片")
    
    # 创建索引映射文件
    index_mapping = []
    
    # 提取并保存测试集图片
    for test_img_idx in tqdm(range(total_test_images), desc="提取测试集图片"):
        # 获取在完整数据集中的索引
        imgidx_te = test_indices[test_img_idx]
        # 获取在73k数据集中的真实索引
        idx73k = stims_ave[imgidx_te]
        
        # 从NSD stimuli数据集中读取原始图片
        img_array = np.squeeze(sdataset[idx73k, :, :, :]).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        # 保存图片，使用测试集索引命名
        img_filename = f"{test_img_idx:05d}.png"
        img_path = os.path.join(subject_output_dir, img_filename)
        img.save(img_path)
        
        # 记录索引映射关系
        index_mapping.append({
            'test_idx': test_img_idx,
            'full_dataset_idx': imgidx_te,
            'nsd_73k_idx': idx73k,
            'filename': img_filename
        })
    
    # 保存索引映射文件
    mapping_file = os.path.join(subject_output_dir, 'index_mapping.txt')
    with open(mapping_file, 'w') as f:
        f.write("# 测试集图片索引映射\n")
        f.write("# test_idx: 测试集编号 (0-981)\n")
        f.write("# full_dataset_idx: 完整数据集中的索引\n")
        f.write("# nsd_73k_idx: NSD 73k数据集中的索引\n")
        f.write("# filename: 保存的文件名\n")
        f.write("test_idx\tfull_dataset_idx\tnsd_73k_idx\tfilename\n")
        
        for mapping in index_mapping:
            f.write(f"{mapping['test_idx']}\t{mapping['full_dataset_idx']}\t{mapping['nsd_73k_idx']}\t{mapping['filename']}\n")
    
    print(f"✓ 成功提取 {total_test_images} 张测试集图片")
    print(f"✓ 图片保存在: {subject_output_dir}")
    print(f"✓ 索引映射保存在: {mapping_file}")
    
    # 显示一些统计信息
    print(f"\n统计信息:")
    print(f"- 被试: {subject}")
    print(f"- 测试集图片数量: {total_test_images}")
    print(f"- 图片编号范围: 00000.png - {total_test_images-1:05d}.png")
    print(f"- 图片尺寸: {img_array.shape}")

if __name__ == "__main__":
    main()
