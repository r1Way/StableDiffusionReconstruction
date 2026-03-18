# 本脚本用于批量运行ridge2.py，对指定被试(subject)的多个脑区(place_rois)进行MLP解码。
# 如果某个脑区的结果文件已存在则跳过，否则调用ridge2.py进行计算。
# 用法：直接运行 python run_ridge2_place.py

# python run_ridge2_place.py
import os
import subprocess

subject = 'subj01'
target = 'init_latent'
place_rois = ['V1','V2','V3','V4','V5','V6','V7','V8','FFC','PIT','VMV1','VMV2','VMV3','VVC']

for roi in place_rois:
    result_path = f'../../decoded/{subject}/{subject}_{roi}_scores_{target}.npy'
    if os.path.exists(result_path):
        print(f'已存在结果: {result_path}，跳过 {roi}')
        continue
    print(f'开始运行: {roi}')
    cmd = [
        'python', 'ridge2.py',
        '--target', target,
        '--roi', roi,
        '--subject', subject
    ]
    subprocess.run(cmd)
