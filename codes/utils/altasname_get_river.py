import os
import pandas as pd

def list_nsd_atlas(nsd_root):
    label_dir = os.path.join(nsd_root, 'nsddata', 'freesurfer', 'fsaverage', 'label')
    if not os.path.exists(label_dir):
        print(f"未找到 label 目录: {label_dir}")
        return
    atlas_files = [f for f in os.listdir(label_dir) if f.endswith('.mgz.ctab')]
    print("可用的脑区图谱（atlas）及其包含的ROI：")
    for fname in atlas_files:
        atlas_path = os.path.join(label_dir, fname)
        try:
            # 读取ctab文件，第一列为数值标签，第二列为ROI名称
            df = pd.read_csv(atlas_path, delimiter=' ', header=None, index_col=0)
            roi_names = df[1].tolist()
            print(f"\n{fname.split('.')[0]}:")
            for roi in roi_names:
                print(f"  - {roi}")
        except Exception as e:
            print(f"\n{fname.split('.')[0]}: 读取失败 ({e})")

if __name__ == "__main__":
    nsd_root = "../../nsd"  # 根据你的实际路径调整
    list_nsd_atlas(nsd_root)