# 本脚本用于交互式查看NSD fMRI beta_trial数据的不同trial和不同z轴切片。
# 运行后可通过滑块动态选择trial编号和z轴切片，实时显示对应的脑图像。
# 用法示例：python view_beta_trial.py --subject subj01 --atlas HCP_MMP1 --color_roi V1

"""
使用方法说明：

1. 基本用法（只看fMRI切片）：
```bash
python view_beta_trial.py --subject subj01
```

2. 启用归一化（灰度显示更清晰）：
- 运行后在左侧选择 Norm 为 on。

3. 染色某个脑区（如V1），可选atlas为streams或HCP_MMP1：
```bash
python view_beta_trial.py --subject subj01 --atlas streams --color_roi early
```

4. 指定ctab文件目录（如默认路径不对）：
```bash
python view_beta_trial.py --subject subj01 --atlas streams --color_roi early --ctab_dir D:/project/2025/StableDiffusionReconstruction/nsd/nsddata/freesurfer/fsaverage/label
```

5. 运行后可用滑块切换trial和切片，左侧可切换x/y/z轴和归一化，染色区域会以红色叠加显示。

注意事项：
- color_roi参数必须与atlas对应的ROI名称一致（可通过ctab文件查看）。
- 归一化和染色可同时使用，便于观察脑区分布。
- 仅支持func1pt8mm格式的atlas染色。
"""

import argparse
from nsd_access import NSDAccess
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, RadioButtons
import numpy as np
import os

def read_roi_list(ctab_path):
    """
    从ctab文件读取ROI名称列表
    """
    import pandas as pd
    df = pd.read_csv(ctab_path, delimiter=' ', header=None, index_col=0)
    return df[1].tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="subject name: subj01 or subj02 or subj05 or subj07 for full-data subjects",
    )
    # 新增atlas和染色ROI选项
    parser.add_argument(
        "--atlas",
        type=str,
        default=None,
        choices=['streams', 'HCP_MMP1'],
        help="选择染色用的atlas，可选: streams 或 HCP_MMP1"
    )
    parser.add_argument(
        "--ctab_dir",
        type=str,
        default="D:/project/2025/StableDiffusionReconstruction/nsd/nsddata/freesurfer/fsaverage/label",
        help="ctab文件目录"
    )
    parser.add_argument(
        "--color_roi",
        type=str,
        default=None,
        help="指定要染色的ROI名称"
    )
    opt = parser.parse_args()
    subject = opt.subject

    # 修正：只在文件顶部import一次NSDAccess，删除main内部的from nsd_access import NSDAccess
    nsda = NSDAccess('../../nsd/')

    # 读取指定被试者的第10个session的beta_trial数据
    beta_trial = nsda.read_betas(
        subject=subject,
        session_index=10,
        trial_index=[], # 获取该 session 的所有 trial
        data_type='betas_fithrf_GLMdenoise_RR',
        data_format='func1pt8mm'
    )
    print("beta_trial type:", type(beta_trial))
    print("beta_trial shape:", getattr(beta_trial, "shape", None))
    print("beta_trial dtype:", getattr(beta_trial, "dtype", None))
    print("beta_trial sample:", beta_trial if isinstance(beta_trial, (list, dict)) else beta_trial[:5])

    # 设置交互控件参数
    trial_max = beta_trial.shape[0] - 1
    x_max = beta_trial.shape[1] - 1
    y_max = beta_trial.shape[2] - 1
    z_max = beta_trial.shape[3] - 1

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.35)  # 增加底部空间用于输入框
    trial_idx = 0
    axis = 'z'
    slice_idx = z_max // 2

    # 初始显示z轴切片
    im = ax.imshow(beta_trial[trial_idx, :, :, slice_idx], cmap='gray')
    ax.set_title(f'Trial {trial_idx}, {axis.upper()} Slice {slice_idx}')
    plt.colorbar(im, ax=ax)

    # trial滑块
    ax_trial = plt.axes([0.25, 0.15, 0.65, 0.03])
    slider_trial = Slider(ax_trial, 'Trial', 0, trial_max, valinit=trial_idx, valstep=1)

    # trial输入框
    ax_trial_box = plt.axes([0.25, 0.21, 0.1, 0.04])
    trial_box = TextBox(ax_trial_box, 'TrialNum', initial=str(trial_idx))

    # 切片滑块
    ax_slice = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider_slice = Slider(ax_slice, 'Slice', 0, z_max, valinit=slice_idx, valstep=1)

    # 切片输入框
    ax_slice_box = plt.axes([0.37, 0.21, 0.1, 0.04])
    slice_box = TextBox(ax_slice_box, 'SliceNum', initial=str(slice_idx))

    # 方向输入框
    ax_axis_box = plt.axes([0.49, 0.21, 0.1, 0.04])
    axis_box = TextBox(ax_axis_box, 'Axis', initial=axis)

    # 方向选择控件
    ax_radio = plt.axes([0.025, 0.4, 0.15, 0.15])
    radio = RadioButtons(ax_radio, ('x', 'y', 'z'), active=2)

    # 新增归一化开关控件
    ax_norm = plt.axes([0.025, 0.6, 0.15, 0.08])
    radio_norm = RadioButtons(ax_norm, ('off', 'on'), active=0)

    # 归一化处理
    normed_beta_trial = None
    norm_enabled = False

    def normalize_data(data):
        # 归一化到0-1，并保持float类型用于灰度显示
        # 修正溢出警告：先转换为float32再做运算
        data = data.astype(np.float32)
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val > min_val:
            normed = (data - min_val) / (max_val - min_val)
        else:
            normed = np.zeros_like(data, dtype=np.float32)
        return normed

    def enhance_contrast(data, lower=2, upper=98):
        # 按分位数裁剪并归一化，增强对比度
        vmin = np.percentile(data, lower)
        vmax = np.percentile(data, upper)
        data = np.clip(data, vmin, vmax)
        if vmax > vmin:
            data = (data - vmin) / (vmax - vmin)
        else:
            data = np.zeros_like(data, dtype=np.float32)
        return data

    # 预先归一化所有trial
    normed_beta_trial = np.zeros_like(beta_trial, dtype=np.float32)
    for t in range(beta_trial.shape[0]):
        normed_beta_trial[t] = normalize_data(beta_trial[t])

    # 读取ROI列表
    roi_list = None
    if opt.atlas is not None:
        ctab_path = os.path.join(opt.ctab_dir, f"{opt.atlas}.mgz.ctab")
        if os.path.exists(ctab_path):
            roi_list = read_roi_list(ctab_path)
            print(f"可选ROI（{opt.atlas}）：{roi_list}")

    # 如果指定了atlas和color_roi，则读取atlas并生成染色mask
    mask = None
    if opt.atlas is not None and opt.color_roi is not None:
        atlas_data, atlas_mapping = nsda.read_atlas_results(subject=subject, atlas=opt.atlas, data_format='func1pt8mm')
        val = atlas_mapping.get(opt.color_roi, 0)
        if val != 0:
            mask = (atlas_data.transpose([2,1,0]) == val).astype(np.uint8)
            print(f"已生成染色mask，ROI: {opt.color_roi}, atlas: {opt.atlas}, mask shape: {mask.shape}")
        else:
            print(f"ROI {opt.color_roi} 在atlas {opt.atlas}中未找到或值为0，无法染色。")

    def update(val):
        t = int(slider_trial.val)
        s = int(slider_slice.val)
        a = radio.value_selected
        use_norm = (radio_norm.value_selected == 'on')
        # 修正x/y轴显示：imshow默认显示行(row)为y，列(col)为x，需转置
        if a == 'x':
            data = beta_trial[t, s, :, :] if not use_norm else normed_beta_trial[t, s, :, :]
            data = data.T
            mask_slice = mask[s, :, :] if mask is not None else None
            if mask_slice is not None:
                mask_slice = mask_slice.T
        elif a == 'y':
            data = beta_trial[t, :, s, :] if not use_norm else normed_beta_trial[t, :, s, :]
            data = data.T
            mask_slice = mask[:, s, :] if mask is not None else None
            if mask_slice is not None:
                mask_slice = mask_slice.T
        else:  # 'z'
            data = beta_trial[t, :, :, s] if not use_norm else normed_beta_trial[t, :, :, s]
            mask_slice = mask[:, :, s] if mask is not None else None
        # 增强对比度
        data = enhance_contrast(data)
        # 修正：主图用imshow显示，mask用ax.imshow显示且设置zorder更高，保证绿色覆盖灰度
        im.set_data(data)
        im.set_cmap('gray_r')
        im.set_clim(0, 1)
        norm_str = "(Norm)" if use_norm else ""
        ax.set_title(f'Trial {t}, {a.upper()} Slice {s} {norm_str}')
        # 清除旧的mask叠加
        for artist in ax.get_images()[1:]:
            artist.remove()
        # 绿色更深一些，且与灰度底图区分明显
        if mask_slice is not None and np.any(mask_slice):
            # 只显示mask区域，其他区域透明，且zorder高于主图
            mask_rgba = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
            mask_rgba[..., 1] = 1.0  # G通道为1，绿色
            mask_rgba[..., 3] = 0.7 * (mask_slice > 0)  # 透明度
            ax.imshow(mask_rgba, zorder=10)
        fig.canvas.draw_idle()

    def update_axis(label):
        # 切换方向时，重设滑块范围和初值
        if label == 'x':
            slider_slice.valmax = x_max
            slider_slice.set_val(x_max // 2)
            slider_slice.ax.set_xlim(slider_slice.valmin, slider_slice.valmax)
            slider_slice.label.set_text('Slice')
        elif label == 'y':
            slider_slice.valmax = y_max
            slider_slice.set_val(y_max // 2)
            slider_slice.ax.set_xlim(slider_slice.valmin, slider_slice.valmax)
            slider_slice.label.set_text('Slice')
        else:  # 'z'
            slider_slice.valmax = z_max
            slider_slice.set_val(z_max // 2)
            slider_slice.ax.set_xlim(slider_slice.valmin, slider_slice.valmax)
            slider_slice.label.set_text('Slice')
        axis_box.set_val(label)
        update(None)

    def trial_box_submit(text):
        try:
            idx = int(text)
            idx = max(0, min(trial_max, idx))
            slider_trial.set_val(idx)
        except Exception:
            pass

    def slice_box_submit(text):
        try:
            idx = int(text)
            if radio.value_selected == 'x':
                maxval = x_max
            elif radio.value_selected == 'y':
                maxval = y_max
            else:
                maxval = z_max
            idx = max(0, min(maxval, idx))
            slider_slice.set_val(idx)
        except Exception:
            pass

    def axis_box_submit(text):
        if text in ['x', 'y', 'z']:
            radio.set_active(['x', 'y', 'z'].index(text))

    trial_box.on_submit(trial_box_submit)
    slice_box.on_submit(slice_box_submit)
    axis_box.on_submit(axis_box_submit)

    slider_trial.on_changed(update)
    slider_slice.on_changed(update)
    radio.on_clicked(update_axis)
    radio_norm.on_clicked(lambda _: update(None))

    plt.show()

if __name__ == "__main__":
    main()
