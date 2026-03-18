# codes/utils 工具脚本文档

## 项目概述

本文档分析 StableDiffusionReconstruction 项目中 `codes/utils` 目录下各个 Python 脚本的用途、功能和区别。该项目是一个脑解码（Brain Decoding）项目，通过 fMRI 信号重建 Stable Diffusion 生成的图像。

---

## 原实验流程（CVPR 2023）

以下是 Takagi & Nishimoto CVPR 2023 论文中的原始实验流程，可作为参考基准：

### 步骤 1: MRI 预处理

```bash
cd codes/utils/
python make_subjmri.py --subject subj01
```

### 步骤 2: 特征提取

```bash
cd codes/utils/
python img2feat_sd1.py --imgidx 0 73000 --gpu 0
```

### 步骤 3: 特征划分

```bash
cd codes/utils/
python make_subjstim.py --featname init_latent --use_stim each --subject subj01
python make_subjstim.py --featname init_latent --use_stim ave --subject subj01
python make_subjstim.py --featname c --use_stim each --subject subj01
python make_subjstim.py --featname c --use_stim ave --subject subj01
```

### 步骤 4: 脑解码训练

```bash
cd codes/utils/
python ridge.py --target c --roi ventral --subject subj01
python ridge.py --target init_latent --roi early --subject subj01
```

### 步骤 5: 图像重建

```bash
cd codes/diffusion_sd1/
python diffusion_decoding.py --imgidx 0 10 --gpu 1 --subject subj01 --method cvpr
```

### 原实验方法说明

| 步骤 | 脚本 | 说明 |
|------|------|------|
| 特征提取 | `img2feat_sd1.py` | 提取 `init_latent` 和 `c` 两种 SD 特征 |
| 特征划分 | `make_subjstim.py` | 按 NSD 实验设计划分训练/测试集 |
| 脑解码 | `ridge.py` | 使用 Ridge 回归进行脑到特征映射 |
| 图像重建 | `diffusion_decoding.py` | 使用 DDIM 采样重建图像 |

### 与本项目的区别

- **原实验**：使用基础 `ridge.py`，单次训练全部特征
- **本项目**：使用 `ridge2.py`/`mlp2.py` 等优化版本，支持分批训练、内存映射、断点续训

---

## 文件分类总览

根据功能用途，utils 目录下的 37 个脚本可分为以下几类：

| 类别 | 数量 | 描述 |
|------|------|------|
| 数据提取与预处理 | 6 | 从 NSD 数据集提取 fMRI 和刺激数据 |
| 特征提取 | 2 | 提取 SD 特征和视觉特征 |
| 特征处理 | 3 | PCA 降维、邻接矩阵计算、识别准确率 |
| Ridge 回归模型 | 5 | 岭回归脑解码 |
| MLP 模型 | 3 | 多层感知机脑解码 |
| 随机森林模型 | 1 | 随机森林脑解码 |
| 图神经网络模型 | 6 | GNN 脑解码 |
| 批量运行脚本 | 3 | 批量执行训练任务 |
| 可视化与调试 | 4 | 查看数据、网络结构、fMRI 切片 |

---

## 1. 数据提取与预处理


### 1.1 fMRI 数据提取

分别读取各个脑区的数据并保存

| 文件 | 功能 | 特点 |
|------|------|------|
| `make_subjmri.py` | 从 NSD 提取 fMRI 数据 | streams 图谱 |
| `make_subjmri2.py` | 从 NSD 提取 fMRI 数据 | streams 图谱 + 内存映射优化 |
| `make_subjmri3.py` | 从 NSD 提取 fMRI 数据 | HCP_MMP1 图谱 |
| `make_subjmri4.py` | 从 NSD 提取 fMRI 数据 | HCP_MMP1 图谱 + 内存映射优化 |

**主要输出文件**（保存在 `../../mrifeat/{subject}/`）：
- `{subject}_stims.npy` - 所有试次的刺激索引
- `{subject}_stims_ave.npy` - 唯一刺激索引
- `{subject}_{roi}_betas_tr.npy` - 训练集原始 beta 值
- `{subject}_{roi}_betas_te.npy` - 测试集原始 beta 值
- `{subject}_{roi}_betas_ave_tr.npy` - 训练集平均 beta 值
- `{subject}_{roi}_betas_ave_te.npy` - 测试集平均 beta 值

**支持的 ROI**: V1, V2, V3, V4, V5, V6, V7, V8, FFC, PIT, VMV1, VMV2, VMV3, VVC, ISP1, V3A, V3B, V6, V6A, V7 等

### 1.2 被试刺激数据

| 文件 | 功能 | 图谱 |
|------|------|------|
| `make_subjstim.py` | 生成被试的刺激特征矩阵，划分训练/测试集将特征数据按 NSD 实验设计划分训练/测试集 | - |


### 1.3 测试图片提取

| 文件 | 功能 |
|------|------|
| `extract_test_images.py` | 从 NSD 数据集提取测试集图片 |

```bash
# 用法示例
python extract_test_images.py --subject subj01
python extract_test_images.py --subject subj01 --output_dir ../../my_test_images
```

---

## 2. 特征提取

### 2.1 Stable Diffusion 特征

| 文件 | 功能 |
|------|------|
| `img2feat_sd.py` | 提取图片的latent特征和文本条件特征 |

**提取的特征**：
- `init_latent`: 图片的 latent 空间特征（6400维）
- `c`: 文本条件特征

**输出目录**: `../../nsdfeat/init_latent/` 和 `../../nsdfeat/c/`

### 2.2 视觉特征提取

| 文件 | 功能 |
|------|------|
| `img2feat_decoded.py` | 提取解码图像的多种视觉特征 |

**提取的特征类型**：
- Inception V3 特征
- AlexNet 特征（多层）
- CLIP 特征（多层）

**输出目录**: `../../identification/{method}/{subject}/`

---

## 3. 特征处理

### 3.1 PCA 降维

| 文件 | 功能 |
|------|------|
| `make_pca_features.py` | 提取每个脑区的若干个（默认20）主成分特征，并分别保存训练集和测试集的PCA特征文件。 |

```bash
# 用法示例
python make_pca_features.py --subject subj01 --n_components 20
```

**输出文件**:
- `{subject}_{roi}_betas_tr_PCA{n}.npy` - 训练集 PCA 特征
- `{subject}_{roi}_betas_ave_te_PCA{n}.npy` - 测试集 PCA 特征

### 3.2 邻接矩阵

| 文件 | 功能 |
|------|------|
| `make_pca_adj_matrix.py` | 计算指定被试的10个脑区主成分均值的相关性邻接矩阵。 |

**输出文件**: `{subject}_pca_adj_matrix.npy`

### 3.3 识别准确率

| 文件 | 功能 |
|------|------|
| `identification.py` | 计算图像识别准确率（基于特征相似度匹配） |

```bash
# 用法示例
python identification.py --subject subj01 --method cvpr --usefeat clip
```

---

## 4. 脑解码模型 - Ridge 回归

Ridge 回归是脑解码的基线方法，计算效率高。

### 文件对比

| 文件 | 特点 | 适用场景 |
|------|------|----------|
| `ridge.py` | 基础版本，单次训练全部特征 | 小规模特征 |
| `ridge2.py` | 分批训练，支持断点续训 | 大规模特征 |
| `ridge3.py` | 内存映射 + float16 优化 | 内存受限场景 |
| `ridge4.py` | 默认 batch_size=500 | 独立结果保存 |
| `ridge5.py` | 基于 PCA 特征 | 降维后数据 |

### 使用示例

```bash
# ridge2.py（推荐）
python ridge2.py --target init_latent --roi ventral --subject subj01
python ridge2.py --target c --roi V1 V2 V3 --subject subj01 --batch_size 500 --use_memmap --resume

# ridge3.py（内存优化）
python ridge3.py --target c --roi ventral --subject subj01 --use_memmap --use_float16

# ridge5.py（PCA 特征）
python ridge5.py --subject subj01 --target init_latent --n_components 20
```

### 输出文件

- 模型: `{subject}_{roi}_pipeline_{target}.joblib`
- 预测结果: `{subject}_{roi}_scores_{target}.npy`
- 模型信息: `{subject}_{roi}_model_info_{target}.joblib`

---

## 5. 脑解码模型 - MLP

多层感知机可以学习非线性映射关系。

### 文件对比

| 文件 | 特点 |
|------|------|
| `mlp2.py` | 分批训练，支持内存映射 |
| `mlp3.py` | 基于 PCA 特征 |
| `mlp2_show_acc.py` | 查看模型准确率 |

### 使用示例

```bash
# mlp2.py
python mlp2.py --target c --roi ventral --subject subj01 --batch_size 500 --use_memmap --resume
python mlp2.py --target init_latent --roi early --subject subj01 --mlp_hidden 256

# mlp3.py（PCA 特征）
python mlp3.py --subject subj01 --target init_latent --n_components 20

# 查看准确率
python mlp2_show_acc.py --subject subj01 --roi ventral --target c --method mlp
```

---

## 6. 脑解码模型 - 随机森林

| 文件 | 特点 |
|------|------|
| `rf2.py` | 分批训练随机森林，支持断点续训 |

```bash
python rf2.py --target c --roi ventral --subject subj01 --rf_n_estimators 100
```

---

## 7. 脑解码模型 - 图神经网络 (GNN)

GNN 利用脑区之间的结构关系进行解码。

### 7.1 基础 GNN

| 文件 | 网络结构 | 特点 |
|------|----------|------|
| `gnn_predict.py` | 2层GCN + 2层FC | 基础版本 |
| `gnn_predict2.py` | 2层GCN + 4层FC | 更深网络 |
| `gnn_predict3.py` | 2层GCN + 4层FC（维度递增） | DCGAN 风格 |

**使用示例**:
```bash
python gnn_predict.py --subject subj01 --target init_latent
python gnn_predict2.py --subject subj01 --target init_latent --n_components 20
python gnn_predict3.py --subject subj01 --target init_latent
```

### 7.2 ROI 级别 CNN + GNN

| 文件 | 特点 |
|------|------|
| `gnn_roi_cnn.py` | ROI 级别 MLP 特征提取 + GNN |
| `gnn_roi_cnn2.py` | GPU 加速 + CSV 记录训练指标 |
| `gnn_roi_cnn3.py` | 4层1D卷积 ROI 特征提取 |

**网络结构**:
```
10个ROI → ROICNN(20维) → GCNConv → GCNConv → FC1(512) → FC2(1024) → Output(6400)
```

**使用示例**:
```bash
python gnn_roi_cnn.py --subject subj01 --target init_latent --test
python gnn_roi_cnn2.py --subject subj01 --target init_latent
python gnn_roi_cnn3.py --subject subj01 --target init_latent --roi_out_dim 32
```

---

## 8. 批量运行脚本

### 8.1 Ridge 批量运行

| 文件 | 功能 |
|------|------|
| `run_ridge2_batch.py` | 运行单个脑区列表 |
| `run_ridge2_place.py` | 自动运行所有 ROI，跳过已完成结果 |

```bash
# 运行所有脑区
python run_ridge2_place.py
```

### 8.2 MLP 批量运行

| 文件 | 功能 |
|------|------|
| `run_mlp_place.py` | 自动运行所有 ROI，跳过已完成结果 |

```bash
python run_mlp_place.py
```

---

## 9. 可视化与调试

### 9.1 fMRI 数据查看

| 文件 | 功能 |
|------|------|
| `view_beta_trial.py` | 交互式查看 fMRI 数据切片 |

```bash
# 基本用法
python view_beta_trial.py --subject subj01

# 染色脑区
python view_beta_trial.py --subject subj01 --atlas streams --color_roi early
python view_beta_trial.py --subject subj01 --atlas HCP_MMP1 --color_roi V1
```

### 9.2 网络结构可视化

| 文件 | 功能 |
|------|------|
| `draw_net_frame.py` | 绘制 GNN 网络结构示意图 |

### 9.3 数据调试

| 文件 | 功能 |
|------|------|
| `see_header_river.py` | 查看 numpy 文件头信息（shape、dtype） |
| `altasname_get_river.py` | 列出可用的脑区图谱及 ROI 列表 |

---

## 完整工作流程

### 步骤 1: 数据准备

```bash
# 1.1 提取被试 fMRI 数据（选择一个）
python make_subjmri.py --subject subj01          # streams 图谱
python make_subjmri2.py --subject subj01        # streams + 内存映射
python make_subjmri3.py --subject subj01        # HCP_MMP1 图谱
python make_subjmri4.py --subject subj01        # HCP_MMP1 + 内存映射

# 1.2 生成刺激特征
python make_subjstim.py --subject subj01  # 将特征数据按 NSD 实验设计划分训练/测试集

# 1.3 提取测试图片
python extract_test_images.py --subject subj01
```

### 步骤 2: 特征提取

```bash
# 2.1 提取 SD 特征（需要 GPU）
python img2feat_sd.py --imgidx 0 10000 --gpu 0 # 提取 

# 2.2 提取视觉特征（在解码后使用）
python img2feat_decoded.py --subject subj01 --method cvpr --gpu 0 # 是评分的时候使用的
```

### 步骤 3: 特征处理（可选）

```bash
# 3.1 PCA 降维
python make_pca_features.py --subject subj01 --n_components 20 # 提取每个脑区的前20个主成分特征

# 3.2 构建邻接矩阵
python make_pca_adj_matrix.py --subject subj01 # 计算指定被试(subject)的10个脑区主成分均值的相关性邻接矩阵。
```

### 步骤 4: 模型训练

```bash
# 4.1 Ridge 回归
python ridge2.py --target init_latent --roi ventral --subject subj01

# 4.2 MLP
python mlp2.py --target init_latent --roi ventral --subject subj01

# 4.3 GNN
python gnn_predict.py --subject subj01 --target init_latent
python gnn_roi_cnn3.py --subject subj01 --target init_latent
```

### 步骤 5: 结果评估

```bash
# 5.1 计算识别准确率
python identification.py --subject subj01 --method cvpr --usefeat clip

# 5.2 查看模型性能
python mlp2_show_acc.py --subject subj01 --roi ventral --target init_latent --method ridge
```

---

## 数据目录结构

```
项目根目录/
├── codes/
│   └── utils/                    # 本文档描述的工具脚本
├── nsd/                          # NSD 数据集
├── mrifeat/                      # fMRI 特征
│   └── {subject}/
│       ├── {subject}_stims.npy
│       ├── {subject}_stims_ave.npy
│       ├── {subject}_{roi}_betas_tr.npy
│       └── ...
├── nsdfeat/                     # 刺激特征
│   ├── init_latent/
│   ├── c/
│   └── subjfeat/
├── decoded/                     # 解码结果
│   └── {subject}/
│       ├── {subject}_{roi}_scores_{target}.npy
│       └── ...
├── test_images/                # 测试图片
│   └── {subject}/
└── docs/                       # 本文档
```

---

## 参数说明

### 常用参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--subject` | 被试名称 | subj01, subj02, subj05, subj07 |
| `--target` | 预测目标 | init_latent, c |
| `--roi` | 脑区 | ventral, early, V1, V2... |
| `--batch_size` | 批次大小 | 500, 1000 |
| `--n_components` | PCA 主成分数 | 20, 50 |
| `--gpu` | GPU 编号 | 0, 1 |
| `--use_memmap` | 使用内存映射 | - |
| `--resume` | 断点续训 | - |
| `--test` | 测试模式（少量数据） | - |

### target 选项

| 值 | 描述 |
|----|------|
| `init_latent` | Stable Diffusion 初始 latent 特征 |
| `c` | 文本条件特征 |
| `c_text` | 文本特征 |
| `c_gan` | GAN 特征 |
| `c_depth` | 深度图特征 |

---

## 版本历史

- v1.0 (2026-03-18): 初始版本，包含 37 个工具脚本的完整文档
