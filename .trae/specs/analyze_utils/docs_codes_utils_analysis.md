# codes/utils 工具脚本文档规范

## 概述

本文档分析 StableDiffusionReconstruction 项目中 `codes/utils` 目录下各个 Python 脚本的用途、功能和区别。该项目是一个脑解码（Brain Decoding）项目，通过 fMRI 信号重建 Stable Diffusion 生成的图像。

## 文件分类

根据功能用途，utils 目录下的脚本可分为以下几类：

### 1. 数据提取与预处理
| 文件名 | 功能描述 |
|--------|----------|
| `extract_test_images.py` | 从 NSD 数据集中提取测试集图片 |
| `make_subjstim.py` | 生成被试的刺激特征矩阵 |
| `make_subjmri.py` | 使用 streams 图谱提取被试 fMRI 数据 |
| `make_subjmri2.py` | 使用 streams 图谱，支持内存映射 |
| `make_subjmri3.py` | 使用 HCP_MMP1 图谱提取 fMRI 数据 |
| `make_subjmri4.py` | 使用 HCP_MMP1 图谱，支持内存映射 |

### 2. 特征提取
| 文件名 | 功能描述 |
|--------|----------|
| `img2feat_sd.py` | 提取 Stable Diffusion 的 latent 特征和文本条件特征 |
| `img2feat_decoded.py` | 提取解码图像的多种视觉特征（Inception/AlexNet/CLIP） |

### 3. 特征处理
| 文件名 | 功能描述 |
|--------|----------|
| `make_pca_features.py` | 对 fMRI 数据进行 PCA 降维 |
| `make_pca_adj_matrix.py` | 计算脑区之间的邻接矩阵（用于 GNN） |
| `identification.py` | 计算图像识别准确率（相似度匹配） |

### 4. 脑解码模型 - Ridge 回归
| 文件名 | 功能描述 |
|--------|----------|
| `ridge.py` | 基础 Ridge 回归模型 |
| `ridge2.py` | 分批训练，支持断点续训 |
| `ridge3.py` | 支持内存映射和 float16 优化 |
| `ridge4.py` | 默认 batch_size=500，结果保存到独立目录 |
| `ridge5.py` | 基于 PCA 特征的 Ridge 回归 |

### 5. 脑解码模型 - MLP
| 文件名 | 功能描述 |
|--------|----------|
| `mlp2.py` | 分批 MLP 回归训练 |
| `mlp3.py` | 基于 PCA 特征的 MLP 回归 |
| `mlp2_show_acc.py` | 查看 MLP/Ridge/RF 模型的预测准确率 |

### 6. 脑解码模型 - 随机森林
| 文件名 | 功能描述 |
|--------|----------|
| `rf2.py` | 分批随机森林回归训练 |

### 7. 脑解码模型 - 图神经网络 (GNN)
| 文件名 | 功能描述 |
|--------|----------|
| `gnn_predict.py` | 基础 GNN 模型（2层GCN + 2层FC） |
| `gnn_predict2.py` | 深度 GNN（2层GCN + 4层FC） |
| `gnn_predict3.py` | 更深的 GNN（维度递增结构） |
| `gnn_roi_cnn.py` | ROI级别 CNN + GNN 融合（基础版） |
| `gnn_roi_cnn2.py` | 支持 GPU 加速的 ROI-GNN |
| `gnn_roi_cnn3.py` | 4层1D卷积的 ROI-GNN |

### 8. 批量运行脚本
| 文件名 | 功能描述 |
|--------|----------|
| `run_ridge2_batch.py` | 批量运行 ridge2.py |
| `run_ridge2_place.py` | 批量运行多个脑区的 ridge2 |
| `run_mlp_place.py` | 批量运行多个脑区的 mlp2 |

### 9. 可视化与调试
| 文件名 | 功能描述 |
|--------|----------|
| `draw_net_frame.py` | 绘制网络结构示意图 |
| `see_header_river.py` | 查看 numpy 文件头信息 |
| `view_beta_trial.py` | 交互式查看 fMRI 数据切片 |
| `altasname_get_river.py` | 列出可用的脑区图谱 |

## 使用流程

### 完整工作流程

1. **数据准备**
   - 运行 `make_subjstim.py` 生成刺激特征
   - 运行 `make_subjmri*.py` 提取 fMRI 数据

2. **特征提取**
   - 运行 `img2feat_sd.py` 提取 SD 特征
   - 运行 `img2feat_decoded.py` 提取视觉特征

3. **特征处理**
   - 运行 `make_pca_features.py` PCA 降维
   - 运行 `make_pca_adj_matrix.py` 构建邻接矩阵
   - 运行 `extract_test_images.py` 提取测试图片

4. **模型训练与预测**
   - 选择合适的解码模型（Ridge/MLP/RF/GNN）
   - 运行对应的训练脚本

5. **结果评估**
   - 运行 `identification.py` 计算识别准确率
   - 运行 `mlp2_show_acc.py` 查看模型性能

## 输出文件位置

- fMRI 特征：`../../mrifeat/{subject}/`
- 刺激特征：`../../nsdfeat/subjfeat/`
- 解码结果：`../../decoded/{subject}/`
- 测试图片：`../../test_images/{subject}/`
