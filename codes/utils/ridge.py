# 导入必要的库
import argparse, os  # 命令行参数解析和操作系统相关操作
import numpy as np  # 数值计算库
import joblib  # 用于保存和加载机器学习模型
from himalaya.backend import set_backend  # Himalaya后端设置
from himalaya.ridge import RidgeCV  # 带交叉验证的脊回归模型
from himalaya.scoring import correlation_score  # 相关性评分函数
from sklearn.pipeline import make_pipeline  # 创建机器学习管道
from sklearn.preprocessing import StandardScaler  # 数据标准化预处理

def main():
    """
    主函数：执行脊回归模型训练和预测
    用于从fMRI数据预测视觉特征（如CLIP特征、深度特征等）
    """

    """
    主函数：执行脊回归模型训练和预测
    用于从fMRI数据预测视觉特征（如CLIP特征、深度特征等）
    """

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()

    # 添加目标变量参数 - 指定要预测的特征类型
    parser.add_argument(
        "--target",
        type=str,
        default='',
        help="Target variable",  # 目标变量，如'c'(CLIP特征)、'init_latent'(初始潜在特征)等
    )
    # 添加ROI参数 - 指定要使用的大脑区域
    parser.add_argument(
        "--roi",
        required=True,
        type=str,
        nargs="*",
        help="use roi name",  # 感兴趣区域名称，如'early'、'ventral'、'lateral'等
    )
    # 添加被试参数 - 指定分析的被试
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )

    # 解析命令行参数
    opt = parser.parse_args()
    target = opt.target  # 目标特征类型
    roi = opt.roi        # 感兴趣区域列表

    # 设置Himalaya计算后端为numpy
    backend = set_backend("numpy", on_error="warn")
    subject = opt.subject  # 被试ID

    # 根据目标类型选择合适的正则化参数alpha
    # 不同特征类型需要不同的正则化强度
    if target == 'c' or target == 'init_latent':  # CVPR相关特征
        alpha = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]  # 较小的alpha值
    else:  # text / GAN / depth decoding (体素数量较多的情况)
        alpha = [10000, 20000, 40000]  # 较大的alpha值，用于高维度特征

    # 创建带交叉验证的脊回归模型
    ridge = RidgeCV(alphas=alpha)

    # 创建数据预处理管道：标准化（零均值，单位方差）
    preprocess_pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),  # 标准化处理
    )
    
    # 创建完整的机器学习管道：预处理 + 脊回归
    pipeline = make_pipeline(
        preprocess_pipeline,  # 数据预处理步骤
        ridge,               # 脊回归模型
    )
    
    # 定义数据路径
    mridir = f'../../mrifeat/{subject}/'      # fMRI数据目录
    featdir = '../../nsdfeat/subjfeat/'       # 特征数据目录  
    savedir = f'../..//decoded/{subject}/'    # 结果保存目录
    os.makedirs(savedir, exist_ok=True)       # 创建保存目录（如果不存在）

    # 初始化训练和测试数据列表
    X = []     # 训练集fMRI数据
    X_te = []  # 测试集fMRI数据
    
    # 遍历所有指定的ROI，加载对应的fMRI数据
    for croi in roi:
        # 根据目标类型选择合适的数据文件
        if 'conv' in target:  # 对于GAN任务，由于特征维度很大，使用平均后的特征
            cX = np.load(f'{mridir}/{subject}_{croi}_betas_ave_tr.npy').astype("float32")
        else:  # 对于其他任务，使用原始特征
            cX = np.load(f'{mridir}/{subject}_{croi}_betas_tr.npy').astype("float32")
        
        # 加载测试集数据（总是使用平均后的数据）
        cX_te = np.load(f'{mridir}/{subject}_{croi}_betas_ave_te.npy').astype("float32")
        
        # 将当前ROI的数据添加到列表中
        X.append(cX)
        X_te.append(cX_te)
    
    # 水平拼接所有ROI的数据，形成最终的特征矩阵
    X = np.hstack(X)      # 训练集：(样本数, 所有ROI的体素数)
    X_te = np.hstack(X_te) # 测试集：(样本数, 所有ROI的体素数)
    
    # 加载目标特征数据（要预测的特征）
    # 训练集：每个样本的特征（用于训练模型）
    Y = np.load(f'{featdir}/{subject}_each_{target}_tr.npy').astype("float32").reshape([X.shape[0], -1])
    # 测试集：平均后的特征（用于评估预测性能）
    Y_te = np.load(f'{featdir}/{subject}_ave_{target}_te.npy').astype("float32").reshape([X_te.shape[0], -1])
    
    # 打印当前处理的信息和数据维度
    print(f'Now making decoding model for... {subject}:  {roi}, {target}')
    print(f'X {X.shape}, Y {Y.shape}, X_te {X_te.shape}, Y_te {Y_te.shape}')
    
    # 训练模型：使用训练集数据拟合脊回归模型
    pipeline.fit(X, Y)
    
    # 保存训练好的完整pipeline模型
    model_path = f'{savedir}/{subject}_{"_".join(roi)}_pipeline_{target}.joblib'
    joblib.dump(pipeline, model_path)
    print(f'Pipeline model saved to: {model_path}')
    
    # 预测：使用训练好的模型对测试集进行预测
    scores = pipeline.predict(X_te)
    
    # 计算预测性能：计算预测值与真实值之间的相关系数
    rs = correlation_score(Y_te.T, scores.T)  # 转置以确保正确的维度匹配
    print(f'Prediction accuracy is: {np.mean(rs):3.3}')  # 打印平均预测准确率

    # 保存预测结果到文件
    np.save(f'{savedir}/{subject}_{"_".join(roi)}_scores_{target}.npy', scores)
    
    # 可选：保存模型相关信息
    model_info = {
        'subject': subject,
        'roi': roi,
        'target': target,
        'X_shape': X.shape,
        'Y_shape': Y.shape,
        'prediction_accuracy': np.mean(rs),
        'best_alpha': pipeline.named_steps['ridgecv'].alpha_
    }
    joblib.dump(model_info, f'{savedir}/{subject}_{"_".join(roi)}_model_info_{target}.joblib')

# 程序入口点
if __name__ == "__main__":
    main()  # 运行主函数
