# 导入必要的库
'''
# 基本使用（与原脚本相同）
python ridge3.py --target c --roi ventral --subject subj01

# 使用内存映射解决内存问题
python ridge3.py --target c --roi ventral --subject subj01 --use_memmap

# 使用float16进一步减少内存
python ridge3.py --target c --roi ventral --subject subj01 --use_memmap --use_float16
'''
import argparse, os, gc  # 添加gc用于垃圾回收 - 【新增】
import numpy as np
import joblib
from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV
from himalaya.scoring import correlation_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import tempfile  # 添加临时文件支持 - 【新增】
import shutil   # 添加文件操作支持 - 【新增】

def create_memory_mapped_array(data, temp_dir, filename):
    """
    创建内存映射数组以减少内存使用 - 【新增函数】
    """
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    filepath = os.path.join(temp_dir, filename)
    # 如果文件已存在则删除
    if os.path.exists(filepath):
        os.remove(filepath)
    
    # 创建内存映射文件
    mmap_array = np.memmap(filepath, dtype='float32', mode='w+', shape=data.shape)
    mmap_array[:] = data.astype('float32')
    mmap_array.flush()
    del mmap_array
    gc.collect()
    
    # 重新以只读模式打开
    return np.memmap(filepath, dtype='float32', mode='r', shape=data.shape)

def optimize_memory_settings():
    """
    优化内存使用设置 - 【新增函数】
    """
    # 限制numpy使用的线程数，减少内存占用
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['NUMEXPR_MAX_THREADS'] = '4'

def main():
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
    
    # 新增参数 - 【新增】
    parser.add_argument(
        "--use_memmap",
        action='store_true',
        help="使用内存映射减少内存占用"
    )
    parser.add_argument(
        "--use_float16",
        action='store_true',
        help="使用float16数据类型减少内存占用"
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default=None,
        help="临时文件目录，默认使用系统临时目录"
    )

    # 解析命令行参数
    opt = parser.parse_args()
    target = opt.target  # 目标特征类型
    roi = opt.roi        # 感兴趣区域列表

    # 优化内存设置 - 【新增】
    optimize_memory_settings()

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

    # 设置临时目录 - 【新增】
    if opt.temp_dir:
        temp_dir = opt.temp_dir
    else:
        temp_dir = os.path.join(tempfile.gettempdir(), f'ridge_temp_{subject}')
    os.makedirs(temp_dir, exist_ok=True)

    # 初始化训练和测试数据列表
    X = []     # 训练集fMRI数据
    X_te = []  # 测试集fMRI数据
    
    print("Loading fMRI data...")  # 【新增进度提示】
    
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
    
    print("Loading target features...")  # 【新增进度提示】
    
    # 加载目标特征数据（要预测的特征）
    # 训练集：每个样本的特征（用于训练模型）
    Y = np.load(f'{featdir}/{subject}_each_{target}_tr.npy').astype("float32").reshape([X.shape[0], -1])
    # 测试集：平均后的特征（用于评估预测性能）
    Y_te = np.load(f'{featdir}/{subject}_ave_{target}_te.npy').astype("float32").reshape([X_te.shape[0], -1])
    
    # 打印当前处理的信息和数据维度
    total_memory_gb = (X.nbytes + Y.nbytes + X_te.nbytes + Y_te.nbytes) / 1024**3
    print(f'Now making decoding model for... {subject}:  {roi}, {target}')
    print(f'X {X.shape}, Y {Y.shape}, X_te {X_te.shape}, Y_te {Y_te.shape}')
    print(f'Estimated total memory usage: {total_memory_gb:.2f} GB')
    
    # 数据类型优化 - 【新增】
    if opt.use_float16:
        print("Converting to float16 to reduce memory usage...")
        X = X.astype(np.float16)
        Y = Y.astype(np.float16) 
        X_te = X_te.astype(np.float16)
        Y_te = Y_te.astype(np.float16)
        print("Memory usage reduced by ~50%")
    
    # 内存映射优化 - 【新增】
    if opt.use_memmap:
        print("Creating memory-mapped arrays...")
        
        # 创建内存映射数组
        X_mmap = create_memory_mapped_array(X, temp_dir, 'X_train.dat')
        Y_mmap = create_memory_mapped_array(Y, temp_dir, 'Y_train.dat')
        X_te_mmap = create_memory_mapped_array(X_te, temp_dir, 'X_test.dat')
        Y_te_mmap = create_memory_mapped_array(Y_te, temp_dir, 'Y_test.dat')
        
        # 清理原始数组，释放内存
        del X, Y, X_te, Y_te
        gc.collect()
        
        # 使用内存映射数组
        X, Y, X_te, Y_te = X_mmap, Y_mmap, X_te_mmap, Y_te_mmap
        print("Memory-mapped arrays created successfully")
    
    try:
        print("Training model...")  # 【新增进度提示】
        
        # 训练模型：使用训练集数据拟合脊回归模型
        pipeline.fit(X, Y)
        
        print("Model training completed successfully")  # 【新增成功提示】
        
        # 保存训练好的完整pipeline模型
        model_path = f'{savedir}/{subject}_{"_".join(roi)}_pipeline_{target}.joblib'
        joblib.dump(pipeline, model_path)
        print(f'Pipeline model saved to: {model_path}')
        
        print("Making predictions...")  # 【新增进度提示】
        
        # 预测：使用训练好的模型对测试集进行预测
        scores = pipeline.predict(X_te)
        
        # 计算预测性能：计算预测值与真实值之间的相关系数
        rs = correlation_score(Y_te.T, scores.T)  # 转置以确保正确的维度匹配
        print(f'Prediction accuracy is: {np.mean(rs):3.3}')  # 打印平均预测准确率

        # 保存预测结果到文件
        np.save(f'{savedir}/{subject}_{"_".join(roi)}_scores_{target}.npy', scores)
        
        # 保存模型相关信息 - 【增强】
        model_info = {
            'subject': subject,
            'roi': roi,
            'target': target,
            'X_shape': X.shape,
            'Y_shape': Y.shape,
            'prediction_accuracy': np.mean(rs),
            'best_alpha': pipeline.named_steps['ridgecv'].alpha_,
            'used_memmap': opt.use_memmap,  # 【新增】
            'used_float16': opt.use_float16,  # 【新增】
            'memory_optimized': True  # 【新增】
        }
        joblib.dump(model_info, f'{savedir}/{subject}_{"_".join(roi)}_model_info_{target}.joblib')
        
        print("Processing completed successfully!")  # 【新增完成提示】
        
    except MemoryError as e:
        print(f"Memory error occurred: {e}")  # 【新增错误处理】
        print("建议尝试:")
        print("1. 使用 --use_memmap 参数")
        print("2. 使用 --use_float16 参数")
        print("3. 同时使用两个参数: --use_memmap --use_float16")
        raise
    
    finally:
        # 清理临时文件 - 【新增】
        if opt.use_memmap and os.path.exists(temp_dir):
            print("Cleaning up temporary files...")
            try:
                shutil.rmtree(temp_dir)
                print("Temporary files cleaned up")
            except Exception as e:
                print(f"Warning: Could not clean up temporary files: {e}")

# 程序入口点
if __name__ == "__main__":
    main()  # 运行主函数