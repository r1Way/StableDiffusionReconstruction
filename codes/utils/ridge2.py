# 导入必要的库
'''
# 基本使用（默认批次大小1000）
python ridge2.py --target c --roi ventral --subject subj01

# 使用更小的批次大小（如果仍有内存问题）
python ridge2.py --target c --roi ventral --subject subj01 --batch_size 500

# 使用内存映射进一步优化内存
python ridge2.py --target c --roi ventral --subject subj01 --batch_size 1000 --use_memmap

# 设置随机种子确保可复现
python ridge2.py --target c --roi ventral --subject subj01 --seed 42

# 断点续训（自动跳过已完成的批次）
python ridge2.py --target c --roi ventral --subject subj01 --resume
'''
import argparse, os, gc  # 添加垃圾回收
import numpy as np
import joblib
from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV
from himalaya.scoring import correlation_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm  # 添加进度条
import random  # 添加随机种子支持

def set_random_seeds(seed):
    """设置所有随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    # 设置环境变量确保numpy多线程的一致性
    os.environ['PYTHONHASHSEED'] = str(seed)

def check_batch_completed(savedir, subject, roi, target, batch_idx):
    """检查指定批次是否已完成训练"""
    model_path = f'{savedir}/{subject}_{"_".join(roi)}_pipeline_{target}_batch_{batch_idx}.joblib'
    return os.path.exists(model_path)

def get_completed_batches(savedir, subject, roi, target, n_batches):
    completed = []
    for batch_idx in range(n_batches):  # 遍历所有批次
        if check_batch_completed(savedir, subject, roi, target, batch_idx):
            completed.append(batch_idx)  # 收集已完成的批次号
    return completed

def load_batch_model(savedir, subject, roi, target, batch_idx):
    """加载已保存的批次模型"""
    model_path = f'{savedir}/{subject}_{"_".join(roi)}_pipeline_{target}_batch_{batch_idx}.joblib'
    try:
        return joblib.load(model_path)
    except Exception as e:
        print(f"加载批次 {batch_idx} 模型失败: {e}")
        return None

def create_memory_mapped_array(data, temp_dir, filename):
    """创建内存映射数组以减少内存使用"""
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

def train_ridge_batch(X, Y_batch, alpha_values, batch_idx):
    """训练单个批次的脊回归模型"""
    try:
        # 创建脊回归模型
        ridge = RidgeCV(alphas=alpha_values)
        
        # 创建预处理管道
        preprocess_pipeline = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
        )
        
        # 创建完整管道
        pipeline = make_pipeline(
            preprocess_pipeline,
            ridge,
        )
        
        # 训练模型
        pipeline.fit(X, Y_batch)
        
        # 立即清理内存
        gc.collect()
        
        return pipeline, True
        
    except Exception as e:
        print(f"批次 {batch_idx} 训练失败: {str(e)}")
        # 清理内存
        gc.collect()
        return None, False

def main():
    """
    主函数：执行内存优化的脊回归模型训练和预测
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default='', help="Target variable")
    parser.add_argument("--roi", required=True, type=str, nargs="*", help="use roi name")
    parser.add_argument("--subject", type=str, default=None, help="subject name")
    parser.add_argument("--batch_size", type=int, default=1000, help="批次大小，用于分批处理Y")
    parser.add_argument("--use_memmap", action='store_true', help="使用内存映射减少内存占用")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，确保可复现性")
    parser.add_argument("--resume", action='store_true', help="断点续训，跳过已完成的批次")
    
    opt = parser.parse_args()
    target = opt.target
    roi = opt.roi
    batch_size = opt.batch_size
    
    # 设置随机种子确保可复现性
    set_random_seeds(opt.seed)
    print(f"设置随机种子: {opt.seed}")
    
    # 设置Himalaya计算后端
    backend = set_backend("numpy", on_error="warn")
    subject = opt.subject
    
    # 选择alpha参数
    if target == 'c' or target == 'init_latent':
        alpha = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    else:
        alpha = [10000, 20000, 40000]
    
    # 定义路径
    mridir = f'../../mrifeat/{subject}/'
    featdir = '../../nsdfeat/subjfeat/'
    savedir = f'../../decoded/{subject}/'
    temp_dir = f'../../temp/{subject}/'  # 临时文件目录
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    print("Loading fMRI data...")
    # 加载fMRI数据
    X = []
    X_te = []
    
    for croi in roi:
        if 'conv' in target:
            cX = np.load(f'{mridir}/{subject}_{croi}_betas_ave_tr.npy').astype("float32")
        else:
            cX = np.load(f'{mridir}/{subject}_{croi}_betas_tr.npy').astype("float32")
        
        cX_te = np.load(f'{mridir}/{subject}_{croi}_betas_ave_te.npy').astype("float32")
        
        X.append(cX)
        X_te.append(cX_te)
    
    # 拼接ROI数据
    X = np.hstack(X).astype('float32')
    X_te = np.hstack(X_te).astype('float32')
    
    print("Loading target features...")
    # 加载目标特征
    Y = np.load(f'{featdir}/{subject}_each_{target}_tr.npy').astype("float32").reshape([X.shape[0], -1])
    Y_te = np.load(f'{featdir}/{subject}_ave_{target}_te.npy').astype("float32").reshape([X_te.shape[0], -1])
    
    print(f'Processing data for... {subject}: {roi}, {target}')
    print(f'X {X.shape}, Y {Y.shape}, X_te {X_te.shape}, Y_te {Y_te.shape}')
    print(f'Estimated memory usage: {(Y.nbytes / 1024**3):.2f} GB')
    
    # 使用内存映射优化（可选）
    if opt.use_memmap:
        print("Creating memory-mapped arrays...")
        X = create_memory_mapped_array(X, temp_dir, 'X_train.dat')
        Y = create_memory_mapped_array(Y, temp_dir, 'Y_train.dat')
        X_te = create_memory_mapped_array(X_te, temp_dir, 'X_test.dat')
        Y_te = create_memory_mapped_array(Y_te, temp_dir, 'Y_test.dat')
    
    # 计算批次数量
    n_features = Y.shape[1]
    n_batches = (n_features + batch_size - 1) // batch_size
    
    print(f"将分 {n_batches} 个批次处理，每批次最多 {batch_size} 个特征")
    
    # 检查断点续训
    completed_batches = []
    if opt.resume:
        completed_batches = get_completed_batches(savedir, subject, roi, target, n_batches)
        if completed_batches:
            print(f"发现已完成的批次: {completed_batches}")
            print(f"将跳过这些批次，继续训练剩余的 {n_batches - len(completed_batches)} 个批次")
        else:
            print("未发现已完成的批次，从头开始训练")
    
    # 存储每个批次的模型和信息
    models = []
    batch_info = []
    
    # 分批次训练
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_features)
        
        # 检查是否需要跳过此批次
        if opt.resume and batch_idx in completed_batches:
            print(f"跳过已完成的批次 {batch_idx + 1}/{n_batches}")
            # 加载已保存的模型
            model = load_batch_model(savedir, subject, roi, target, batch_idx)
            if model is not None:
                models.append(model)
                batch_info.append((start_idx, end_idx))
            continue
        
        print(f"\n处理批次 {batch_idx + 1}/{n_batches}: 特征 {start_idx} 到 {end_idx-1}")
        
        try:
            # 提取当前批次的Y数据
            Y_batch = Y[:, start_idx:end_idx].copy()  # 使用copy确保数据连续
            
            # 训练当前批次
            model, success = train_ridge_batch(X, Y_batch, alpha, batch_idx)
            
            if success:
                models.append(model)
                batch_info.append((start_idx, end_idx))
                
                # 保存当前批次的模型
                model_path = f'{savedir}/{subject}_{"_".join(roi)}_pipeline_{target}_batch_{batch_idx}.joblib'
                joblib.dump(model, model_path)
                print(f'批次 {batch_idx} 模型已保存到: {model_path}')
            else:
                print(f'批次 {batch_idx} 训练失败，跳过')
            
            # 立即清理当前批次的内存
            del Y_batch
            if not success:
                del model
            gc.collect()
            
        except Exception as e:
            print(f"批次 {batch_idx} 处理出错: {e}")
            # 清理内存
            gc.collect()
            continue
    
    print(f"\n成功训练了 {len(models)} 个批次模型")
    
    # 对测试集进行预测
    print("开始预测...")
    all_predictions = []
    batch_accuracies = []
    
    for batch_idx, (model, (start_idx, end_idx)) in enumerate(zip(models, batch_info)):
        print(f"预测批次 {batch_idx + 1}/{len(models)}")
        
        try:
            # 对当前批次进行预测
            Y_te_batch = Y_te[:, start_idx:end_idx]
            predictions = model.predict(X_te)
            all_predictions.append(predictions)
            
            # 计算当前批次的性能
            if Y_te_batch.shape[1] > 0:
                rs_batch = correlation_score(Y_te_batch.T, predictions.T)
                batch_accuracy = np.mean(rs_batch)
                batch_accuracies.append(batch_accuracy)
                print(f'批次 {batch_idx} 预测准确率: {batch_accuracy:.3f}')
            
            # 立即清理内存
            del predictions, Y_te_batch, rs_batch
            gc.collect()
            
        except Exception as e:
            print(f"批次 {batch_idx} 预测出错: {e}")
            # 清理内存
            gc.collect()
            continue
    
    # 合并所有预测结果
    if all_predictions:
        scores = np.hstack(all_predictions)
        
        # 计算总体性能
        rs = correlation_score(Y_te.T, scores.T)
        overall_accuracy = np.mean(rs)
        print(f'总体预测准确率: {overall_accuracy:.3f}')
        
        # 保存预测结果
        np.save(f'{savedir}/{subject}_{"_".join(roi)}_scores_{target}.npy', scores)
        
        # 保存模型信息
        model_info = {
            'subject': subject,
            'roi': roi,
            'target': target,
            'X_shape': X.shape,
            'Y_shape': Y.shape,
            'n_batches': len(models),
            'batch_size': batch_size,
            'prediction_accuracy': overall_accuracy,
            'batch_info': batch_info,
            'batch_accuracies': batch_accuracies,
            'seed': opt.seed,  # 保存随机种子
            'completed_time': str(gc.get_count())  # 添加完成时间戳
        }
        joblib.dump(model_info, f'{savedir}/{subject}_{"_".join(roi)}_model_info_{target}.joblib')
        
        print(f'预测结果已保存到: {savedir}')
        
        # 清理大型数组
        del scores, rs, all_predictions
        gc.collect()
        
    else:
        print("没有成功的预测结果")
    
    # 清理所有主要数据结构
    del X, Y, X_te, Y_te, models
    gc.collect()
    
    # 清理临时文件
    if opt.use_memmap:
        print("清理临时文件...")
        import shutil
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print("临时文件清理完成")
        except Exception as e:
            print(f"清理临时文件时出错: {e}")
    
    print("程序执行完成，内存已清理")

if __name__ == "__main__":
    main()
