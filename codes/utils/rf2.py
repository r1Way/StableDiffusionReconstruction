import argparse, os, gc
import numpy as np
import joblib
from himalaya.backend import set_backend
from himalaya.scoring import correlation_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import random

def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def check_batch_completed(savedir, subject, roi, target, batch_idx):
    model_path = f'{savedir}/{subject}_{"_".join(roi)}_pipeline_{target}_rf_batch_{batch_idx}.joblib'
    return os.path.exists(model_path)

def get_completed_batches(savedir, subject, roi, target, n_batches):
    completed = []
    for batch_idx in range(n_batches):
        if check_batch_completed(savedir, subject, roi, target, batch_idx):
            completed.append(batch_idx)
    return completed

def load_batch_model(savedir, subject, roi, target, batch_idx):
    model_path = f'{savedir}/{subject}_{"_".join(roi)}_pipeline_{target}_rf_batch_{batch_idx}.joblib'
    try:
        return joblib.load(model_path)
    except Exception as e:
        print(f"加载批次 {batch_idx} 模型失败: {e}")
        return None

def create_memory_mapped_array(data, temp_dir, filename):
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    filepath = os.path.join(temp_dir, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    mmap_array = np.memmap(filepath, dtype='float32', mode='w+', shape=data.shape)
    mmap_array[:] = data.astype('float32')
    mmap_array.flush()
    del mmap_array
    gc.collect()
    return np.memmap(filepath, dtype='float32', mode='r', shape=data.shape)

def load_large_npy_to_memmap(npy_path, shape, dtype, temp_dir, filename, chunk_size=1000):
    """将大npy文件分块加载到memmap，避免一次性占用大量内存"""
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    filepath = os.path.join(temp_dir, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    mmap_array = np.memmap(filepath, dtype=dtype, mode='w+', shape=shape)
    npy = np.load(npy_path, mmap_mode='r')
    n_rows = shape[0]
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        mmap_array[start:end] = npy[start:end]
    mmap_array.flush()
    del mmap_array, npy
    gc.collect()
    return np.memmap(filepath, dtype=dtype, mode='r', shape=shape)

def train_rf_batch(X, Y_batch, batch_idx, rf_params):
    try:
        rf = RandomForestRegressor(**rf_params)
        preprocess_pipeline = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
        )
        pipeline = make_pipeline(
            preprocess_pipeline,
            rf,
        )
        pipeline.fit(X, Y_batch)
        gc.collect()
        return pipeline, True
    except Exception as e:
        print(f"批次 {batch_idx} 训练失败: {str(e)}")
        gc.collect()
        return None, False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default='', help="Target variable")
    parser.add_argument("--roi", required=True, type=str, nargs="*", help="use roi name")
    parser.add_argument("--subject", type=str, default=None, help="subject name")
    parser.add_argument("--batch_size", type=int, default=1000, help="批次大小，用于分批处理Y")
    parser.add_argument("--use_memmap", action='store_true', help="使用内存映射减少内存占用")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，确保可复现性")
    parser.add_argument("--resume", action='store_true', help="断点续训，跳过已完成的批次")
    parser.add_argument("--rf_n_estimators", type=int, default=100, help="随机森林树数量")
    parser.add_argument("--rf_max_depth", type=int, default=None, help="最大树深度")
    parser.add_argument("--rf_n_jobs", type=int, default=-1, help="并行线程数")
    opt = parser.parse_args()
    target = opt.target
    roi = opt.roi
    batch_size = opt.batch_size

    set_random_seeds(opt.seed)
    print(f"设置随机种子: {opt.seed}")

    backend = set_backend("numpy", on_error="warn")
    subject = opt.subject

    mridir = f'../../mrifeat/{subject}/'
    featdir = '../../nsdfeat/subjfeat/'
    savedir = f'../../decoded/{subject}/'
    temp_dir = f'../../temp/{subject}/'
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    print("Loading fMRI data...")
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
    X = np.hstack(X).astype('float32')
    X_te = np.hstack(X_te).astype('float32')

    print("Loading target features...")
    y_tr_path = f'{featdir}/{subject}_each_{target}_tr.npy'
    y_te_path = f'{featdir}/{subject}_ave_{target}_te.npy'
    # 用mmap_mode='r'只读方式获取shape
    y_tr_shape = np.load(y_tr_path, mmap_mode='r').reshape([X.shape[0], -1]).shape
    y_te_shape = np.load(y_te_path, mmap_mode='r').reshape([X_te.shape[0], -1]).shape

    if opt.use_memmap:
        print("Creating memory-mapped arrays for large Y...")
        X = create_memory_mapped_array(X, temp_dir, 'X_train.dat')
        X_te = create_memory_mapped_array(X_te, temp_dir, 'X_test.dat')
        Y = load_large_npy_to_memmap(y_tr_path, y_tr_shape, 'float32', temp_dir, 'Y_train.dat', chunk_size=1000)
        Y_te = load_large_npy_to_memmap(y_te_path, y_te_shape, 'float32', temp_dir, 'Y_test.dat', chunk_size=1000)
    else:
        Y = np.load(y_tr_path).astype("float32").reshape([X.shape[0], -1])
        Y_te = np.load(y_te_path).astype("float32").reshape([X_te.shape[0], -1])

    print(f'Processing data for... {subject}: {roi}, {target}')
    print(f'X {X.shape}, Y {Y.shape}, X_te {X_te.shape}, Y_te {Y_te.shape}')
    print(f'Estimated memory usage: {(Y.nbytes / 1024**3):.2f} GB')

    n_features = Y.shape[1]
    n_batches = (n_features + batch_size - 1) // batch_size
    print(f"将分 {n_batches} 个批次处理，每批次最多 {batch_size} 个特征")

    completed_batches = []
    if opt.resume:
        completed_batches = get_completed_batches(savedir, subject, roi, target, n_batches)
        if completed_batches:
            print(f"发现已完成的批次: {completed_batches}")
            print(f"将跳过这些批次，继续训练剩余的 {n_batches - len(completed_batches)} 个批次")
        else:
            print("未发现已完成的批次，从头开始训练")

    models = []
    batch_info = []

    rf_params = {
        "n_estimators": opt.rf_n_estimators,
        "max_depth": opt.rf_max_depth,
        "n_jobs": opt.rf_n_jobs,
        "random_state": opt.seed,
    }

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_features)
        if opt.resume and batch_idx in completed_batches:
            print(f"跳过已完成的批次 {batch_idx + 1}/{n_batches}")
            model = load_batch_model(savedir, subject, roi, target, batch_idx)
            if model is not None:
                models.append(model)
                batch_info.append((start_idx, end_idx))
            continue
        print(f"\n处理批次 {batch_idx + 1}/{n_batches}: 特征 {start_idx} 到 {end_idx-1}")
        try:
            Y_batch = Y[:, start_idx:end_idx].copy()
            model, success = train_rf_batch(X, Y_batch, batch_idx, rf_params)
            if success:
                models.append(model)
                batch_info.append((start_idx, end_idx))
                model_path = f'{savedir}/{subject}_{"_".join(roi)}_pipeline_{target}_rf_batch_{batch_idx}.joblib'
                joblib.dump(model, model_path)
                print(f'批次 {batch_idx} 模型已保存到: {model_path}')
            else:
                print(f'批次 {batch_idx} 训练失败，跳过')
            del Y_batch
            if not success:
                del model
            gc.collect()
        except Exception as e:
            print(f"批次 {batch_idx} 处理出错: {e}")
            gc.collect()
            continue

    print(f"\n成功训练了 {len(models)} 个批次模型")

    print("开始预测...")
    all_predictions = []
    batch_accuracies = []
    for batch_idx, (model, (start_idx, end_idx)) in enumerate(zip(models, batch_info)):
        print(f"预测批次 {batch_idx + 1}/{len(models)}")
        try:
            Y_te_batch = Y_te[:, start_idx:end_idx]
            predictions = model.predict(X_te)
            all_predictions.append(predictions)
            if Y_te_batch.shape[1] > 0:
                rs_batch = correlation_score(Y_te_batch.T, predictions.T)
                batch_accuracy = np.mean(rs_batch)
                batch_accuracies.append(batch_accuracy)
                print(f'批次 {batch_idx} 预测准确率: {batch_accuracy:.3f}')
            del predictions, Y_te_batch, rs_batch
            gc.collect()
        except Exception as e:
            print(f"批次 {batch_idx} 预测出错: {e}")
            gc.collect()
            continue

    if all_predictions:
        scores = np.hstack(all_predictions)
        rs = correlation_score(Y_te.T, scores.T)
        overall_accuracy = np.mean(rs)
        print(f'总体预测准确率: {overall_accuracy:.3f}')
        np.save(f'{savedir}/{subject}_{"_".join(roi)}_scores_{target}_rf.npy', scores)
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
            'seed': opt.seed,
            'rf_params': rf_params,
        }
        joblib.dump(model_info, f'{savedir}/{subject}_{"_".join(roi)}_model_info_{target}_rf.joblib')
        print(f'预测结果已保存到: {savedir}')
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
