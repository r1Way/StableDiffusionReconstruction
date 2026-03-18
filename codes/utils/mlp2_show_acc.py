# python mlp2_show_acc.py --subject subj01 --roi ventral --target c --method mlp
# python mlp2_show_acc.py --subject subj01 --roi early --target init_latent --method mlp
# 这个脚本 mlp2_show_acc.py 是用来查看模型训练和预测结果的准确率信息文件，
# 也就是你用 mlp2.py、ridge2.py 或 rf2.py 训练和预测后保存的 model_info 文件。
# ../../decoded/{subject}/{subject}_{roi}_model_info_{target}{suffix}.joblib
# 这个 .joblib 文件里保存了：
# 1.总体预测准确率（prediction_accuracy）
# 2.每个 batch 的预测准确率（batch_accuracies）
# 3.其它模型参数和训练信息
import argparse
import os
import joblib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True, help="subject name")
    parser.add_argument("--roi", required=True, type=str, nargs="+", help="ROI name(s)")
    parser.add_argument("--target", type=str, required=True, help="target name")
    parser.add_argument("--method", type=str, default="mlp", choices=["mlp", "ridge", "rf"], help="方法类型: mlp, ridge, rf")
    parser.add_argument("--decoded_dir", type=str, default="../../decoded", help="decoded dir")
    opt = parser.parse_args()

    subject = opt.subject
    roi = opt.roi
    target = opt.target
    method = opt.method
    decoded_dir = opt.decoded_dir

    # 文件名后缀
    if method == "mlp":
        suffix = "_mlp"
    elif method == "rf":
        suffix = "_rf"
    else:
        suffix = ""  # ridge 默认无后缀

    info_path = os.path.join(decoded_dir, subject, f"{subject}_{'_'.join(roi)}_model_info_{target}{suffix}.joblib")
    if not os.path.exists(info_path):
        print(f"文件不存在: {info_path}")
        return

    info = joblib.load(info_path)
    batch_acc = info.get("batch_accuracies", None)
    overall_acc = info.get("prediction_accuracy", None)

    print(f"文件: {info_path}")
    print(f"总体预测准确率: {overall_acc:.4f}")
    print("每个batch预测准确率:")
    if batch_acc is not None:
        for i, acc in enumerate(batch_acc):
            print(f"  Batch {i+1}: {acc:.4f}")
    else:
        print("无分批准确率信息（可能是单模型/单批次）")

if __name__ == "__main__":
    main()
