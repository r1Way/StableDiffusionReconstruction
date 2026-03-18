# python run_ridge2_batch.py
import subprocess

subject = "subj01"
target = "init_latent"
# roi_list = ["midlateral", "midparietal", "parietal", "lateral"]
roi_list = ["lateral"]
for roi in roi_list:
    cmd = [
        "python", "ridge2.py",
        "--target", target,
        "--roi", roi,
        "--subject", subject
    ]
    print(f"运行: {' '.join(cmd)}")
    subprocess.run(cmd)
