
import os
import numpy as np
import subprocess

# Environment inputs
splits = int(os.getenv("SPLITS", 1))
split_id = int(os.getenv("ID", 0))
device = os.getenv("DEVICE", "0")
test_gpu = os.getenv("TEST_GPU", "1")

# Load full case list
cases_array = np.load("cases_array.npy", allow_pickle=True)
chunks = np.array_split(list(enumerate(cases_array)), splits)
cases_list = chunks[split_id]

print(f"[INFO] Running chunk {split_id}/{splits} with {len(cases_list)} cases on device {device}")

for case_idx, case_name in cases_list:
    print(f"🚀 Launching case {case_name} in isolated subprocess")

    env = os.environ.copy()
    env["DEVICE"] = device
    env["TEST_GPU"] = test_gpu
    env["CASE_NAME"] = case_name
    env["CASE_IDX"] = str(case_idx)

    log_dir = os.path.join("logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{case_name}.log")

    with open(log_path, "w") as log_file:
        result = subprocess.run(
            ["python", "main_co2_isolated_final.py"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env
        )

    if result.returncode == 0:
        print(f"✅ Case {case_name} completed.")
    else:
        print(f"❌ Case {case_name} failed. See {log_path}")
