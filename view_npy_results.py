import numpy as np

# Change this path to your .npy file
npy_path = "lr_sweep_results_step_1.npy"

data = np.load(npy_path, allow_pickle=True)

print(f"Loaded {npy_path} with {len(data)} experiments.")

for i, result in enumerate(data):
    print(f"Experiment {i+1}:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print()
