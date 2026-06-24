"""This script demonstrates how to write the 4 experimental datasets (each with 1 curve) to a single HDF5 file,
 compatible with the BNN training scripts in the slurm folder"""
import numpy as np
import os
import argparse
import numpy as np
import os
import sys
import h5py

arr = np.array([

# ---------------- Dataset 2 ----------------
[
np.array([
69.97, 70.53, 71.09, 71.64, 72.20, 72.76, 73.31, 73.87, 74.43,
74.99, 75.54, 76.10, 76.66, 77.21, 77.77, 78.33, 78.88, 79.44,
80.00, 80.56, 81.11, 81.67, 82.78, 85.01, 87.24, 89.73, 91.70,
96.15, 100.72, 105.06, 109.52
]),
np.array([
0.24, 0.70, 1.83, 4.28, 8.27, 14.5, 23.4, 35.4, 50.0,
67.0, 87.0, 107.0, 129.0, 152.0, 175.0, 197.0, 223.0, 245.0,
270.0, 295.0, 318.0, 343.0, 385.0, 487.0, 568.0, 662.0,
715.0, 847.0, 949.0, 1065.0, 1133.0
]), 
np.array([
0.01,0.004,0.01,0.02,0.04,0.07,0.1,0.2,0.3,0.3,0.4,0.5,0.7,0.8,0.9,
1.0,1.0,1.0,1.0,2.0,2.0,2.0,2.0,
3.0,3.0,3.0,4.0,4.0,5.0,
6.0,6.0, 
])
],

# ---------------- Dataset 3 ----------------
[
np.array([
49.085, 49.585, 50.085, 50.585, 51.085, 51.585, 52.085, 52.585,
53.085, 53.585, 54.085, 54.585, 55.085, 55.585, 56.085, 56.585,
57.085, 57.585, 59.585, 62.085, 64.585
]),
np.array([
0.02265, 0.05549, 0.23170, 0.57220, 1.2650, 2.5000, 6.8990,
12.240, 27.932, 46.167, 69.020, 86.910, 120.63,
134.09, 165.80, 187.65, 208.61, 241.91,
340.29, 459.86, 530.68
]), 
np.array([0.00955,0.02774,0.04362,0.07130,0.07656,0.15810,0.37780,
0.49911,1.0049,1.0775,2.2478,3.4180,3.6824,3.7022,3.9799,4.1980,4.7005,6.8690,7.7463,9.7483,13.566
])
],

# ---------------- Dataset 5 ----------------
[
np.array([
45.783, 46.220, 46.657, 47.093, 47.967, 48.404, 48.840,
49.714, 50.588, 51.461, 52.335, 53.208, 54.082, 54.955,
56.703, 57.576, 58.450, 59.760, 61.070, 62.381, 63.691, 65.220
]),
np.array([
0.0012545, 0.0054868, 0.022556, 0.046980, 0.22987,
0.42208, 0.89664, 3.0675, 8.9960, 20.669,
39.230, 64.949, 95.977, 127.48, 179.45,
223.23, 261.73, 276.40, 340.66, 362.15,
404.03, 463.14
]), 
np.array([
0.00088705,0.0018289,0.0048090,0.0085774,0.017682,0.10237,0.052744,0.21372,
0.39187,0.91524,1.5011,1.3895,2.2153,2.7628,3.3526,5.7182,6.3311,6.2980,
8.3312,6.5544,6.3037,7.5152
])
],

# ---------------- Dataset 7 ----------------
[
np.array([
46.350, 46.750, 47.150, 47.550, 47.950, 48.350, 48.750,
49.150, 49.550, 49.950, 50.350, 50.750, 51.150, 51.550,
51.950, 52.350, 52.750, 53.150, 53.550, 53.950,
54.350, 54.750, 55.150, 56.350, 57.550, 58.750, 60.750
]),
np.array([
0.00058510, 0.0018760, 0.0047950, 0.012710, 0.032752,
0.10546, 0.32779, 0.56880, 1.2786, 4.4852,
9.3462, 16.118, 26.120, 36.617, 53.351,
77.972, 97.483, 111.78, 136.61, 151.73,
172.68, 194.54, 220.71, 276.25, 354.42,
409.43, 505.49
]), 
np.array([
0.00038769,0.00078712,0.0012497,0.0024932,0.0043635,0.014830,0.039975,0.057021,0.12228,0.19987,0.28782,0.36777,0.55965,0.93437,
0.87945,1.6494,1.1033,2.2397,1.1353,2.8318,1.4871,1.8389,3.8695,3.6713,5.3128,6.3973,5.6506
])
]

], dtype=object)

idx = 0
print(arr[idx][0].shape, arr[idx][1].shape, arr[idx][2].shape)

def write_experiments_h5(arr, out_file="experiments.h5"):
    # Basic metadata (these don't really apply to experimental curves, but keep schema-compatible)
    noise_ratio = 0.0
    cv = 0.0
    mean_spacing = np.nan
    n_curves = int(arr.shape[0])
    n_noise = 1  # each "curve" here is a single dataset, not repeated noisy draws

    E_chunks, S_chunks, Se_chunks, Strue_chunks = [], [], [], []
    params_list = []

    for i in range(n_curves):
        E_data = np.asarray(arr[i][0], dtype=float).ravel()
        Sigma_meas = np.asarray(arr[i][1], dtype=float).ravel()
        Sigma_err_std = np.asarray(arr[i][2], dtype=float).ravel()

        # defensive: match lengths & drop non-finite
        m = min(E_data.size, Sigma_meas.size)
        E_data = E_data[:m]
        Sigma_meas = Sigma_meas[:m]
        good = np.isfinite(E_data) & np.isfinite(Sigma_meas)
        E_data = E_data[good]
        Sigma_meas = Sigma_meas[good]

        # sort by energy
        order = np.argsort(E_data)
        E_data = E_data[order]
        Sigma_meas = Sigma_meas[order]
        Sigma_err_std = Sigma_err_std[order]

        # "true" = measured (since experimental)
        Sigma_true = Sigma_meas.copy()

        # error bars not provided → zeros
        #Sigma_err_std = np.zeros_like(Sigma_meas, dtype=float)

        E_chunks.append(E_data)
        S_chunks.append(Sigma_meas)
        Se_chunks.append(Sigma_err_std)
        Strue_chunks.append(Sigma_true)

        # unknown Wong params for experiment → zeros (keeps file compatible)
        params_list.append([0.0, 0.0, 0.0])

    lengths = np.array([len(x) for x in E_chunks], dtype=np.int64)
    idx_ptr = np.zeros(len(lengths) + 1, dtype=np.int64)
    idx_ptr[1:] = np.cumsum(lengths)

    E_cat = np.concatenate(E_chunks) if lengths.sum() > 0 else np.array([], dtype=float)
    Sigma_cat = np.concatenate(S_chunks) if lengths.sum() > 0 else np.array([], dtype=float)
    Sigma_err_cat = np.concatenate(Se_chunks) if lengths.sum() > 0 else np.array([], dtype=float)
    Sigma_true_cat = np.concatenate(Strue_chunks) if lengths.sum() > 0 else np.array([], dtype=float)
    params_all = np.array(params_list, dtype=float)

    if not out_file.endswith(".h5"):
        out_file = os.path.splitext(out_file)[0] + ".h5"

    with h5py.File(out_file, "w") as f:
        f.create_dataset("E_cat", data=E_cat, compression="gzip")
        f.create_dataset("Sigma_cat", data=Sigma_cat, compression="gzip")
        f.create_dataset("Sigma_err_cat", data=Sigma_err_cat, compression="gzip")
        f.create_dataset("Sigma_true_cat", data=Sigma_true_cat, compression="gzip")
        f.create_dataset("idx_ptr", data=idx_ptr)
        f.create_dataset("params", data=params_all)

        # Metadata as attributes (schema-compatible)
        f.attrs.update(
            noise_ratio=float(noise_ratio),
            cv=float(cv),
            mean_spacing=float(mean_spacing) if np.isfinite(mean_spacing) else -1.0,
            n_curves=int(n_curves),
            n_noise=int(n_noise),
        )

    print(f"Saved {n_curves} curves to {out_file}")
    print(f"Total points: {int(lengths.sum())}")
    print("idx_ptr shape:", idx_ptr.shape, "params shape:", params_all.shape)

if __name__ == "__main__":
    write_experiments_h5(arr, out_file="experiments.h5")