import numpy as np
import os
import argparse
import numpy as np
import os
import sys
import h5py
if "../src" not in sys.path:
    sys.path.append("../src")  

from wong import sigma


def generate_datasets(n_curves, n_noise, noise_ratio, cv, mean_spacing, out_file, seed=None):
    """
    Generate synthetic Wong datasets and save to a single HDF5 file.
    Each call creates one .h5 file containing:
      - /E_cat, /Sigma_cat, /Sigma_err_cat, /Sigma_true_cat : concatenated arrays
      - /idx_ptr : curve index boundaries
      - /params : true parameters [E0, R0, hbarw] per curve
      - attributes for metadata (noise_ratio, cv, mean_spacing, etc.)
    """
    if seed is not None:
        np.random.seed(seed)

    # Parameter ranges
    E0_range = (50.0, 80.0)
    R0_range = (8.0, 15.0)
    hbarw_range = (2.0, 8.0)

    E_chunks, S_chunks, Se_chunks, Strue_chunks = [], [], [], []
    params_list = []

    for _ in range(n_curves):
        # Sample true parameters
        E0_true = np.random.uniform(*E0_range)
        R0_true = np.random.uniform(*R0_range)
        hbarw_true = np.random.uniform(*hbarw_range)

        E_min = max(0.0, E0_true - 10.0)
        E_max = E0_true + 20.0
        n_points_est = int(np.ceil((E_max - E_min) / mean_spacing)) + 1

        for _ in range(n_noise):
            # Construct random or regular energy grid
            if cv > 0.0:
                spacings = np.random.normal(mean_spacing, cv * mean_spacing, n_points_est)
                spacings = np.clip(spacings, 1e-3, None)
                E_data = E_min + np.cumsum(spacings)
                E_data = E_data[E_data <= E_max]
                if E_data.size == 0 or E_data[-1] < E_max - 1e-12:
                    E_data = np.append(E_data, E_max)
            else:
                E_data = np.arange(E_min, E_max + mean_spacing * 0.5, mean_spacing)

            E_data = np.unique(E_data.astype(float))
            Sigma_true = sigma(E_data, E0_true, R0_true, hbarw_true)
            Sigma_err_std = np.abs(noise_ratio * Sigma_true)
            noise = np.random.normal(0.0, Sigma_err_std)
            Sigma_meas = Sigma_true + noise

            E_chunks.append(E_data)
            S_chunks.append(Sigma_meas)
            Se_chunks.append(Sigma_err_std)
            Strue_chunks.append(Sigma_true)
            params_list.append([E0_true, R0_true, hbarw_true])

    lengths = np.array([len(x) for x in E_chunks], dtype=np.int64)
    idx_ptr = np.zeros(len(lengths) + 1, dtype=np.int64)
    idx_ptr[1:] = np.cumsum(lengths)

    E_cat = np.concatenate(E_chunks)
    Sigma_cat = np.concatenate(S_chunks)
    Sigma_err_cat = np.concatenate(Se_chunks)
    Sigma_true_cat = np.concatenate(Strue_chunks)
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

        # Metadata as attributes
        f.attrs.update(
            noise_ratio=noise_ratio,
            cv=cv,
            mean_spacing=mean_spacing,
            n_curves=n_curves,
            n_noise=n_noise,
        )

    print(f"Saved {len(lengths)} datasets to {out_file} (HDF5 format)")