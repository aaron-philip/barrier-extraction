import numpy as np
import os
import argparse
import numpy as np
import os
import sys
if "../src" not in sys.path:
    sys.path.append("../src")  

from wong import sigma

def generate_datasets(n_curves, n_noise, noise_ratio, cv, mean_spacing, out_file, seed=None):
    """
    Generate synthetic Wong datasets and save in a packed format:
      - E_cat, Sigma_cat, Sigma_err_cat, Sigma_true_cat: concatenated 1-D float arrays
      - idx_ptr: int array of length (N+1); dataset i is [idx_ptr[i]:idx_ptr[i+1]]
      - params: shape (N, 3) with [E0, R0, hbarw] per dataset
      - metadata: noise_ratio, cv, mean_spacing, n_curves, n_noise
    """
    if seed is not None:
        np.random.seed(seed)

    # Parameter ranges
    E0_range   = (50.0, 80.0)
    R0_range   = (8.0, 15.0)
    hbarw_range = (2.0, 8.0)

    E_chunks, S_chunks, Se_chunks, Strue_chunks = [], [], [], []
    params_list = []

    for _ in range(n_curves):
        # Sample true parameters
        E0_true   = np.random.uniform(*E0_range)
        R0_true   = np.random.uniform(*R0_range)
        hbarw_true = np.random.uniform(*hbarw_range)

        E_min = max(0.0, E0_true - 10.0)
        E_max = E0_true + 20.0

        # Estimated point count for regular grid as a baseline
        n_points_est = int(np.ceil((E_max - E_min) / mean_spacing)) + 1

        for _ in range(n_noise):
            # === Construct grid ===
            if cv > 0.0:
                # Random spacings ~ N(mean_spacing, (cv*mean_spacing)^2)
                spacings = np.random.normal(mean_spacing, cv * mean_spacing, n_points_est)
                spacings = np.clip(spacings, 1e-3, None)  # avoid nonpositive
                E_data = E_min + np.cumsum(spacings)
                E_data = E_data[E_data <= E_max]
                # Ensure last point reaches E_max
                if E_data.size == 0 or E_data[-1] < E_max - 1e-12:
                    E_data = np.append(E_data, E_max)
            else:
                E_data = np.arange(E_min, E_max + mean_spacing * 0.5, mean_spacing)

            # Unique + sorted
            E_data = np.unique(E_data.astype(float))

            # === Wong formula ===
            Sigma_true = sigma(E_data, E0_true, R0_true, hbarw_true)

            # === Add noise ===
            Sigma_err_std = np.abs(noise_ratio * Sigma_true)
            noise         = np.random.normal(0.0, Sigma_err_std)
            Sigma_meas    = Sigma_true + noise

            # Append chunks
            E_chunks.append(E_data.astype(float))
            S_chunks.append(Sigma_meas.astype(float))
            Se_chunks.append(Sigma_err_std.astype(float))
            Strue_chunks.append(Sigma_true.astype(float))
            params_list.append([E0_true, R0_true, hbarw_true])

    # Build idx_ptr and concatenate (this avoids dtype=object)
    lengths = np.array([len(x) for x in E_chunks], dtype=np.int64)
    idx_ptr = np.zeros(len(lengths) + 1, dtype=np.int64)
    idx_ptr[1:] = np.cumsum(lengths)

    E_cat       = np.concatenate(E_chunks).astype(float, copy=False)
    Sigma_cat   = np.concatenate(S_chunks).astype(float, copy=False)
    Sigmae_cat  = np.concatenate(Se_chunks).astype(float, copy=False)
    Sigmat_cat  = np.concatenate(Strue_chunks).astype(float, copy=False)
    params_all  = np.array(params_list, dtype=float)  # shape (N, 3)

    np.savez_compressed(
        out_file,
        E_cat=E_cat,
        Sigma_cat=Sigma_cat,
        Sigma_err_cat=Sigmae_cat,
        Sigma_true_cat=Sigmat_cat,
        idx_ptr=idx_ptr,
        params=params_all,
        noise_ratio=float(noise_ratio),
        cv=float(cv),
        mean_spacing=float(mean_spacing),
        n_curves=int(n_curves),
        n_noise=int(n_noise),
    )

    print(f"Saved {len(lengths)} datasets to {out_file} (packed format)")