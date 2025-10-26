import numpy as np
import os
import argparse

def generate_datasets(n_curves, n_noise, noise_ratio, dev_points, n_points, out_file, seed=None):
    """
    Generate synthetic Wong datasets and save all results in a single .npz file.

    Arrays will be shaped as:
      - E:        (N, n_points)        sampled energy grids
      - Sigma:    (N, n_points)        noisy cross sections
      - Sigma_err:(N, n_points)        per-point errors
      - params:   (N, 3)               true parameters [E0, R0, hbarw] for each curve
    where N = n_curves * n_noise.
    """
    if seed is not None:
        np.random.seed(seed)

    # Parameter ranges
    E0_range = (50.0, 80.0)
    R0_range = (8.0, 15.0)
    hbarw_range = (2.0, 8.0)

    N = n_curves * n_noise

    # Containers
    all_E = np.zeros((N, n_points))
    all_Sigma = np.zeros((N, n_points))
    all_Sigma_err = np.zeros((N, n_points))
    all_params = np.zeros((N, 3))

    idx = 0
    for ci in range(n_curves):
        # Sample parameters
        E0_true = np.random.uniform(*E0_range)
        R0_true = np.random.uniform(*R0_range)
        hbarw_true = np.random.uniform(*hbarw_range)

        # Base energy grid
        E_min = max(0.0, E0_true - 10.0)
        E_max = E0_true + 20.0
        if E_max <= E_min:
            E_max = E_min + 20.0
        E_base = np.linspace(E_min, E_max, n_points)

        for ni in range(n_noise):
            # Jitter grid
            if dev_points > 0:
                E_jittered = E_base.copy()
                avg_spacing = (E_max - E_min) / (n_points - 1)
                jitter_scale = dev_points * avg_spacing
                for k in range(1, n_points-1):
                    E_jittered[k] += np.random.normal(0, jitter_scale)
                E_jittered = np.clip(E_jittered, E_min, E_max)
                E_jittered[0], E_jittered[-1] = E_min, E_max
                E_jittered.sort()
                E_data = E_jittered
            else:
                E_data = E_base

            # Wong cross section
            coef = (np.pi * (R0_true**2) * hbarw_true) / (2 * np.maximum(E_data, 1e-6))
            Sigma_true = coef * np.log(1.0 + np.exp((2 * np.pi / max(hbarw_true, 1e-6)) * (E_data - E0_true)))

            # Noise
            Sigma_err = noise_ratio * Sigma_true
            Sigma_err = np.where(Sigma_err == 0, noise_ratio * 1.0, Sigma_err)
            noise = np.random.normal(0.0, Sigma_err)
            Sigma_meas = Sigma_true + noise

            # Store
            all_E[idx] = E_data
            all_Sigma[idx] = Sigma_meas
            all_Sigma_err[idx] = Sigma_err
            all_params[idx] = [E0_true, R0_true, hbarw_true]
            idx += 1

    # Save once
    np.savez(out_file, E=all_E, Sigma=all_Sigma, Sigma_err=all_Sigma_err, params=all_params,
             noise_ratio=noise_ratio, dev_points=dev_points)

    print(f"Saved {N} datasets in one file: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_curves", type=int, required=True)
    parser.add_argument("--n_noise", type=int, required=True)
    parser.add_argument("--noise_ratio", type=float, required=True)
    parser.add_argument("--dev_points", type=float, default=0.0)
    parser.add_argument("--n_points", type=int, required=True)
    parser.add_argument("--out_file", type=str, required=True, help="Path to output .npz file")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    generate_datasets(args.n_curves, args.n_noise, args.noise_ratio,
                      args.dev_points, args.n_points, args.out_file, seed=args.seed)
