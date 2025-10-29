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
    Generate synthetic Wong datasets and save all results in a single .npz file.

    Each dataset corresponds to a simulated fusion excitation function sampled
    on an energy grid with a given mean spacing and optional irregularity. 
    Random noise is added to emulate experimental uncertainties.

    All generated datasets for a given experiment are saved together in a single
    `.npz` file containing arrays for energies, cross sections, uncertainties,
    and true parameters.

    Parameters
    ----------
    n_curves : int
        Number of distinct underlying Wong parameter sets to generate.
        Each curve corresponds to a different combination of (E0, R0, ħω).

    n_noise : int
        Number of noisy realizations (replicates) to generate for each curve.
        Each replicate shares the same true parameters but has independent
        noise and jitter in the energy grid.

    noise_ratio : float
        Relative standard deviation of the Gaussian noise applied to the 
        cross section at each energy point. For example, 
        `noise_ratio = 0.01` corresponds to 1% measurement noise.

    cv : float
        Coefficient of variation (fractional standard deviation) for the
        irregularity of the energy grid spacing. A value of:
          - `cv = 0.0` → perfectly uniform grid
          - `cv = 0.25` → energy step sizes fluctuate by 25% of mean spacing
        The spacing between consecutive energy points is sampled from a normal
        distribution: `ΔE ~ N(mean_spacing, (cv * mean_spacing)^2)`.

    mean_spacing : float
        Average energy step size in MeV between consecutive points on the grid.
        Together with the sampled (E0, R0, ħω) values, this determines how many
        energy points are used to cover the interval `[E_min, E_max]`, where:
            E_min = max(0, E0_true - 10)
            E_max = E0_true + 20
        Smaller values correspond to higher experimental resolution.

    out_file : str
        Full path to the output `.npz` file. The file will contain the arrays:
            - `E`          : list of energy grids (object array)
            - `Sigma`      : list of noisy cross sections
            - `Sigma_err`  : list of per-point noise levels
            - `params`     : array of true parameters [E0, R0, ħω]
            - `noise_ratio`, `cv`, `mean_spacing`

    seed : int, optional
        Random seed for reproducibility. If None, the global NumPy RNG state is used.


    Saves:
      - E:  energy grids
      - Sigma:   noisy cross sections
      - Sigma_err_std:   per-point error standard deviation
      - params:        true parameters [E0, R0, ħω]
    """

    if seed is not None:
        np.random.seed(seed)

    # Parameter ranges
    E0_range = (50.0, 80.0)
    R0_range = (8.0, 15.0)
    hbarw_range = (2.0, 8.0)

    datasets = []
    params_list = []

    for ci in range(n_curves):
        # Sample true parameters
        E0_true = np.random.uniform(*E0_range)
        R0_true = np.random.uniform(*R0_range)
        hbarw_true = np.random.uniform(*hbarw_range)

        E_min = E0_true - 10.0
        E_max = E0_true + 20.0

        # Determine how many points are needed to cover the range
        n_points_est = int(np.ceil((E_max - E_min) / mean_spacing)) + 1

        for ni in range(n_noise):
            # === Construct irregular grid ===
            if cv > 0.0:
                # Random spacings from normal(mean, cv*mean)
                spacings = np.random.normal(mean_spacing, cv * mean_spacing, n_points_est)
                spacings = np.clip(spacings, 1e-3, None)
                E_data = E_min + np.cumsum(spacings)
                E_data = E_data[E_data <= E_max]
                # Ensure first/last coverage
                if E_data[-1] < E_max:
                    E_data = np.append(E_data, E_max)
            else:
                E_data = np.arange(E_min, E_max + mean_spacing / 2, mean_spacing)

            # Guarantee unique sorted grid
            E_data = np.unique(E_data)

            # === Wong formula ===
            Sigma_true = sigma(E_data, E0_true, R0_true, hbarw_true)

            # === Add noise ===
            Sigma_err_std = np.abs(noise_ratio * Sigma_true)
            noise = np.random.normal(0.0, Sigma_err_std)
            Sigma_meas = Sigma_true + noise

            datasets.append((E_data, Sigma_meas, Sigma_err_std, Sigma_true))
            params_list.append([E0_true, R0_true, hbarw_true])

    # Since grids can vary in length, we store as object arrays
    
    E_all = np.array([d[0] for d in datasets], dtype=object)
    Sigma_all = np.array([d[1] for d in datasets], dtype=object)
    Sigma_err_all = np.array([d[2] for d in datasets], dtype=object)
    Sigma_true_all = np.array([d[3] for d in datasets], dtype=object)
    params_all = np.array(params_list)

    np.savez(out_file,
             E=E_all,
             Sigma=Sigma_all,
             Sigma_err=Sigma_err_all,
             Sigma_true=Sigma_true_all,
             params=params_all,
             noise_ratio=noise_ratio,
             cv=cv,
             mean_spacing=mean_spacing)

    print(f"Saved {len(datasets)} datasets to {out_file}")
