import numpy as np
from scipy.optimize import root_scalar
import findiff as fd
import matplotlib.pyplot as plt

class WongCrossSection:
    """
    Implements Wong's parabolic-barrier formula for two nuclei,
    using analytical derivative to find RB (V'(r) = 0),
    and numerical second derivative to compute ℏω.
    """

    # Physical constants
    hbarc = 197.3269804       # ħc in MeV·fm
    e2 = 1.4399764            # e² / (4πε₀) in MeV·fm
    mass_proton = 938.27208816  # MeV
    mass_neutron = 939.56542052  # MeV
    mN = 2 * (mass_neutron * mass_proton) / (mass_neutron + mass_proton)

    def __init__(self, A1, Z1, A2, Z2, r0=1.44, a=0.65, V0=50.0):
        self.A1, self.Z1 = A1, Z1
        self.A2, self.Z2 = A2, Z2
        self.r0 = r0
        self.a = a
        self.V0 = V0

        # Initial guess for RB
        self.RB_guess = r0 * (A1**(1/3) + A2**(1/3))

        # Reduced mass μ
        self.mu = (WongCrossSection.mN * A1 * A2) / (A1 + A2)



        r_grid = np.linspace(self.RB_guess - 5, self.RB_guess + 5, 1000)
        Vp_grid = np.abs(self.V_prime(r_grid))

        # Find index where |V'(r)| is minimized
        idx_min = np.argmin(Vp_grid)
        self.RB = r_grid[idx_min]
        self.VB = self.V_total(self.RB)


        # ℏω from second derivative
        r_min = max(0.01, self.RB - 5)
        r_max = self.RB + 5
        r_arr = np.linspace(r_min, r_max, 200)

        d2_grid = fd.FinDiff(0, r_arr[1] - r_arr[0], 2, acc=4)
        Vpp_arr = d2_grid(self.V_total(r_arr))

        idx_RB = np.argmin(np.abs(r_arr - self.RB))
        Vpp = Vpp_arr[idx_RB]

        self.hbar_omega = np.sqrt(abs(Vpp) / self.mu) * WongCrossSection.hbarc

        # Save arrays for plotting
        self._r_arr = r_arr
        self._V_arr = self.V_total(r_arr)
        self._Vpp_arr = Vpp_arr
    
    # Define V'(r)
    def V_prime(self, r):
            R_ws = self.r0 * (self.A1**(1/3) + self.A2**(1/3))
            exp_term = np.exp((r - R_ws) / self.a)
            denom = (1 + exp_term)**2
            Vn_prime = self.V0 / self.a * exp_term / denom
            Vc_prime = - (self.Z1 * self.Z2 * WongCrossSection.e2) / r**2
            return Vn_prime + Vc_prime
    
    def V_coulomb(self, r):
        return (self.Z1 * self.Z2 * WongCrossSection.e2) / r

    def V_nuclear(self, r):
        R_ws = self.r0 * (self.A1**(1/3) + self.A2**(1/3))
        return -self.V0 / (1 + np.exp((r - R_ws) / self.a))

    def V_total(self, r):
        return self.V_coulomb(r) + self.V_nuclear(r)

    def sigma(self, E, mb=True):
        """
        Compute Wong cross section σ(E) for given energy E.

        Parameters
        ----------
        E : float or array_like
            Center-of-mass energy in MeV.

        Returns
        -------
        sigma : float or ndarray
            Fusion (reaction) cross section in fm².
        """
        E = np.array(E, dtype=float)
        self.x = 2 * np.pi * (E - self.VB) / self.hbar_omega
        sig = (self.hbar_omega / (2 * E)) * self.RB**2 * np.log(1 + np.exp(self.x))
        if mb:
            return sig * 10.0
        else:
            return sig
        

    def plot_potential(self, show=True):
        """
        Plot V(r) and V''(r) near RB (for debugging).
        """
        plt.figure(figsize=(8, 6))
        plt.plot(self._r_arr, self._V_arr, label='V(r)', color='blue')
        plt.plot(self._r_arr, self._Vpp_arr, label="V''(r)", color='red')
        plt.axvline(self.RB, color='k', linestyle='--', label=f'RB = {self.RB:.2f} fm')
        plt.axhline(0, color='gray', linestyle=':')
        plt.xlabel('r [fm]')
        plt.ylabel('Potential [MeV] or Curvature [MeV/fm²]')
        plt.title('Potential and Second Derivative near RB')
        plt.legend()
        plt.grid(True)
        if show:
            plt.show()


def sigma(E, E0, R0, hbarw):
    return 10*(0.5*R0 **2 * hbarw/ E) * np.log(1 + np.exp(2*np.pi*(E - E0) / hbarw))