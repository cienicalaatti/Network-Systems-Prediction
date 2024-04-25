from typing import Tuple

import numpy as np

from scipy.integrate import solve_ivp

from numpy.random import default_rng
rng = default_rng()


def linked_lorenz(NE, z, a, n_timesteps, h_range = 0.06, h = 0.02, c = 0.3, rho = 28.0, beta = 2.667, sigma = 10.0, sigma_dyn = 0.1)  -> Tuple[np.ndarray, np.ndarray]:
    # h_range is the heterogeneity parameter, 0.06 is the value used in the paper
    rng = default_rng()
    hg = h_range*(-1.0 + 2.0*rng.random(NE)) # heterogeneity of the Lorenz oscillators
    
    def linked_lorenz_diff(t, z):
        rng = default_rng()
        nrng = rng.normal(0, sigma_dyn, 3*NE) # mu, sigma, size
        dz_dt = np.zeros(3*NE)
        for k in range(NE):
                # dx[k]/dt:
                for l in range(NE):
                    dz_dt[3*k] += a[k][l]*(z[3*l+1] - z[3*k+1])
                dz_dt[3*k] = -sigma*(z[3*k] -z[3*k+1] + c*dz_dt[3*k]) + nrng[3*k] # c = 0.3 in the paper
                # dy[k]/dt:
                dz_dt[3*k+1] = (rho*(1+hg[k]) - z[3*k+2])*z[3*k] - z[3*k+1] + nrng[3*k+1] # rho = 28 in the paper
                # dz[k]/dt:
                dz_dt[3*k+2] = z[3*k]*z[3*k+1] - beta*z[3*k+2] #+ nrng[3*k+2] # beta = 8/3 in the paper
        return dz_dt
    

    t = np.arange(0, n_timesteps * h, h)
    
    sol = solve_ivp(linked_lorenz_diff,  
                        t_span=(0.0, n_timesteps*h),
                        y0=z,
                        dense_output=True, vectorized=True)
    
    
    return sol.sol(t).T, t



#     def linked_lorenz_diff(t, z):
#         rng = default_rng()
#         nrng = rng.normal(0, sigma_dyn, 3*NE) # mu, sigma, size
#         dz_dt = np.zeros(3*NE)
#         for k in range(NE):
#                 # dx[k]/dt:
#                 for l in range(NE):
#                     dz_dt[3*k] += a[k][l]*(z[3*l+1] - z[3*k+1])
#                 dz_dt[3*k] = c*dz_dt[3*k] + sigma*(z[3*k+1] - z[3*k]) + nrng[3*k] # c = 0.3 in the paper
#                 # dy[k]/dt:
#                 dz_dt[3*k+1] = (rho*(1+hg[k]) - z[3*k+2])*z[3*k] - z[3*k+1] + nrng[3*k+1] # rho = 28 in the paper
#                 dz_dt[3*k+2] = z[3*k]*z[3*k+1] - beta*z[3*k+2] #+ nrng[3*k+2] # beta = 8/3 in the paper
#         return dz_dt