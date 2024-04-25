import numpy as np
from scipy.integrate import solve_ivp

def kuramoto(K:float, omega:np.ndarray, a:np.ndarray, init:np.ndarray, n_timesteps:int, h:float = 0.02, method:str = 'RK45'):
    '''
    Simulate time series data for a network of Kuramoto oscillators.
    
    Parameters:

    K -- coupling strength

    omega -- (N,) numpy array of natural frequencies of the oscillators

    a -- adjacency matrix of the oscillator network

    init -- (N,) numpy array containing the initial states of the oscillators

    n_timesteps -- number of simulation steps

    h -- integration stepsize

    Returns:

    a tuple of numpy arrays containing the system time series and the time steps.
    '''
    def diff(t, theta, a, omega, K):
        theta_i, theta_j = np.meshgrid(theta, theta)
        coupling_terms = a * np.sin(theta_j - theta_i)

        d_theta = omega + K * coupling_terms.sum(axis=0)
        return d_theta

    t = np.arange(0,n_timesteps*h, h)
    sol = solve_ivp(fun=diff, t_span=(0.0, n_timesteps*h), y0=init, args=(a, omega, K), dense_output=True, vectorized=False, method=method, rtol=1e-5, atol=1e-7)

    return sol.sol(t).T, t