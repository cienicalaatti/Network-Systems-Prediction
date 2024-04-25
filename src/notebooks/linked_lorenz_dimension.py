import click
import pandas as pd
#from skopt import gp_minimize
#from scipy import linalg
#import networkx as nx
import numpy as np
import sys
sys.path.insert(0, '..')
import numpy as np
from typing import Tuple
from scipy.integrate import solve_ivp
from numpy.random import default_rng
rng = default_rng()


def write_output(NE: int, L: float, outfile: str, c,
                 tau, N, T, eps, timestep, kaplan_yorke: float, lyespec, seeds):
    col = ['NE', 'L', 'seed']
    df = pd.DataFrame(columns=col)

    df['seed'] = seeds
    df['NE'] = NE
    df['L'] = L
    df['c'] = c

    df['kaplan_yorke'] = kaplan_yorke

    df['tau'] = tau
    df['N'] = N
    df['T'] = T
    df['eps'] = eps
    df['timestep'] = timestep
    df = df.join(pd.DataFrame(lyespec))

    df.to_csv(outfile+'.csv', index=False)

    return

def createA(L, NE):
    # create a connection matrix for the network of linked lorenz systems
    # L - number of links
    # NE - number of elements (lorenz systems)
    ind_list = []
    for i in range(NE):
        for j in range(NE):
            if i!=j:
                ind_list.append((i, j))
    ind = np.random.choice(range(NE*NE-NE), L, replace=False)

    A = np.zeros((NE, NE))
    for index in ind:
        A[ind_list[index][0], ind_list[index][1]] = 1
    return A


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
                dz_dt[3*k+2] = z[3*k]*z[3*k+1] - beta*z[3*k+2] + nrng[3*k+2] # beta = 8/3 in the paper
        return dz_dt


    t = np.arange(0, (n_timesteps+1) * h, h)

    sol = solve_ivp(linked_lorenz_diff,
                        t_span=(0.0, (n_timesteps+1)*h),
                        y0=z,
                        dense_output=True)


    return sol.sol(t).T[-1, :]

# calculate lyapunov specturm for lorenz system
def calculate_lyespec(NE, L, C, tau, N, T, eps, timestep):
    # calulate lyapunov specturm for network of linked lorenz systems
    # NE - number of elements (lorenz systems)
    # L - number of links between nodes
    # C - connection strength
    # tau = 10000        # number of initial timesteps
    # N = 1000           # total number of renormalizations
    # T = 200            # number of timesteps between renormalizations
    # eps = 0.000001     # perturbation magnitude
    # timestep = 0.01

    # returns lyapunov exponents in a vector

    # help function to make sure R part of QR decomposition is positive
    def qr_pos(Q, R):
        sgn = np.sign(np.diag(R))
        R_new = np.diag(sgn).dot(R)
        Q_new = Q.dot(np.diag(sgn))
        return Q_new, R_new

    m = 3*NE             # number of exponents to compute
    dim = 3*NE
    log_r = 0

    #state0 = np.random.random(3*NE)
    state0 = np.random.uniform(-1,1,NE*3)
    state0[0::3] = 20*state0[0::3]
    state0[1::3] = 20*state0[1::3]
    state0[2::3] = 25+15*state0[2::3]
    #a = np.random.random((3,3))
    #a = np.random.choice([0,1], (NE,NE), p=[1-p, p])
    #a = np.zeros((3,3))
    a = createA(L, NE)

    # compute u(0)
    u0 = linked_lorenz(NE, state0, a, tau, h = timestep, h_range=0.0, c=C, sigma_dyn=0)
    u_prev = u0

    # choose initial orthogonal directions
    Q0 = np.diag(np.ones(dim))
    #Q0 = np.random.random((3,3))
    Q_prev = Q0

    # for total N renormalizations
    for j in range(N):
        #if j%100==0:
         #   print(j)

        # compute uj = u(tj)
        uj = linked_lorenz(NE, u_prev, a, T, h = timestep,  h_range=0.0, c=C, sigma_dyn=0)

        # for m lyapunov exponents/directions
        Phi_Q = np.zeros((dim, dim))
        for i in range(m):
            w = linked_lorenz(NE, u_prev + eps*Q_prev[:, i], a, T, h = timestep,  h_range=0.0, c=C, sigma_dyn=0)
            Phi_Q[:, i] = (w-uj)/eps

        Q, R = np.linalg.qr(Phi_Q)  ### Q, R decomposition. Q is a matrix with orthonormal columns.
        Q, R = qr_pos(Q, R)
        log_r = log_r + np.log(np.diag(R))

        u_prev = uj
        Q_prev = Q

    #  return the calculated exponents
    return log_r/(N*T*timestep)

def kaplan_yorke(lyevec):
    lysum = 0
    i=0
    while i < len(lyevec):
        lysum = lysum + lyevec[i]
        i=i+1
        if lysum < 0:
            return i-1 + (lysum - lyevec[i-1])/np.abs(lyevec[i-1])

    return float("NaN")


@click.command()
@click.option("--ne", required=True, type=int, help="Number of elements (lorenz systems)")
@click.option("--l", required=True, type=int, help="Number of links between nodes")
@click.option("--n_iter", required=True, type=int, help="number of random realizations of the network of lorenz systems")
@click.option("--c", required=False, default=0.3, help="connection strength of linked systems")
@click.option("--tau", required=False, default=10000, help="number of initial timesteps")
@click.option("--n", required=False, default=1000, help="total number of renormalizations")
@click.option("--t", required=False, default=200, help="number of timesteps between renormalizations")
@click.option("--eps", required=False, default=0.000001, help="perturbation magnitude")
@click.option("--timestep",  required=False, default=0.01)
@click.option("--outfile", required=True, type=str, help="File where output is written.")

def main(ne, l, n_iter, c, tau, n, t, eps, timestep, outfile):
    sq = np.random.SeedSequence()
    seeds = sq.generate_state(n_iter)
    lyespec = np.zeros((n_iter, 3*ne))
    ky = np.zeros(n_iter)

    for iteration in range(n_iter):
        seed = seeds[iteration]
        np.random.seed(seed)

        ## calculate lyapunov exponents
        lyespec_ = calculate_lyespec(ne, l, c, tau, n, t, eps, timestep)
        ## calculate kaplan_yorke dimension
        ky_ = kaplan_yorke(lyespec_)

        lyespec[iteration, :] = lyespec_
        ky[iteration] = ky_


    ## save the results as df:
    write_output(ne, l, outfile, c, tau, n, t, eps, timestep, ky, lyespec, seeds)

if __name__ == "__main__":
    main()
