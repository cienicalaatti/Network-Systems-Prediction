import numpy as np
import os

import sys
import builtins

# from reservoirpy import ESN
from typing import Tuple
from scipy.integrate import solve_ivp
import networkx as nx

def kuramoto(N:int, K:float, omega:np.ndarray, a:np.ndarray, init:np.ndarray, n_timesteps:int, h:float = 0.02):
    
    def diff(t, theta, a, omega, K):
        theta_i, theta_j = np.meshgrid(theta, theta)
        coupling_terms = a * np.sin(theta_j - theta_i)

        d_theta = omega + K * coupling_terms.sum(axis=0)
        return d_theta

    t = np.arange(0,n_timesteps*h, h)
    sol = solve_ivp(fun=diff, t_span=(0.0, n_timesteps*h), y0=init, args=(a, omega, K), dense_output=True, vectorized=False, method="LSODA", atol=1e-7, rtol=1e-5)

    return np.mod(sol.sol(t).T, 2 * np.pi), t

def kuramoto_euler(Nstep, dt, A, omega, K, init):
    theta = np.zeros((30, Nstep))
    theta[:,0] = init
    for i in range(Nstep-1):
        dtheta = omega + K * np.sum(A.T * np.sin(theta[:, i] - theta[:, i, np.newaxis]), axis=1)
        theta[:, i + 1] = theta[:, i] + dt * dtheta
    return np.mod(theta.T, 2*np.pi)

def linked_lorenz(NE, z, a, n_timesteps, h = 0.02, rho = 28.0, beta = 2.667, sigma = 10.0, c = 0.09, method='RK45')  -> Tuple[np.ndarray, np.ndarray]:
    
    def linked_lorenz_diff(t, z):
        dz_dt = np.zeros(3*NE)
        for k in range(NE):
                # dx[k]/dt:
                for l in range(NE):
                    dz_dt[3*k] += a[k][l]*(z[3*l+1] - z[3*k+1])
                dz_dt[3*k] = -sigma*(z[3*k] -z[3*k+1] + c * dz_dt[3*k])    
                # dy[k]/dt:
                dz_dt[3*k+1] = rho*z[3*k] - z[3*k+1] - z[3*k+2]*z[3*k]
                # dz[k]/dt:
                dz_dt[3*k+2] = z[3*k]*z[3*k+1] - beta*z[3*k+2]
        return dz_dt
    

    t = np.arange(0, n_timesteps * h, h)
    
    if method == 'Euler':
        temp = None
        t = np.zeros((n_timesteps, 3*NE))
        t[0,:] = z
        for i in range(n_timesteps-1):
            t[i+1] = t[i] + h * linked_lorenz_diff(temp, t[i])
        return t
    sol = solve_ivp(linked_lorenz_diff,  
                        t_span=(0.0, n_timesteps*h),
                        y0=z,
                        dense_output=True, vectorized=False,
                        method = method)
    
    
    return sol.sol(t).T

def generate_a(N = 30, D = 3, DELTA = 0.8, G = 5):
    #np.random.seed(543)
    omega = np.random.uniform(-np.pi/2, np.pi/2, size=N)

    g = nx.empty_graph()
    g.add_nodes_from(range(N)) #network with N nodes and no edges
    try:
        while any(g.degree(node) < D for node in g.nodes()):
            i = np.random.choice([node for node in g.nodes() if g.degree(node) < D])
            j = np.random.choice([node for node in g.nodes() if g.degree(node) < D and not g.has_edge(i, node) and node != i])

            p = (DELTA**G)/(DELTA**G + np.abs(omega[i]-omega[j])**G)

            if np.random.uniform(0,1) < p:
                g.add_edge(i,j)
        a = nx.to_numpy_array(g)
        return a, omega, g
    except:
        return generate_a()

def find_assortative_net(target, n_trials = 200):
    """
    find frequency-assortative network with given assortativity.
    target: target assortativity, between 0 and 1
    """
    closest_a = np.inf
    a = None
    closest_omega = None
    
    for i in range(n_trials):
        a, omega, g = generate_a()
        for node, value in zip(g.nodes(), omega):
            g.nodes[node]['omega'] = value
        assort = nx.numeric_assortativity_coefficient(g,'omega')
        if np.abs(target-assort) < np.abs(target-closest_a):
            closest_a = assort
            closest_omega = omega
    print(closest_a) #check how close we got to the target
    return a, closest_omega

def qr_pos(Q, R):
    sgn = np.sign(np.diag(R))
    R_new = np.diag(sgn).dot(R)
    Q_new = Q.dot(np.diag(sgn))
    return Q_new, R_new

def lyaspec(a, omega, init, dt):
    tau = int(500 / dt)        # number of initial timesteps
    N = 500           # total number of renormalizations
    T = int(6 / dt)            # number of timesteps between renormalizations
    m = 30        # number of exponents to compute
    eps = 0.000001      # perturbation magnitude

    dim = 30
    log_r = 0

    timestep = dt
    N_network = 30
    K = 0.5
    u0, _ = kuramoto(N=N_network, K=K, omega=omega, a=a, init=init, n_timesteps=tau, h=timestep)

    u_prev = u0[-1, :]
    # choose initial orthogonal directions
    Q0 = np.diag(np.ones(dim))
    Q_prev = Q0

    # for total N renormalizations
    for j in range(N):
        # compute uj = u(tj)
        uj, _ = kuramoto(N=N_network, K=K, omega=omega, a=a, init=u_prev, n_timesteps=T, h=timestep)
        uj = uj[-1]
        
        # for m lyapunov exponents/directions
        Phi_Q = np.zeros((dim, dim))
        for i in range(m):
            w, _ = kuramoto(N=N_network, K=K, omega=omega, a=a, init=u_prev + eps*Q_prev[:, i], n_timesteps=T, h=timestep)
            w = w[-1]
            Phi_Q[:, i] = (w-uj)/eps

        Q, R = np.linalg.qr(Phi_Q)  ### Q, R decomposition. Q is a matrix with orthonormal columns.
        Q, R = qr_pos(Q, R)
        log_r = log_r + np.log(np.diag(R))
        
        u_prev = uj
        Q_prev = Q
        

    # calculate the exponents
    return log_r/(N*T*timestep)

def lyaspec_euler(a, omega, init, dt):
    tau = int(500 / dt)        # number of initial timesteps
    N = 500           # total number of renormalizations
    T = int(6 / dt)            # number of timesteps between renormalizations
    m = 30          # number of exponents to compute
    eps = 0.000001      # perturbation magnitude

    dim = 30
    log_r = 0

    timestep = dt
    N_network = 30
    K = 0.5

    u0 = kuramoto_euler(Nstep=tau, dt=timestep, A=a, omega=omega, K=K, init=init)

    u_prev = u0[-1, :]
    # choose initial orthogonal directions
    Q0 = np.diag(np.ones(dim))
    Q_prev = Q0

    # for total N renormalizations
    for j in range(N):
        # compute uj = u(tj)
        uj = kuramoto_euler(K=K, omega=omega, A=a, init=u_prev, Nstep=T, dt=timestep)
        uj = uj[-1]
        
        # for m lyapunov exponents/directions
        Phi_Q = np.zeros((dim, dim))
        for i in range(m):
            w = kuramoto_euler(K=K, omega=omega, A=a, init=u_prev + eps*Q_prev[:, i], Nstep=T, dt=timestep)
            w = w[-1]
            Phi_Q[:, i] = (w-uj)/eps

        Q, R = np.linalg.qr(Phi_Q)  ### Q, R decomposition. Q is a matrix with orthonormal columns.
        Q, R = qr_pos(Q, R)
        log_r = log_r + np.log(np.diag(R))
        
        u_prev = uj
        Q_prev = Q
   
    return log_r/(N*T*timestep)

def lyaspec_lorenz(a, init, dt, method, c):
    tau = int(100 / dt)        # number of initial timesteps
    N = 1000           # total number of renormalizations
    T = int(1 / dt)            # number of timesteps between renormalizations
    m = 30           # number of exponents to compute
    eps = 0.000001      # perturbation magnitude

    NE = 10
    dim = 3*NE
    log_r = 0

    timestep = dt

    u0 = linked_lorenz(NE, z = init, a = a, n_timesteps = tau, h = timestep,
                            rho = 28.0, beta = 2.667, sigma = 10.0, method=method, c=c)

    u_prev = u0[-1, :]
    # choose initial orthogonal directions
    Q0 = np.diag(np.ones(dim))
    Q_prev = Q0

    # for total N renormalizations
    for j in range(N):
        # compute uj = u(tj)
        uj = linked_lorenz(NE, z = u_prev, a = a, n_timesteps = T, h = timestep,
                                rho = 28.0, beta = 2.667, sigma = 10.0, method=method, c=c)
        uj = uj[-1, :]
        
        # for m lyapunov exponents/directions
        Phi_Q = np.zeros((dim, dim))
        for i in range(m):
            w = linked_lorenz(NE, z = u_prev + eps*Q_prev[:, i], a = a, n_timesteps = T, h = timestep,
                                rho = 28.0, beta = 2.667, sigma = 10.0, method=method, c=c)
            w = w[-1, :]
            Phi_Q[:, i] = (w-uj)/eps

        Q, R = np.linalg.qr(Phi_Q)  ### Q, R decomposition. Q is a matrix with orthonormal columns.
        Q, R = qr_pos(Q, R)
        log_r = log_r + np.log(np.diag(R))
        
        u_prev = uj
        Q_prev = Q
   
    return log_r/(N*T*timestep)


def estimate_lyap_kuramoto(job):

    lyap_own = []
    lyap_euler = []

    for run in range(5):
        timesteps = [0.01, 0.02, 0.03, 0.04]
        init = np.random.uniform(0, 2*np.pi, size=30)
        a, omega, _ = generate_a(N = 30, D = 3, DELTA = 0.08, G = 5)
        
        for dt in timesteps:
            own = lyaspec(a, omega, init, dt)
            keshav = lyaspec_euler(a, omega, init, dt)
            lyap_own.append(own)
            lyap_euler.append(keshav)

        data = [{"lsoda":lyap_own, "euler":lyap_euler}]
        np.save(f"./outputs/kuramoto/N30/lyespec_dt_test/{job}_{run}.npy", data)

def estimate_lyap_lorenz(job_number):

    rk45 = []
    euler = []

    data = [{"Euler":euler}]
    #data = [{"RK45":rk45}]

    timesteps = [0.005, 0.0075, 0.01]
    #dt = 0.01
    #c_list =[0, 0.1, 0.2, 0.3, 0.4, 0.5]
    c = 0.09
    NE = 10
    init = 2*(np.random.random(3*NE)-0.5)
    np.random.seed(14)
    a, _, _ = generate_a(N=10, D=3)
    np.random.seed()
    for dt in timesteps:
        for method in data[0].keys():
            data[0][method].append(lyaspec_lorenz(a, init, dt, method, c))

    np.save(f"./outputs/linked_lorenz/N10/lyespec_dt_test/{method}_{20+job_number}.npy", data)

def lyap_vs_assortativity(job):
    assorts = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    dt = 0.02
    #np.random.seed(777)
    for run in range(5):
        data = [{"euler":[]}]
        for assort in assorts:
            init = np.random.uniform(0, 2*np.pi, size=30)
            a, omega = find_assortative_net(target=assort)
            data[0]["euler"].append(lyaspec_euler(a, omega, init, dt))
            print(assort)
        np.save(f"./outputs/kuramoto/N30/lyespec_dt_test/assortativity/{job}_{run}.npy", data)


def main():
    job_number = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
    estimate_lyap_lorenz(job_number)

if __name__ == "__main__":
    main()