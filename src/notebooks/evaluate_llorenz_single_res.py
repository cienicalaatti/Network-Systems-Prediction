import numpy as np
import reservoirpy
import pandas as pd
import pickle
import networkx as nx
from scipy.integrate import solve_ivp
import os

def generate_a():
    N = 10
    D = 3
    S = 0.8
    G = 5
    K = 0.5
    try:
        omega = np.random.uniform(-np.pi/2, np.pi/2, size=N)

        g = nx.empty_graph()
        g.add_nodes_from(range(N)) #network with N nodes and no edges

        while any(g.degree(node) < D for node in g.nodes()):
            i = np.random.choice([node for node in g.nodes() if g.degree(node) < D])
            j = np.random.choice([node for node in g.nodes() if g.degree(node) < D and not g.has_edge(i, node) and node != i])

            p = (S**G)/(S**G + np.abs(omega[i]-omega[j])**G)

            if np.random.uniform(0,1) < p:
                g.add_edge(i,j)

        a = nx.to_numpy_array(g)
        return a, omega, g
    except:
        return generate_a()

def linked_lorenz(NE, z, a, n_timesteps, h = 0.02, rho = 28.0, beta = 2.667, sigma = 10.0, c = 0.09, method='RK45'):
    
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
        t = np.zeros((n_timesteps, 3*NE))
        t[0,:] = z
        for i in range(n_timesteps-1):
            t[i+1] = t[i] + h * linked_lorenz_diff(_, t[i])
        return t
    sol = solve_ivp(linked_lorenz_diff,  
                        t_span=(0.0, n_timesteps*h),
                        y0=z,
                        dense_output=True, vectorized=False,
                        method = method)
    
    
    return sol.sol(t).T

def generate_data(a, dt):
    methods = ["RK45"]
    timestep = dt
    N=10
    steps = int(600/timestep)
    trsteps = int((steps*5)/8)
    valsteps = int((steps*3)/16)
    dimension = 3
    z = 2*(np.random.random(dimension*N)-0.5) #initial state
    for method in methods:
        data = linked_lorenz(N, z = z, a = a, n_timesteps = 80000, h = timestep,
                                rho = 28.0, beta = 2.667, sigma = 10.0, method=method)
        
        train = [{"data":data[10000:trsteps], "dt":timestep, "adjacency_matrix":a, "N":N, "initLen":0}]
        validation = [{"data":data[trsteps:trsteps+valsteps], "dt":timestep, "adjacency_matrix":a, "N":N, "initLen":0}]
        test = [{"data":data[trsteps+valsteps:], "dt":timestep, "adjacency_matrix":a, "N":N, "initLen":0}]
        
        np.save(f"./notebooks/simulation_data/linked_lorenz{N}/{method.lower()}/train.npy", train)
        np.save(f"./notebooks/simulation_data/linked_lorenz{N}/{method.lower()}/validation.npy", validation)
        np.save(f"./notebooks/simulation_data/linked_lorenz{N}/{method.lower()}/test.npy", test)
    return train, validation, test

def construct_data_lorenz(a: np.ndarray, data: np.ndarray) -> dict:
    '''
    Given time series data for linked Lorenz system and its adjacency matrix, return training data for each reservoir for the parallel scheme.
    
    Parameters:
    a -- adjacency matrix of the system
    data -- simulated Lorenz system data
    
    Returns:
    a dictionary with oscillator indices as keys and numpy arrays of the given oscillator and its neighbours as values.
    
    Note:
    Assumes the data to be structured so that data[:,3*i:3*i+1] correspond to the i'th subsystem
    '''
    assert data.shape[1] % 3 == 0, "Data should have N*3 columns"
    N = data.shape[1] // 3
    datadict = {}
    
    for system_index in range(N):
        
        datadict[system_index] = data[:, 3*system_index : 3*system_index+3] # data from the subsystem itself
        
        neighbours_indices = np.nonzero(a[system_index])[0]
        for neighbour_index in neighbours_indices: # data from each neighbour of the subsystem
            datadict[system_index] = np.concatenate(
            (datadict[system_index], data[:, 3*neighbour_index : 3*neighbour_index+3]),
            axis = 1)
    return datadict


def get_last_state(internal_trained):
    return internal_trained[-1][-1:,:].T


def rout(r):
    r_out = r.copy()
    r_out[1::2] = np.power(r_out[1::2], 2)
    return r_out

def drout_(r):
    r_out = np.ones(r.shape)
    r_out[1::2] = 2*r[1::2]
    return r_out

def compute_output(train, validation):
    
    train_data = train[0]["data"]
    sync_data = validation[0]["data"]

    params = pd.read_csv("./outputs/linked_lorenz/opt/ll10_opt_single_res_rk45_dt005_median.csv")
    N_in = 30
    input_bias = False
    N_r = 3000
    spectral_radius = params.spectral_radius[0]
    avg_dgr = params.in_degree[0]
    regularization_coef = np.power(10.0, params.exponent_regparam[0].astype(int))
    input_scaling = params.input_con_prob[0]
    leak_rate = params.inverse_timescale[0]

    test_len = 10000
                
    W = reservoirpy.mat_gen.generate_internal_weights(N=N_r, spectral_radius=spectral_radius, proba=avg_dgr/N_r)

    W_in = np.zeros((N_r, N_in))
    indices = np.random.randint(0, N_in, N_r) #nonzero element indices for W_in
    W_in[np.arange(N_r), indices] = np.random.uniform(-input_scaling, input_scaling, N_r) #adding random values for the nonzero indices 

    reservoir = reservoirpy.ESN(lr=leak_rate, W=W, Win=W_in, input_bias=input_bias, ridge=regularization_coef, Wfb=None, fbfunc=None, routfunc=rout)
    train_in = train_data[:train_data.shape[0]-1, :]
    train_out = train_data[0+1:train_data.shape[0],:]

    internal_trained = reservoir.train(inputs=[train_in,], teachers=[train_out,], wash_nr_time_step=5000, verbose=False, regcoef=regularization_coef, regbias=False)

    #run the reservoir over the validation data

    sync_in = sync_data[:sync_data.shape[0], :]
    out_sync, internal_sync = reservoir.run(inputs=[sync_in], init_state=get_last_state(internal_trained))
        
    outputs, internal_states = reservoir.run_auto(init_state=get_last_state(internal_sync), steps=test_len)
    np.save(f"./outputs/linked_lorenz/N{train[0]['N']}/pred_single_res_rk45_{str(train[0]['dt'])[2:]}.npy", outputs)
    return outputs

def compute_vt(test, test_len, threshold, outputs, vt_sub):
    
    N = test[0]["N"]
    test = test[0]["data"]

    for i in range(N):
        normalisation = np.sqrt(np.average(np.power(np.linalg.norm(test[0:test_len, 3*i:3*i+3], axis=1), 2)))
        e = (np.linalg.norm(test[0:test_len, 3*i:3*i+3] - outputs[0:test_len, 3*i:3*i+3], axis=1)) / normalisation
        valid_time = np.where(e > threshold)[0]

        if not np.any(valid_time):
            vt_sub.append(test_len)
        else:
            vt_sub.append(valid_time[0])

    return vt_sub

def main():
    job = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))

    test_len = 5000
    threshold = 0.4
    dt = 0.005
    result_path = f"./outputs/linked_lorenz/N10/dt0_005/vt_single_{job}.pkl"
    try:
        with open(result_path, 'rb') as file:
            vt_sub = pickle.load(file)
    except FileNotFoundError:
        vt_sub = []
    np.random.seed(14)
    a, _, _ = generate_a()
    np.random.seed()

    for i in range(1):
        try:
            with open(result_path, 'rb') as file:
                vt_sub = pickle.load(file)
        except FileNotFoundError:
            vt_sub = []

        train, validation, test = generate_data(a, dt)
        outputs = compute_output(train, validation)
        vt_sub = compute_vt(test, test_len, threshold, outputs, vt_sub)

        with open(result_path, 'wb') as file:
            pickle.dump(vt_sub, file)
            
if __name__ == "__main__":
    main()