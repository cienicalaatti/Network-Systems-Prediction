import numpy as np
import pandas as pd
import networkx as nx
import pickle
from shared_memory_parallel_run import sm_parallel_run
from reservoirpy.datasets.chaos import kuramoto

def generate_a(N, S):
    try:
        N = N
        D = 3
        S = S
        G = 5
        K = 0.5
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
        return generate_a(N, S)
    

def kuramoto_euler(Nstep, dt, A, omega, K, init):
    theta = np.zeros((30, Nstep))
    theta[:,0] = init
    for i in range(Nstep-1):
        dtheta = omega + K * np.sum(A.T * np.sin(theta[:, i] - theta[:, i, np.newaxis]), axis=1)
        theta[:, i + 1] = theta[:, i] + dt * dtheta
    return np.mod(theta.T, 2*np.pi)
    

def generate_data(dt, N, omega, a):
    N = 30
    D = 3
    S = 0.8
    G = 5
    K = 0.5
    timestep = dt
    steps = int(1600/timestep)
    trsteps = int((steps*5)/8)
    valsteps = int((steps*3)/16)
    init=np.random.uniform(0, 2*np.pi, size=N)

    data, t = kuramoto.kuramoto(K=K, omega=omega, a=a, init=init, n_timesteps=steps, h=timestep, method='LSODA')
    train = [{"data":data[:trsteps], "dt":timestep, "adjacency_matrix":a, "natural_freq":omega, "K":K, "N":N, "initLen":0}]
    validation = [{"data":data[trsteps:trsteps+valsteps], "dt":timestep, "adjacency_matrix":a, "natural_freq":omega, "K":K, "N":N, "initLen":0}]
    test = [{"data":data[trsteps+valsteps:], "dt":timestep, "adjacency_matrix":a, "natural_freq":omega, "K":K, "N":N, "initLen":0}]

    np.save(f"./notebooks/simulation_data/coupled_kuramoto/N{N}/lsoda_train_{str(timestep)[2:]}.npy", train)
    np.save(f"./notebooks/simulation_data/coupled_kuramoto/N{N}/lsoda_validation_{str(timestep)[2:]}.npy", validation)
    np.save(f"./notebooks/simulation_data/coupled_kuramoto/N{N}/lsoda_test_{str(timestep)[2:]}.npy", test)

    data = kuramoto_euler(steps, timestep, a, omega, K, init)

    train = [{"data":data[:trsteps], "dt":timestep, "adjacency_matrix":a, "natural_freq":omega, "K":K, "N":N, "initLen":0}]
    validation = [{"data":data[trsteps:trsteps+valsteps], "dt":timestep, "adjacency_matrix":a, "natural_freq":omega, "K":K, "N":N, "initLen":0}]
    test = [{"data":data[trsteps+valsteps:], "dt":timestep, "adjacency_matrix":a, "natural_freq":omega, "K":K, "N":N, "initLen":0}]

    np.save(f"./notebooks/simulation_data/coupled_kuramoto/N{N}/euler_train_{str(timestep)[2:]}.npy", train)
    np.save(f"./notebooks/simulation_data/coupled_kuramoto/N{N}/euler_validation_{str(timestep)[2:]}.npy", validation)
    np.save(f"./notebooks/simulation_data/coupled_kuramoto/N{N}/euler_test_{str(timestep)[2:]}.npy", test)

def construct_data(a: np.ndarray, data: np.ndarray) -> dict:
    '''
    Given time series data for a network of kuramoto oscillators and its adjacency matrix, return training data for each reservoir for the parallel scheme.
    
    Parameters:
    a -- adjacency matrix of the system
    data -- time series data for all of the oscillators
    
    Returns:
    a dictionary with oscillator indices as keys and numpy arrays of the given oscillator and its neighbours as values.
    
    Note:
    assuming the first N/2 columns of data contain the sin(theta_i) and the latter N/2 contain the cos(theta_i)
    '''
    n_oscillators = data.shape[1] // 2
    oscillator_data = {}

    for oscillator_index in range(n_oscillators):
        # Calculate the column indices for the sin and cos of the oscillator
        sin_index = oscillator_index
        cos_index = n_oscillators + oscillator_index
        # Get the indices of the neighbors for the oscillator
        neighbors_indices = np.nonzero(a[oscillator_index])[0]

        # Calculate the column indices for the sin and cos of the neighbors
        neighbors_sin_indices = neighbors_indices
        neighbors_cos_indices = n_oscillators + neighbors_indices
        # Concatenate the sin and cos time series for oscillator i and its neighbors
        oscillator_data[oscillator_index] = np.concatenate(
        (data[:, sin_index].reshape(-1, 1),
        data[:, cos_index].reshape(-1, 1),),axis=1)

        for s, c in zip(neighbors_sin_indices, neighbors_cos_indices):
            oscillator_data[oscillator_index] = np.concatenate(
            (oscillator_data[oscillator_index],
             data[:, s].reshape(-1, 1),
             data[:, c].reshape(-1, 1)),axis=1)

    
    return oscillator_data

def R_t(theta, a):
    e = np.e**(1j*theta)
    return np.sum(a * e[:,np.newaxis])

def calculate_R(a:np.ndarray, data:np.ndarray, N:int, d:int):
        R = np.apply_along_axis(R_t, axis=1, arr=data, a=a)
        return R/(N*d)

def recover_phases(data:np.ndarray):
    n_col = data.shape[1]
    sine_array = data[:, :int(n_col/2)]
    cosine_array = data[:, int(n_col/2):]
    return np.arctan2(sine_array, cosine_array)

def recover_phases2(data):
    sine_values = data[:, :, 0]
    cosine_values = data[:, :, 1]
    phase_data = np.arctan2(sine_values, cosine_values)
    return phase_data

def parallel_vt(key, vt_sub, vt_order, dt, threshold=0.4, test_len = 9999):
    # subsystem valid times
    test = np.load(f"./notebooks/simulation_data/coupled_kuramoto/N30/{key}_test_{str(dt)[2:]}.npy", allow_pickle=True)
    test_data = np.concatenate((np.sin(test[0]["data"]), np.cos(test[0]["data"])), axis=1)
    test_datadict = construct_data(a=test[0]["adjacency_matrix"], data=test_data)
    N = test[0]["N"]
    
    outputs = np.load(f"./outputs/kuramoto/N{test[0]['N']}/pred_all_{key}_{str(test[0]['dt'])[2:]}.npy")
    
    for osc in range(30):
        normalisation = np.sqrt(np.average(np.power(np.linalg.norm(test_datadict[osc][0:test_len, 0:2], axis=1), 2)))
        e = np.linalg.norm(test_datadict[osc][0:test_len, 0:2] - outputs[0:test_len, osc, :], axis=1)/normalisation
        valid_time = np.where(e > threshold)[0]
        if not np.any(valid_time):
            vt_sub[key].append(test_len)
        else:
            vt_sub[key].append(valid_time[0])
            
    # order parameter valid time
    phases_test = recover_phases(test_data)
    phases_pred = recover_phases2(outputs)
    D = np.mean(np.sum(test[0]["adjacency_matrix"], axis=1))
    rt_test = calculate_R(a=test[0]["adjacency_matrix"], data=phases_test, N=N, d=D)
    rt_pred = calculate_R(a=test[0]["adjacency_matrix"], data=phases_pred, N=N, d=D)
    
    normalisation = np.sqrt(np.average(np.power(abs(rt_test[0:test_len]), 2)))
    e = (np.abs(rt_test[0:test_len] - rt_pred[0:test_len])) / normalisation
    valid_time = np.where(e > threshold)[0]
    
    if not np.any(valid_time):
        vt_order[key].append(test_len)
    else:
        vt_order[key].append(valid_time[0])
        
    return vt_sub, vt_order


def parallel_dt_test(dt, vt_sub, vt_order):
    np.random.seed(772477)
    N = 30
    S = 0.8
    a, omega, g = generate_a(N, S)

    np.random.seed() #reseed to get only the specific network
    generate_data(dt, 30, omega, a)

    sm_parallel_run('euler')
    sm_parallel_run('lsoda')

    vt_sub, vt_order = parallel_vt("lsoda", vt_sub, vt_order, dt)
    vt_sub, vt_order = parallel_vt("euler", vt_sub, vt_order, dt)
    return vt_sub, vt_order

def main():
    try:
        with open('./outputs/kuramoto/N30/dt0_03/vt_03.pkl', 'rb') as f:
            vt_sub = pickle.load(f)
            vt_order = pickle.load(f)
    except FileNotFoundError:
        vt_sub = {"euler":[], "lsoda":[]}
        vt_order = {"euler":[], "lsoda":[]}

    dt = 0.03
    for i in range(50):
        vt_sub, vt_order = parallel_dt_test(dt, vt_sub, vt_order)

    with open(f'./outputs/kuramoto/N30/dt0_03/vt_{str(dt)[2:]}.pkl', 'wb') as f:
        pickle.dump(vt_sub, f)
        pickle.dump(vt_order, f)

if __name__ == "__main__":
    main()