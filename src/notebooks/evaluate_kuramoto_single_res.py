import numpy as np
import reservoirpy
import pandas as pd

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

def get_last_state(internal_trained):
    return internal_trained[-1][-1:,:].T

def recover_phases(data:np.ndarray):
    n_col = data.shape[1]
    sine_array = data[:, :int(n_col/2)]
    cosine_array = data[:, int(n_col/2):]
    return np.arctan2(sine_array, cosine_array)

def rout(r):
    r_out = r.copy()
    r_out[1::2] = np.power(r_out[1::2], 2)
    return r_out

def drout_(r):
    r_out = np.ones(r.shape)
    r_out[1::2] = 2*r[1::2]
    return r_out

def compute_output():
    train = np.load("./notebooks/simulation_data/coupled_kuramoto/N30/lsoda_train_02.npy", allow_pickle=True)
    sync = np.load("./notebooks/simulation_data/coupled_kuramoto/N30/lsoda_validation_02.npy", allow_pickle=True)
    test = np.load("./notebooks/simulation_data/coupled_kuramoto/N30/lsoda_test_02.npy", allow_pickle=True)
    train_data = np.concatenate((np.sin(train[0]["data"]), np.cos(train[0]["data"])), axis=1)
    sync_data = np.concatenate((np.sin(sync[0]["data"]), np.cos(sync[0]["data"])), axis=1)
    test_data = np.concatenate((np.sin(test[0]["data"]), np.cos(test[0]["data"])), axis=1)

    params = pd.read_csv("./outputs/kuramoto_opt/kuramoto_30_opt_single_res_lsoda_dt02_median.csv")
    N_in = 60
    input_bias = False
    #N_r = params.Dr[0]
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
    #train the reservoir
    train_in = train_data[:train_data.shape[0]-1, :]
    train_out = train_data[0+1:train_data.shape[0],:]

    internal_trained = reservoir.train(inputs=[train_in,], teachers=[train_out,], wash_nr_time_step=5000, verbose=True, regcoef=regularization_coef, regbias=False)

    sync_in = sync_data[:sync_data.shape[0], :]
    out_sync, internal_sync = reservoir.run(inputs=[sync_in], init_state=get_last_state(internal_trained))
        
    outputs, internal_states = reservoir.run_auto(init_state=get_last_state(internal_sync), steps=test_len)
    np.save(f"./outputs/kuramoto/N{train[0]['N']}/pred_all_single_res_lsoda_{str(train[0]['dt'])[2:]}.npy", outputs)

def main():
    compute_output()

if __name__ == "__main__":
    main()