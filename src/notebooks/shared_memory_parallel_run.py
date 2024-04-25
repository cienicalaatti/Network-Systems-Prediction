import numpy as np
import pandas as pd
import pickle
import click
import reservoirpy
from parallel_utils import construct_data_kuramoto, construct_data_lorenz, get_neighbours, generateW
import joblib


def get_last_state(internal_states):
        return internal_states[-1][-1:,:].T

def get_next_state(W_all, Win_all, leak_rate, u, x):
    # linear transformation
    x1 = np.einsum('ijk,ki->ij', Win_all, u) + np.einsum('ijk,ik->ij',W_all, x)
        
    # previous states memory leak and non-linear transformation
    x1 = (1-leak_rate)*x + leak_rate*np.tanh(x1)

    # return the next state computed
    return x1

def rout(r):
    r_out = r.copy()
    r_out[1::2] = np.power(r_out[1::2], 2)
    return r_out

def drout(r):
    n = len(r)
    return np.eye(n)

def run_auto_parallel(steps, res_dict, init_state, Win_all, W_all, leak_rate, Wout_all):
    dim_in = res_dict[0][0].dim_inp
    n_res = len(res_dict.keys())
    dim_out = res_dict[0][0].dim_out
    neighbor_indices = [res_dict[i][0].neighbours for i in range(n_res)]
        
    outputs = np.zeros((steps, n_res, dim_out))
    states = np.zeros((steps, n_res, res_dict[0][0].N))
    x = init_state.reshape(n_res, res_dict[0][0].N)
        
    for i in range(steps):
        #output
        out = np.einsum('ijk,ik->ij', Wout_all, x)
        outputs[i,:,:] = out
            
        #arrange outputs to fit the reservoir state computation
        out_mod = np.zeros((dim_in, n_res))
        out_mod[:dim_out, :] = out[np.arange(n_res), :].T
        out_mod[dim_out:, :] = out[neighbor_indices,:].reshape((n_res, dim_in-dim_out)).T
        
        states[i,:,:] = x
        #next state
        x = get_next_state(W_all, Win_all, leak_rate, out_mod, x)
            
    return outputs, states, neighbor_indices

def setup_single_res(train, train_datadict, rank, N, spectral_radius, input_proba, leak_rate, regularization_coef, k, subsys_dim):
    n_inputs = train_datadict[rank].shape[1]
        
    # generating W
    W = generateW(N, spectral_radius, k)

    #Win
    Win = np.zeros((N, n_inputs))
    input_scaling = input_proba
    indices = np.random.randint(0, n_inputs, N) #nonzero element indices for W_in
    Win[np.arange(N), indices] = np.random.uniform(-input_scaling, input_scaling, N) #adding random values for the nonzero indices 

    train_in = train_datadict[rank][:train_datadict[rank].shape[0]-1, :]
    train_out = train_datadict[rank][0+1:train_datadict[rank].shape[0], 0:subsys_dim]

    reservoir = reservoirpy.ESN(lr=leak_rate, W=W, Win=Win, input_bias=False, ridge=regularization_coef, Wfb=None, fbfunc=None, routfunc=None, neighbours=get_neighbours(train[0]["adjacency_matrix"], rank))

    internal_trained = reservoir.train(inputs=[train_in, ], teachers=[train_out, ], wash_nr_time_step=5000, verbose=False, regcoef=regularization_coef, regbias=False)
        
    return rank, reservoir, internal_trained

def run_single_res(sync_datadict, res_dict, rank):
    sync_in = sync_datadict[rank][:sync_datadict[rank].shape[0], :]
    _, internal_sync = res_dict[rank][0].run(inputs=[sync_in], init_state=get_last_state(res_dict[rank][1]))
    return rank, get_last_state(internal_sync)

def kuramoto_setup(N, method, dt):
    trdatafile = f"./notebooks/simulation_data/coupled_kuramoto/N{N}/{method}_train_{dt}.npy"
    syncdatafile = f"./notebooks/simulation_data/coupled_kuramoto/N{N}/{method}_validation_{dt}.npy"
    train = np.load(trdatafile, allow_pickle=True)
    traindata = np.concatenate((np.sin(train[0]["data"]), np.cos(train[0]["data"])), axis=1)
    sync = np.load(syncdatafile, allow_pickle=True)
    sync_data = np.concatenate((np.sin(sync[0]["data"]), np.cos(sync[0]["data"])), axis=1)
    train_datadict = construct_data_kuramoto(a=train[0]["adjacency_matrix"], data=traindata)
    sync_datadict = construct_data_kuramoto(a=sync[0]["adjacency_matrix"], data=sync_data)
    params = pd.read_csv(f"./outputs/kuramoto/opt/opt_all_{method}_dt{dt}.csv")
    return train, train_datadict, sync_datadict, params

def llorenz_setup(N, method, dt):

    trdatafile = f"./notebooks/simulation_data/linked_lorenz{N}/{method}/train.npy"
    syncdatafile = f"./notebooks/simulation_data/linked_lorenz{N}/{method}/validation.npy"    
    train = np.load(trdatafile, allow_pickle=True)
    traindata = train[0]["data"]
    train_datadict = construct_data_lorenz(a=train[0]["adjacency_matrix"], data=traindata)
    sync = np.load(syncdatafile, allow_pickle=True)
    sync_data = sync[0]["data"]
    sync_datadict = construct_data_lorenz(a=sync[0]["adjacency_matrix"], data=sync_data)
    params = pd.read_csv(f"./outputs/linked_lorenz/opt/opt_all_{method}_dt{dt}.csv")

    return train, train_datadict, sync_datadict, params

def sm_parallel_run(system, N_res, dt, N_r, method, subsys_dim, test_len):

    #load data
    if system == "kuramoto":
        train, train_datadict, sync_datadict, params = kuramoto_setup(N=30, method=method, dt=dt)
    elif system == "linked_lorenz":
        train, train_datadict, sync_datadict, params = llorenz_setup(N=10, method=method, dt=dt)
    
    #read hyperparameters
    spectral_radius = params.spectral_radius[0]
    k = int(params.in_degree[0])
    regularization_coef = np.power(10.0, params.exponent_regparam[0].astype(int))
    input_proba = params.input_con_prob[0]
    leak_rate = params.inverse_timescale[0]

    # train parallel reservoirs
    res = joblib.Parallel(n_jobs=-1, verbose=0, prefer="processes")(
    joblib.delayed(setup_single_res)(train, train_datadict, i, N_r, spectral_radius, input_proba, leak_rate, regularization_coef, k, subsys_dim) for i in range(N_res)) # magic number here

    #arrange reservoir objects into a dictionary
    res_dict = {r[0]: (r[1], r[2]) for r in res}

    #sync run over the validation portion of the data
    sync_states = joblib.Parallel(n_jobs=-1, verbose=0, prefer="processes")(
    joblib.delayed(run_single_res)(sync_datadict, res_dict, i) for i in range(N_res))

    states_dict = {r[0]: r[1] for r in sync_states}

    # concatenate parallel reservoir matrices for autonomous run
    W_all = np.stack([res_dict[i][0].W for i in res_dict.keys()], axis=0)
    Win_all = np.stack([res_dict[i][0].Win for i in res_dict.keys()], axis=0)
    Wout_all = np.stack([res_dict[i][0].Wout for i in res_dict.keys()], axis=0)
    leak_rate = np.stack([res_dict[i][0].lr for i in res_dict.keys()], axis=0)
    leak_rate = leak_rate.reshape(N_res,1)
    init_states = np.stack([states_dict[i] for i in res_dict.keys()], axis=0)

    out, states, neighbour_indices = run_auto_parallel(test_len, res_dict, init_states, Win_all, W_all, leak_rate, Wout_all)
    
    return out, states, W_all, Win_all, Wout_all, leak_rate, neighbour_indices

@click.command()
@click.option("--system", required=True, type=str, help="Dynamical system to be predicted (kuramoto or linked_lorenz)")
@click.option("--N_res", required=True, type=int, help="Number of parallel reservoirs")
@click.option("--dt", required=True, type=float, help="timestep used in numerical integration")
@click.option("--N_r", required=True, type=int, help="Size of each parallel reservoir")
@click.option("--method", required=True, type=str, help="integration method used to generate the data")
@click.option("--subsys_dim", required=True, type=int, help="number of dimensions in for each subsystem")
@click.option("--test_len", required=True, type=int, help="number of autonomous prediction steps")

def main(system, N_res, dt, N_r, method, subsys_dim, test_len):
    dt = str(dt)[2:]
    out, states, W_all, Win_all, Wout_all, leak_rate, neighbour_indices = sm_parallel_run(system, N_res, dt, N_r, method, subsys_dim, test_len)
    data = [{"out":out, "states":states, "W_all":W_all, "Win_all":Win_all, "Wout_all":Wout_all, "leak_rate":leak_rate, "neighbour_indices":neighbour_indices}]
    np.save(f"./outputs/{system}/N{N_res}/parallel_outputs.npy", data)

if __name__ == "__main__":
    main()