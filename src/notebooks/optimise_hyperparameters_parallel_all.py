import numpy as np
import pandas as pd
import reservoirpy
import networkx as nx
import click
from skopt import gp_minimize
import joblib
from parallel_utils import construct_data_kuramoto, construct_data_lorenz, get_neighbours, generateW

min_vt_inv_global = 100
W_opt_global = None
Win_opt_global = None


def preprocess_lorenz(trdatafile, testdatafile):
    train = np.load(trdatafile, allow_pickle=True)
    testdata = np.load(testdatafile, allow_pickle=True)[0]["data"]
    train_datadict = construct_data_lorenz(a=train[0]["adjacency_matrix"], data=train[0]["data"])
    test_datadict = construct_data_lorenz(a=train[0]["adjacency_matrix"], data=testdata)
    return train, testdata, train_datadict, test_datadict

def preprocess_kuramoto(trdatafile, testdatafile):
    train = np.load(trdatafile, allow_pickle=True)
    test = np.load(testdatafile, allow_pickle=True)
    traindata = np.concatenate((np.sin(train[0]["data"]), np.cos(train[0]["data"])), axis=1)
    testdata = np.concatenate((np.sin(test[0]["data"]), np.cos(test[0]["data"])), axis=1)
    train_datadict = construct_data_kuramoto(a=train[0]["adjacency_matrix"], data=traindata)
    test_datadict = construct_data_kuramoto(a=train[0]["adjacency_matrix"], data=testdata)
    return train, testdata, train_datadict, test_datadict



def get_optparam(W_topo, N, trdatafile, testdatafile, boundr,
                 ncalls, nstarts, inputseed, subsystem_dim, test_len):

    if inputseed is not None:
        np.random.seed(inputseed)
        seed = [inputseed]
    else:
        sq = np.random.SeedSequence()
        seed = sq.generate_state(1)
        np.random.seed(seed)

    train, testdata, train_datadict, test_datadict = preprocess_lorenz(trdatafile, testdatafile)


    input_bias = False  # add a constant input to 1

    def setup_single_res(rank, N, spectral_radius, input_proba, leak_rate, regularization_coef, k, subsystem_dim):
        n_inputs = train_datadict[rank].shape[1]
        
        # generating W
        W = generateW(N, spectral_radius, k)

        #Win
        Win = np.zeros((N, n_inputs))
        input_scaling = input_proba
        indices = np.random.randint(0, n_inputs, N) #nonzero element indices for W_in
        Win[np.arange(N), indices] = np.random.uniform(-input_scaling, input_scaling, N) #adding random values for the nonzero indices 

        train_in = train_datadict[rank][:train_datadict[rank].shape[0]-1, :]
        train_out = train_datadict[rank][0+1:train_datadict[rank].shape[0], 0:subsystem_dim]

        reservoir = reservoirpy.ESN(lr=leak_rate, W=W, Win=Win, input_bias=input_bias, ridge=regularization_coef, Wfb=None, fbfunc=None, routfunc=None, neighbours=get_neighbours(train[0]["adjacency_matrix"], rank))

        internal_trained = reservoir.train(inputs=[train_in, ], teachers=[train_out, ], wash_nr_time_step=5000, verbose=False, regcoef=regularization_coef, regbias=False)
        
        return rank, reservoir, internal_trained

    def f(x):
        spectral_radius = x[0]
        input_proba = x[1]
        input_std = x[2]
        leak_rate = x[3]
        regularization_coef = np.power(10.0, x[4].astype(int))
        k = x[5]

        vt_inv = get_vt_inv(N, spectral_radius, input_proba, leak_rate, regularization_coef, k, subsystem_dim, test_len)
        return vt_inv

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

    def run_auto_parallel(steps, res_dict, init_state, Win_all, W_all, leak_rate, Wout_all):
        dim_in = res_dict[0][0].dim_inp
        n_res = len(res_dict.keys())
        dim_out = res_dict[0][0].dim_out
        neighbor_indices = [res_dict[i][0].neighbours for i in range(n_res)]
        
        outputs = np.zeros((steps, n_res, dim_out))
        x = init_state.reshape(n_res, res_dict[0][0].N)
        
        for i in range(steps):
            #output
            out = np.einsum('ijk,ik->ij', Wout_all, x)
            outputs[i,:,:] = out
            
            #arrange outputs to fit the reservoir state computation
            out_mod = np.zeros((dim_in, n_res))
            out_mod[:dim_out, :] = out[np.arange(n_res), :].T
            out_mod[dim_out:, :] = out[neighbor_indices,:].reshape((n_res, dim_in-dim_out)).T
            
            #next state
            x = get_next_state(W_all, Win_all, leak_rate, out_mod, x)
            
        return outputs

    
    def get_vt_inv(N_res, N, spectral_radius, input_proba, leak_rate, regularization_coef, k, subsystem_dim, test_len):

        res = joblib.Parallel(n_jobs=-1, verbose=0, prefer="processes")(
        joblib.delayed(setup_single_res)(i, N, spectral_radius, input_proba, leak_rate, regularization_coef, k, subsystem_dim) for i in range(N_res))


        res_dict = {r[0]: (r[1], r[2]) for r in res}

        W_all = np.stack([res_dict[i][0].W for i in res_dict.keys()], axis=0)
        Win_all = np.stack([res_dict[i][0].Win for i in res_dict.keys()], axis=0)
        Wout_all = np.stack([res_dict[i][0].Wout for i in res_dict.keys()], axis=0)
        leak_rate = np.stack([res_dict[i][0].lr for i in res_dict.keys()], axis=0)
        leak_rate = leak_rate.reshape(N_res,1)
        init_states = np.stack([get_last_state(res_dict[i][1]) for i in res_dict.keys()], axis=0)

        threshold = 0.4
        out = run_auto_parallel(test_len, res_dict, init_states, Win_all, W_all, leak_rate, Wout_all)

        #computing average valid time of prediction (and minimize the inverse)



        times = []
        sys_dim = res_dict[0][0].dim_out
        
        for s in range(len(res_dict.keys())):
            normalisation = np.sqrt(np.average(np.power(np.linalg.norm(test_datadict[s][0:test_len, 0:sys_dim], axis=1), 2)))
            e = np.linalg.norm(test_datadict[s][0:test_len, 0:sys_dim] - out[0:test_len, s, :], axis=1)/normalisation

            valid_time = np.where(e > threshold)[0]
            if not np.any(valid_time):
                times.append(test_len)
            else:
                times.append(valid_time[0])
        vt_inv = 1/np.mean(times)
        
        global min_vt_inv_global
        global W_opt_global
        global Win_opt_global
        if vt_inv < min_vt_inv_global:
            min_vt_inv_global = vt_inv
            W_opt_global = W_all
            Win_opt_global = Win_all
        return vt_inv


    # param boundariers:
        # spectral_radius = x[0]
        # input_proba = x[1]
        # input_std = x[2]
        # leak_rate = x[3]
        # reg_coeff_exponent = x[4]
        # avg_degree = x[5]

    # The default optimisation boundaries A:
    parameter_boundaries = [(0.0, 1.5), (0.1, 1.0), (0.1, 1.5), (0.05, 1), (-9, 0)]
    # Other optimisation boundaries:
    if boundr == 'B':
        parameter_boundaries = [(0.0, 1.5), (0.0, 1.0), (0.0, 1.5), (0.0, 1), (-5, 0)]
    if boundr == 'lappu':
        parameter_boundaries = [(0.3, 1.5), (0.1, 1.0), (0.3, 1.5), (0.07, 0.11), (-5, 5)]
    if boundr == 'kuramoto':
        parameter_boundaries = [(0.0, 1.2), (0.1, 1.0), (0.3, 1.5), (0.05, 0.2), (-9, 0)]
    if boundr == 'linked_lorenz':
        parameter_boundaries = [(0.0, 1.5), (0.1, 1.0), (0.1, 1.5), (0.05, 1), (-9, 0)]

    if W_topo == '-':
        parameter_boundaries.append((1, 5))
        # parameter_boundaries.append((1, 10))

    # Hyperparameter optimisation. gp_minimize (in skopt - scikit-optimize) does Bayesian optimisation using Gaussian processes.
    res = gp_minimize(f,                  # the function to minimize
                      parameter_boundaries,
                      acq_func="EI",      # the acquisition function
                      n_calls=ncalls,         # the number of evaluations of f
                      n_random_starts=nstarts)  # the number of random initialization points

    

    return res.x, min_vt_inv_global


@click.command()
@click.option("--system", required=True, type=str, help="Dynamical system to be predicted (kuramoto or linked_lorenz)")
@click.option("--N_res", required=True, type=int, help="Number of parallel reservoirs")
@click.option("--dt", required=True, type=float, help="timestep used in numerical integration")
@click.option("--N_r", required=True, type=int, help="Size of each parallel reservoir")
@click.option("--method", required=True, type=str, help="integration method used to generate the data")
@click.option("--subsys_dim", required=True, type=int, help="number of dimensions in for each subsystem")
@click.option("--test_len", required=True, type=int, help="number of autonomous prediction steps")
@click.option("--ncalls", required=True, type=int, default=100, help="number of Bayesian optimisation iterations")
@click.option("--nstarts", required=True, type=int, default=2, help="number of random inits during optimisation calls")
@click.option("--seed", required=False, type=int, default=None, help="rng seed")



def main(system, N_res, dt, N_r, method, subsys_dim, test_len, ncalls, nstarts, seed):
    
    dt = str(dt)[2:]
    
    if system == "kuramoto":
        trdatafile = f"./notebooks/simulation_data/coupled_kuramoto/N{N_res}/{method}_train_{dt}.npy"
        testdatafile = f"./notebooks/simulation_data/coupled_kuramoto/N{N_res}/{method}_validation_{dt}.npy"
    elif system == "linked_lorenz":
        trdatafile = f"./notebooks/simulation_data/linked_lorenz{N_res}/{method}/train.npy"
        testdatafile = f"./notebooks/simulation_data/linked_lorenz{N_res}/{method}/validation.npy"

    w_topo = '-'
    boundr = system
    rng = np.random.default_rng()
    if seed == None:
        seed = rng.integers(999999)
    
    optx, vt_inv = get_optparam(
        w_topo, N_r, trdatafile, testdatafile, boundr, ncalls, nstarts, seed, subsys_dim, test_len)
    optx.append(1/vt_inv)
    optx.append(seed)
    df = pd.DataFrame([optx], columns=["spectral_radius", "input_con_prob", "input_specrad", "inverse_timescale", "exponent_regparam", "in_degree", "valid_time", "seed"])
    df.to_csv(f"./outputs/{system}/opt/opt_all_{method}_dt{dt}.csv", index=False)

if __name__ == '__main__':
    main()
