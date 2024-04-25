import numpy as np
import pandas as pd
import joblib
# from reservoirpy import ESN
from shared_memory_parallel_run import run_single_res, setup_single_res, construct_data, get_next_state, construct_data_lorenz

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
        if i < steps-1:
            x = get_next_state(W_all, Win_all, leak_rate, out_mod, x)

    return outputs, x

def qr_pos(Q, R):
    sgn = np.sign(np.diag(R))
    R_new = np.diag(sgn).dot(R)
    Q_new = Q.dot(np.diag(sgn))
    return Q_new, R_new

def recover_phases(data):
    sine_values = data[:, 0]
    cosine_values = data[:, 1]
    phase_data = np.arctan2(sine_values, cosine_values)
    return phase_data

def sine_cosine(phase_data):
    sine_values = np.sin(phase_data)
    cosine_values = np.cos(phase_data)
    reconstructed_data = np.stack([sine_values, cosine_values], axis=-1)
    return reconstructed_data

def kaplan_yorke(lyevec):
    lysum = 0
    i=0
    while lysum >= 0 and i < len(lyevec):
        lysum = lysum + lyevec[i]
        i=i+1    
    return i-1 + (lysum - lyevec[i-1])/np.abs(lyevec[i-1])

def lyaspec(dt, method):

    # setup parallel reservoirs
    #trdatafile = f"./notebooks/simulation_data/coupled_kuramoto/N20/{method}_train_{str(dt)[2:]}.npy"
    #syncdatafile = f"./notebooks/simulation_data/coupled_kuramoto/N20//{method}_validation_{str(dt)[2:]}.npy"
    trdatafile = f"./notebooks/simulation_data/linked_lorenz10/{method}/train.npy"
    syncdatafile = f"./notebooks/simulation_data/linked_lorenz10/{method}/validation.npy"

    train = np.load(trdatafile, allow_pickle=True)
    #traindata = np.concatenate((np.sin(train[0]["data"]), np.cos(train[0]["data"])), axis=1)
    traindata = train[0]["data"]
    train_datadict = construct_data_lorenz(a=train[0]["adjacency_matrix"], data=traindata)

    sync = np.load(syncdatafile, allow_pickle=True)
    #syncdata = np.concatenate((np.sin(sync[0]["data"]), np.cos(sync[0]["data"])), axis=1)
    syncdata = sync[0]["data"]
    sync_datadict = construct_data_lorenz(a=sync[0]["adjacency_matrix"], data=syncdata)

    #params = pd.read_csv(f"./outputs/kuramoto_opt/kuramoto_20_opt_all_{method}_dt{str(dt)[2:]}.csv")
    params = pd.read_csv(f"./outputs/linked_lorenz/opt/ll10_opt_all_{method}_dt{str(dt)[2:]}_median.csv")
    N = 300
    spectral_radius = params.spectral_radius[0]
    k = int(params.in_degree[0])
    regularization_coef = np.power(10.0, params.exponent_regparam[0].astype(int))
    input_proba = params.input_con_prob[0]
    leak_rate = params.inverse_timescale[0]

    res = joblib.Parallel(n_jobs=-1, verbose=0, prefer="processes")(
    joblib.delayed(setup_single_res)(train, train_datadict, i, N, spectral_radius, input_proba, leak_rate, regularization_coef, k) for i in range(10)) # magic number here

    res_dict = {r[0]: (r[1], r[2]) for r in res}

    sync_states = joblib.Parallel(n_jobs=-1, verbose=0, prefer="processes")(
    joblib.delayed(run_single_res)(sync_datadict, res_dict, i) for i in range(10)) #magic number here

    states_dict = {r[0]: r[1] for r in sync_states}

    W_all = np.stack([res_dict[i][0].W for i in res_dict.keys()], axis=0)
    Win_all = np.stack([res_dict[i][0].Win for i in res_dict.keys()], axis=0)
    Wout_all = np.stack([res_dict[i][0].Wout for i in res_dict.keys()], axis=0)
    leak_rate = np.stack([res_dict[i][0].lr for i in res_dict.keys()], axis=0)
    leak_rate = leak_rate.reshape(10,1) # magic number here
    init_states = np.stack([states_dict[i] for i in res_dict.keys()], axis=0)

    dim_in = res_dict[0][0].dim_inp
    n_res = len(res_dict.keys())
    dim_out = res_dict[0][0].dim_out
    neighbor_indices = [res_dict[i][0].neighbours for i in range(n_res)]

    tau = int(0.02 / dt)        # number of initial timesteps (the reservoir is already in an evolved state after training, no need for long initial run)
    N = 100           # total number of renormalizations
    T = int(0.2 / dt)            # number of timesteps between renormalizations
    m = 30       # number of exponents to compute
    eps = 0.000001      # perturbation magnitude

    dim = 30
    log_r = 0

    timestep = dt

    u0, prev_state = run_auto_parallel(tau, res_dict, init_states, Win_all, W_all, leak_rate, Wout_all)
    #u0 = recover_phases(u0) # from sin/cos into phases to be comparable with original data and lyap estimations
    u_prev = u0[-1]
    # choose initial orthogonal directions
    Q0 = np.diag(np.ones(dim))
    Q_prev = Q0

    # for total N renormalizations
    for renorm in range(N):
        # compute uj = u(tj)
        uj, uj_state = run_auto_parallel(T, res_dict, prev_state, Win_all, W_all, leak_rate, Wout_all)
        uj = uj[-1]
        
        # for m lyapunov exponents/directions
        Phi_Q = np.zeros((dim, dim))
        for i in range(m):

            out = u_prev + (eps*Q_prev[:, i]).reshape(n_res, dim_out)
            out_mod = np.zeros((dim_in, n_res))
            out_mod[:dim_out, :] = out[np.arange(n_res), :].T
            out_mod[dim_out:, :] = out[neighbor_indices,:].reshape((n_res, dim_in-dim_out)).T

            prev_state = get_next_state(W_all, Win_all, leak_rate, out_mod, prev_state)

            w, _ = run_auto_parallel(T, res_dict, prev_state, Win_all, W_all, leak_rate, Wout_all)
            w = w[-1]
            Phi_Q[:, i] = ((w-uj)/eps).reshape(-1)

        Q, R = np.linalg.qr(Phi_Q)  ### Q, R decomposition. Q is a matrix with orthonormal columns.
        Q, R = qr_pos(Q, R)
        log_r = log_r + np.log(np.diag(R))
        
        u_prev = uj
        prev_state = uj_state
        Q_prev = Q

        #how does the spectrum evolve over computation?
        if renorm % 2 == 0:
            print("maximum exponent:", np.max(log_r/(N*T*timestep)))
            print("KY:", kaplan_yorke(log_r/(N*T*timestep)))
        

    # calculate the exponents
    return log_r/(N*T*timestep)


def main():
    Euler = []
    lsoda = []
    rk45 = []
    #data = [{"euler":Euler, "lsoda":lsoda}]
    data = [{"rk45": rk45}]

    timesteps = [0.01]

    for dt in timesteps:
        for method in data[0].keys():
            data[0][method].append(lyaspec(dt, method))
    #np.save(f"./outputs/kuramoto/N20/lyespec_dt_test/reservoir_dt{str(dt)[2:]}.npy", data)
    np.save(f"./outputs/linked_lorenz/N10/lyespec_dt_test/reservoir_dt{str(dt)[2:]}.npy", data)
if __name__ == "__main__":
    main()