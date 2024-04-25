import numpy as np
import pandas as pd
import networkx as nx

def construct_data_kuramoto(a: np.ndarray, data: np.ndarray) -> dict:
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

def get_neighbours(a, node):
    return list(np.nonzero(a[node])[0])

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


# generating the network matrix (W) of the reservoir (Erdos-Renyi network)
def generateW(N, sp, k):
    spectral_radius = sp
    W = np.zeros((N, N))
    for i in range(N):
        indices = np.random.choice(N, k, replace=False)
        W[i, indices] = np.random.normal(size=k)
    W = spectral_radius/max(abs(np.linalg.eig(W)[0]))*W
    return W