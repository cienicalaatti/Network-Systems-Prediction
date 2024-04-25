from typing import Callable, Sequence
from .esn import ESN
import numpy as np

class ParallelESN(ESN):

    def __init__(self, lr: float, W: np.ndarray, Win: np.ndarray, input_bias: bool = True, reg_model: Callable = None, ridge: float = None, Wfb: np.ndarray = None, fbfunc: Callable = None, routfunc: Callable = None, typefloat: np.dtype = np.float64, rank: int = None, neighbours: list = None):
        super().__init__(lr, W, Win, input_bias, reg_model, ridge, Wfb, fbfunc, routfunc, typefloat, rank, neighbours)
        self.n_res = W.shape[0]

    def _autocheck_dimensions(self):
        """ Auto-check to see if ESN matrices have correct dimensions."""
        #override the parent method
        pass

    def _get_next_state(self, u, x):

        # linear transformation
        x1 = np.einsum('ijk,ki->ij', self.Win, u) + np.einsum('ijk,ik->ij',self.W, x)
        
        # previous states memory leak and non-linear transformation
        x1 = (1-self.lr)*x + self.lr*np.tanh(x1)

        # return the next state computed
        return x1

    def _compute_states(self, inputs, init_state):
        if init_state is None:
            init_state = np.zeros(shape=(50,320))

        states = np.zeros((self.n_res, self.N, len(inputs)))
        current_state = init_state.copy()
        
        for t in range(inputs.shape[0]):
            current_state = self._get_next_state(inputs[t,:,:], current_state)
            states[:,:,t] = current_state
            
        return states
    
    def fit_readout(self, states: Sequence, teachers: Sequence, reg_model: Callable = None, ridge: float = None, regbias: bool = True, force_pinv: bool = False, verbose: bool = False) -> np.ndarray:
        if (ridge is not None) or (reg_model is not None):
            reg_model = self._get_regression_model(ridge, reg_model)
            if verbose:
                print('Ridge regression with coef', ridge)
        elif force_pinv:
            reg_model = self._get_regression_model(None, None)
        else:
            reg_model = self.reg_model

        # Building Wout with a linear regression model.
        # saving the output matrix in the ESN object for later use
        Wout = reg_model(self.routfunc(states), teachers)

        # return readout matrix
        return Wout
    
    def train(self, inputs: Sequence, teachers: Sequence, wash_nr_time_step: int = 0, regbias: bool = True, regcoef: float = None, workers: int = -1, backend: str = "threading", verbose: bool = False) -> Sequence:
    
        all_states = self._compute_states(inputs=inputs, init_state=None)

        self.Wout = self.fit_readout(states=all_states, teachers=teachers, ridge=regcoef)

        self.dim_out = self.Wout.shape[0]