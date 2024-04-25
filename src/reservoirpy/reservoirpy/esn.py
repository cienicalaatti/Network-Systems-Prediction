# -*- coding: utf-8 -*-
#!/usr/bin/env python -W ignore::DeprecationWarning
"""reservoirpy/ESN

Simple, parallelizable implementation of Echo State Networks.

@author: Xavier HINAUT
xavier.hinaut@inria.fr
Copyright Xavier Hinaut 2018

I would like to thank Mantas Lukosevicius for his code that was used as inspiration for this code:
http://minds.jacobs-university.de/mantas/code
"""
import os
import time
import warnings
import pickle
from typing import Sequence, Callable, Tuple, Union, Dict

import joblib
import numpy as np
from scipy import linalg
from tqdm import tqdm

from .utils import check_values, _save
from .regression_models import sklearn_linear_model
from .regression_models import ridge_linear_model
from .regression_models import pseudo_inverse_linear_model

import matplotlib.pyplot as plt
from mpi4py import MPI

class ESN(object):

    def __init__(self,
                 lr: float,
                 W: np.ndarray,
                 Win: np.ndarray,
                 input_bias: bool=True,
                 reg_model: Callable=None,
                 ridge: float=None,
                 Wfb: np.ndarray=None,
                 fbfunc: Callable=None,
                 routfunc: Callable= None,
                 actfunc: Callable= None,
                 typefloat: np.dtype=np.float64):
        """Base class of Echo State Networks

        Arguments:
            lr {float} -- Leaking rate
            W {np.ndarray} -- Reservoir weights matrix
            Win {np.ndarray} -- Input weights matrix

        Keyword Arguments:
            input_bias {bool} -- If True, will add a constant bias
                                 to the input vector (default: {True})
            reg_model {Callable} -- A scikit-learn linear model function to use for regression. Should
                                   be None if ridge is used. (default: {None})
            ridge {float} -- Ridge regularization coefficient for Tikonov regression. Should be None
                             if reg_model is used. (default: {None})
            Wfb {np.array} -- Feedback weights matrix. (default: {None})
            fbfunc {Callable} -- Feedback activation function. (default: {None})
            routfunc {Callable} -- Reservoir output function. (default: {None})
            actfunc {Callable} -- Reservoir activation function. (default: {None}) In default case uses tanh.
            typefloat {np.dtype} -- Float precision to use (default: {np.float32})

        Raises:
            ValueError: If a feedback matrix is passed without activation function.
            NotImplementedError: If trying to set input_bias to False. This is not
                                 implemented yet.
        """

        self.W = W
        self.Win = Win
        self.Wout = None # output weights matrix. must be learnt through training.
        self.Wfb = Wfb

        # check if dimensions of matrices are coherent
        self._autocheck_dimensions()
        self._autocheck_nan()

        self.N = self.W.shape[1] # number of neurons
        self.in_bias = input_bias
        self.dim_inp = self.Win.shape[1] # dimension of inputs (including the bias at 1)
        self.dim_out = None
        if self.Wfb is not None:
            self.dim_out = self.Wfb.shape[1] # dimension of outputs

        self.typefloat = typefloat
        self.lr = lr # leaking rate

        self.reg_model = self._get_regression_model(ridge, reg_model)
        self.fbfunc = fbfunc
        if self.Wfb is not None and self.fbfunc is None:
            raise ValueError(f"If a feedback matrix is provided, \
                fbfunc must be a callable object, not {self.fbfunc}.")

        if routfunc is None:
            def rout(r):
                return r
            routfunc = rout
        self.routfunc = routfunc
        
        if actfunc is None:
            def actf(x):
                return np.tanh(x)
            actfunc = actf
        self.actfunc = actfunc


    def __repr__(self):
        trained = True
        if self.Wout is None:
            trained = False
        fb = True
        if self.Wfb is None:
            fb=False
        out = f"ESN(trained={trained}, feedback={fb}, N={self.N}, "
        out += f"lr={self.lr}, input_bias={self.in_bias}, input_dim={self.N})"
        return out


    def _get_regression_model(self, ridge: float=None, sklearn_model: Callable=None):
        """Set the type of regression used in the model. All regression models available
        for now are described in reservoipy.regression_models:
            - any scikit-learn linear regression model (like Lasso or Ridge)
            - Tikhonov linear regression (l1 regularization)
            - Solving system with pseudo-inverse matrix
        Keyword Arguments:
            ridge {[float]} -- Ridge regularization coefficient. (default: {None})
            sklearn_model {[Callable]} -- scikit-learn regression model to use. (default: {None})

        Raises:
            ValueError: if ridge and scikit-learn models are requested at the same time.

        Returns:
            [Callable] -- A linear regression function.
        """
        if ridge is not None and sklearn_model is not None:
            raise ValueError("ridge and sklearn_model can't be defined at the same time.")

        elif ridge is not None:
            self.ridge = ridge
            return ridge_linear_model(self.ridge)

        elif sklearn_model is not None:
            self.sklearn_model = sklearn_model
            return sklearn_linear_model(self.sklearn_model)

        else:
            return pseudo_inverse_linear_model()

    def _autocheck_nan(self):
        """ Auto-check to see if some important variables do not have a problem (e.g. NAN values). """
        #assert np.isnan(self.W).any() == False, "W matrix should not contain NaN values."
        assert np.isnan(self.Win).any() == False, "Win matrix should not contain NaN values."
        if self.Wfb is not None:
            assert np.isnan(self.Wfb).any() == False, "Wfb matrix should not contain NaN values."


    def _autocheck_dimensions(self):
        """ Auto-check to see if ESN matrices have correct dimensions."""
        # W dimensions check list
        assert len(self.W.shape) == 2, f"W shape should be (N, N) but is {self.W.shape}."
        assert self.W.shape[0] == self.W.shape[1], f"W shape should be (N, N) but is {self.W.shape}."

        # Win dimensions check list
        assert len(self.Win.shape) == 2, f"Win shape should be (N, input) but is {self.Win.shape}."
        err = f"Win shape should be ({self.W.shape[1]}, input) but is {self.Win.shape}."
        assert self.Win.shape[0] == self.W.shape[0], err


    def _autocheck_io(self,
                      inputs,
                      outputs=None):

        # Check if inputs and outputs are lists
        assert type(inputs) is list, "Inputs should be a list of numpy arrays"
        if outputs is not None:
            assert type(outputs) is list, "Outputs should be a list of numpy arrays"

        # check if Win matrix has coherent dimensions with input dimensions
        if self.in_bias:
            err = f"With bias, Win matrix should be of shape ({self.N}, "
            err += f"{inputs[0].shape[1] + 1}) but is {self.Win.shape}."
            assert self.Win.shape[1] == inputs[0].shape[1] + 1, err
        else:
            err = f"Win matrix should be of shape ({self.N}, "
            err += f"{self.dim_inp}) but is {self.Win.shape}."
            assert self.Win.shape[1] == inputs[0].shape[1], err

        if outputs is not None:
            # check feedback matrix
            if self.Wfb is not None:
                err = f"With feedback, Wfb matrix should be of shape ({self.N}, "
                err += f"{outputs[0].shape[1]}) but is {self.Wfb.shape}."
                assert outputs[0].shape[1] == self.Wfb.shape[1], err


    def _get_next_state(self,
                        single_input: np.ndarray,
                        feedback: np.ndarray=None,
                        last_state: np.ndarray=None) -> np.ndarray:
        """Given a state vector x(t) and an input vector u(t), compute the state vector x(t+1).

        Arguments:
            single_input {np.ndarray} -- Input vector u(t).

        Keyword Arguments:
            feedback {np.ndarray} -- Feedback vector if enabled. (default: {None})
            last_state {np.ndarray} -- Current state to update x(t). If None,
                                       state is initialized to 0. (default: {None})

        Raises:
            RuntimeError: feedback is enabled but no feedback vector is available.

        Returns:
            np.ndarray -- Next state x(t+1).
        """

        # check if the user is trying to add empty feedback
        if self.Wfb is not None and feedback is None:
            raise RuntimeError("Missing a feedback vector.")

        # warn if the user is adding a feedback vector when feedback is not available
        # (might have forgotten the feedback weights matrix)
        if self.Wfb is None and feedback is not None:
            warnings.warn("Feedback vector should not be passed to update_state if no feedback matrix is provided.", UserWarning)

        # first initialize the current state of the ESN
        if last_state is None:
            x = np.zeros((self.N,1),dtype=self.typefloat)
            warnings.warn("No previous state was passed for computation of next state. Will assume a 0 vector as initial state to compute the update.", UserWarning)
        else:
            x = last_state

        # add bias
        if self.in_bias:
            u = np.hstack((1, single_input)).astype(self.typefloat)
        else:
            u = single_input

        # linear transformation
        x1 = np.dot(self.Win, u.reshape(self.dim_inp, 1)) \
            + self.W.dot(x)

        # add feedback if requested
        if self.Wfb is not None:
            x1 += np.dot(self.Wfb, self.fbfunc(feedback))

        # previous states memory leak and non-linear transformation
        x1 = (1-self.lr)*x + self.lr*self.actfunc(x1)

        # return the next state computed
        return x1


    def _compute_states(self,
                        input: np.ndarray,
                        forced_teacher: np.ndarray=None,
                        init_state: np.ndarray=None,
                        init_fb: np.ndarray=None,
                        wash_nr_time_step: int=0,
                        input_id: int=None) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Compute all states generated from a single sequence of inputs.

        Arguments:
            input {np.ndarray} -- Sequence of inputs.

        Keyword Arguments:
            forced_teacher {np.ndarray} -- Ground truth vectors to use as feedback
                                           during training, if feedback is enabled.
                                           (default: {None})
            init_state {np.ndarray} -- Initialization vector for states. (default: {None})
            init_fb {np.ndarray} -- Initialization vector for feedback. (default: {None})
            wash_nr_time_step {int} -- Number of states to considered as transitory
                            when training. (default: {0})
            input_id {int} -- Index of the input in the queue. Used for parallelization
                              of computations. (default: {None})

        Raises:
            RuntimeError: raised if no teachers are specifiyed for training with feedback.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], np.ndarray] -- Index of the input in queue
            and computed states, or just states if no index is provided.
        """

        if self.Wfb is not None and forced_teacher is None and self.Wout is None:
            raise RuntimeError("Impossible to use feedback without readout matrix or teacher forcing.")

        # to track successives internal states of the reservoir
        states = np.zeros((self.N, len(input)-wash_nr_time_step), dtype=self.typefloat)

        # if a feedback matrix is available, feedback will be set to 0 or to
        # a specific value.
        if self.Wfb is not None:
            if init_fb is None:
                last_feedback = np.zeros((self.dim_out, 1), dtype=self.typefloat)
            else:
                last_feedback = init_fb.copy()
        else:
            last_feedback = None

        # State is initialized to 0 or to a specific value.
        if init_state is None:
            current_state = np.zeros((self.N, 1),dtype=self.typefloat)
        else:
            current_state = init_state.copy().reshape(-1, 1)

        # for each time step in the input
        for t in range(input.shape[0]):
            # compute next state from current state
            current_state = self._get_next_state(input[t, :], feedback=last_feedback, last_state=current_state)

            # compute last feedback
            if self.Wfb is not None:
                # during training outputs are equal to teachers for feedback
                if forced_teacher is not None:
                    last_feedback = forced_teacher[t,:].reshape(forced_teacher.shape[1], 1).astype(self.typefloat)
                # feedback of outputs, computed with Wout
                else:
                    last_feedback = np.dot(self.Wout, np.vstack((1,current_state))).astype(self.typefloat)
                last_feedback = last_feedback.reshape(self.dim_out, 1)

            # will track all internal states during inference, and only the
            # states after wash_nr_time_step during training.
            if t >= wash_nr_time_step:
                states[:, t-wash_nr_time_step] = current_state.reshape(-1,).astype(self.typefloat)

        if input_id is None:
            return 0, states
        return input_id, states


    def compute_all_states(self,
                           inputs: Sequence[np.ndarray],
                           forced_teachers: Sequence[np.ndarray]=None,
                           init_state: np.ndarray=None,
                           init_fb: np.ndarray=None,
                           wash_nr_time_step: int=0,
                           workers: int=-1,
                           backend: str="threading",
                           verbose: bool=True) -> Sequence[np.ndarray]:
        """Compute all states generated from sequences of inputs.

        Arguments:
            inputs {Sequence[np.ndarray]} -- Sequence of input sequences.

        Keyword Arguments:
            forced_teachers {Sequence[np.ndarray]} -- Sequence of ground truth
                                                      sequences, for training with
                                                      feedback. (default: {None})
            init_state {np.ndarray} -- State initialization vector
                                       for all inputs. (default: {None})
            init_fb {np.ndarray} -- Feedback initialization vector
                                    for all inputs, if feedback is
                                    enabled. (default: {None})
            wash_nr_time_step {int} -- Number of states to considered as transitory
                            when training. (default: {0})
            workers {int} -- if n >= 1, will enable parallelization of
                             states computation with n threads/processes, if possible.
                             if n = -1, will use all available resources for
                             parallelization. (default: {-1})
            backend {str} -- Backend used for parallelization of
                             states computations. Available backends are
                             `threadings`(recommended, see `train` Note), `multiprocessing`,
                             `loky` (default: {"threading"}).
            verbose {bool} -- if `True`, display progress in stdout.

        Returns:
            Sequence[np.ndarray] -- All computed states.
        """

        # initialization of workers
        loop = joblib.Parallel(n_jobs=workers, backend=backend)
        delayed_states = joblib.delayed(self._compute_states)

        # progress bar if needed
        if verbose:
            track = tqdm
        else:
            track = lambda x, text: x

        # no feedback training or running
        if forced_teachers is None:
            all_states = loop(delayed_states(inputs[i], wash_nr_time_step=wash_nr_time_step, input_id=i,
                                             init_state=init_state, init_fb=init_fb)
                              for i in track(range(len(inputs)), "Computing states"))
        # feedback training
        else:
            all_states = loop(delayed_states(inputs[i], forced_teachers[i], wash_nr_time_step=wash_nr_time_step,
                                             input_id=i, init_state=init_state, init_fb=init_fb)
                              for i in track(range(len(inputs)), "Computing states"))

        # input ids are used to make sure that the returned states are in the same order
        # as inputs, because parallelization can change this order.
        return [s[1] for s in sorted(all_states, key=lambda x: x[0])]


    def compute_outputs(self,
                        states: Sequence[np.ndarray],
                        verbose: bool=False) -> Sequence[np.ndarray]:
        """Compute all readouts of a given sequence of states.

        Arguments:
            states {Sequence[np.ndarray]} -- Sequence of states.

        Keyword Arguments:
            verbose {bool} -- Set verbosity.

        Raises:
            RuntimeError: no readout matrix Wout is available.
            Consider training model first, or load an existing matrix.

        Returns:
            Sequence[np.ndarray] -- All outputs of readout matrix.
        """
        # because all states and readouts will be concatenated,
        # first save the indexes of each inputs states in the concatenated vector.
        if self.Wout is not None:
            idx = [None] * len(states)
            c = 0
            for i, s in enumerate(states):
                idx[i] = [j for j in range(c, c+s.shape[1])]
                c += s.shape[1]

            if verbose:
                print("Computing outputs...")
                tic = time.time()

            #states[0][1::2] = np.power(states[0][1::2], 2)              ### Wout r r2
            x = np.hstack(states)

            # check if Wout has bias
            if self.Wout.shape[1] > x.shape[0]:
                x = np.vstack((np.ones((x.shape[1],), dtype=self.typefloat), x))   # adding row of ones to x

            y = np.dot(self.Wout, self.routfunc(x)).astype(self.typefloat)

            if verbose:
                toc = time.time()
                print(f"Outputs computed! (in {toc - tic}sec)")

            # return separated readouts vectors corresponding to the saved
            # indexes built with input states.
            return [y[:, i] for i in idx]

        else:
            raise RuntimeError("Impossible to compute outputs: no readout matrix available.")


    def fit_readout(self,
                    states: Sequence,
                    teachers: Sequence,
                    reg_model: Callable=None,
                    ridge: float=None,
                    regbias: bool=True,
                    force_pinv: bool=False,
                    verbose: bool=False) -> np.ndarray:
        """Compute a readout matrix by fitting the states computed by the ESN
        to the ground truth expected values, using the regression model defined
        in the ESN.

        Arguments:
            states {Sequence} -- All states computed.
            teachers {Sequence} -- All ground truth vectors.

        Keyword Arguments:
            reg_model {scikit-learn regression model} -- Use a scikit-learn regression model. (default: {None})
            ridge {float} -- Use Tikhonov regression and set regularization parameter to ridge. (default: {None})
            force_pinv -- Overwrite all previous parameters and force pseudo-inverse resolution. (default: {False})
            verbose {bool} -- (default: {False})

        Returns:
            np.ndarray -- Readout matrix.
        """
        # switch the regression model used at instanciation if needed.
        # WARNING: this change won't be saved by the save function.
        if (ridge is not None) or (reg_model is not None):
            reg_model = self._get_regression_model(ridge, reg_model)
            if verbose:
                print('Ridge regression with coef', ridge)
        elif force_pinv:
            reg_model = self._get_regression_model(None, None)
        else:
            reg_model = self.reg_model

        # check if network responses are valid
        check_values(array_or_list=states, value=None)

        if verbose:
            tic = time.time()
            print("Linear regression...")

        # concatenate the lists (along timestep axis)
        X = np.hstack(states).astype(self.typefloat)
        Y = np.hstack(teachers).astype(self.typefloat)

        if regbias:
            # Adding ones for regression with bias b in (y = a*x + b)
            X = np.vstack((np.ones((1, X.shape[1]),dtype=self.typefloat), X))

        # Building Wout with a linear regression model.
        # saving the output matrix in the ESN object for later use
        Wout = reg_model(self.routfunc(X), Y)

        if verbose:
            toc = time.time()
            print(f"Linear regression done! (in {toc - tic} sec)")

        # return readout matrix
        return Wout


    def train(self,
              inputs: Sequence[np.ndarray],
              teachers: Sequence[np.ndarray],
              wash_nr_time_step: int=0,
              regbias: bool=True,
              regcoef: float=None,
              workers: int=-1,
              backend: str="threading",
              verbose: bool=False) -> Sequence[np.ndarray]:
        """Train the ESN model on a sequence of inputs.

        Arguments:
            inputs {Sequence[np.ndarray]} -- Training set of inputs.
            teachers {Sequence[np.ndarray]} -- Training set of ground truth.

        Keyword Arguments:
            wash_nr_time_step {int} -- Number of states to considered as transitory
                            when training. (default: {0})
            workers {int} -- if n >= 1, will enable parallelization of
                             states computation with n threads/processes, if possible.
                             if n = -1, will use all available resources for
                             parallelization.
            backend {str} -- Backend used for parallelization of
                             states computations. Available backends are
                             `threadings`(recommended, see Note), `multiprocess`,
                             `loky` (default: {"threading"}).
            verbose {bool} -- if `True`, display progress in stdout.

        Returns:
            Sequence[np.ndarray] -- All states computed, for all inputs.

        Note:
            If only one input sequence is provided ("continuous time" inputs), workers should be 1,
            because parallelization is impossible. In other cases, if using large NumPy arrays during
            computation (which is often the case), prefer using `threading` backend to avoid huge
            overhead. Multiprocess is a good idea only in very specific cases, and this code is not
            (yet) well suited for this.
        """
        ## Autochecks of inputs and outputs
        self._autocheck_io(inputs=inputs, outputs=teachers)

        if verbose:
            steps = np.sum([i.shape[0] for i in inputs])
            print(f"Training on {len(inputs)} inputs ({steps} steps)-- wash: {wash_nr_time_step} steps")

        # compute all states
        all_states = self.compute_all_states(inputs,
                                             forced_teachers=teachers,
                                             wash_nr_time_step=wash_nr_time_step,
                                             workers=workers,
                                             backend=backend,
                                             verbose=verbose)

        all_teachers = [t[wash_nr_time_step:].T for t in teachers]

        # compute readout matrix
        self.Wout = self.fit_readout(all_states, all_teachers, verbose=verbose, ridge=regcoef, regbias=regbias)

        # save the expected dimension of outputs
        self.dim_out = self.Wout.shape[0]

        # return all internal states
        return [st.T for st in all_states]


    def run(self,
            inputs: Sequence[np.ndarray],
            init_state: np.ndarray=None,
            init_fb: np.ndarray=None,
            workers: int=-1,
            backend: str="threading",
            verbose: bool=False) -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]:
        """Run the model on a sequence of inputs, and returned the states and
           readouts vectors.

        Arguments:
            inputs {Sequence[np.ndarray]} -- Sequence of inputs.

        Keyword Arguments:
            init_state {np.ndarray} -- State initialization vector
                                       for all inputs. (default: {None})
            init_fb {np.ndarray} -- Feedback initialization vector
                                    for all inputs, if feedback is
                                    enabled. (default: {None})
            workers {int} -- if n >= 1, will enable parallelization of
                             states computation with n threads/processes, if possible.
                             if n = -1, will use all available resources for
                             parallelization.
            backend {str} -- Backend used for parallelization of
                             states computations. Available backends are
                             `threadings`(recommended, see Note), `multiprocess`,
                             `loky` (default: {"threading"}).
            verbose {bool} -- if `True`, display progress in stdout.

        Returns:
            Tuple[Sequence[np.ndarray], Sequence[np.ndarray]] -- All states and readouts,
                                                                 for all inputs.

        Note:
            If only one input sequence is provided ("continuous time" inputs), workers should be 1,
            because parallelization is impossible. In other cases, if using large NumPy arrays during
            computation (which is often the case), prefer using `threading` backend to avoid huge
            overhead. Multiprocess is a good idea only in very specific cases, and this code is not
            (yet) well suited for this.
        """

        if verbose:
            steps = np.sum([i.shape[0] for i in inputs])
            print(f"Running on {len(inputs)} inputs ({steps} steps)")

        ## Autochecks of inputs
        self._autocheck_io(inputs=inputs)

        all_states = self.compute_all_states(inputs,
                                             init_state=init_state,
                                             init_fb=init_fb,
                                             workers=workers,
                                             backend=backend,
                                             verbose=verbose)

        all_outputs = self.compute_outputs(all_states)
        # return all_outputs, all_int_states
        return [st.T for st in all_outputs], [st.T for st in all_states]


    def run_auto(self,
            init_state: np.ndarray=None,
            steps: int=100,
            data: np.ndarray=None,
            errorthreshold: float=None,
            verbose: bool=False) -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]:
        """Run the model autonomously starting from given initial state of the reservoir.

        Arguments:
            init_state {np.ndarray} -- State initialization vector
                                       for the nodes of the reservoir. (default: {None})

        Keyword Arguments:
            steps {int} -- Number of steps which the
                           model is run for.
            verbose {bool} -- if `True`, display progress in stdout.

            - Valid time calculation -
            - Autonomous run stops when error threshold is exceeded if following arguments are given:
            data {np.ndarray} -- The real trajectory of the system to which the autonomous
                                 runprediction is compared.
            errortreshold {float} -- Threshold for error between predicted and true state.

        Returns:
            Tuple[Sequence[np.ndarray], Sequence[np.ndarray]] -- All states and readouts.
            or List[Sequence[np.ndarray], Sequence[np.ndarray], float] -- Above and validtimesteps.
        """

        if init_state is None:
            raise RuntimeError("Impossible to run autonomously without initial state of the reservoir.")
        if (data is not None) and (errorthreshold is None):
            raise RuntimeError('''To run autonomously until prediction error exceeds
                                  threshold value, the errorthreshold argument is needed.''')

        r = init_state
        states = np.zeros((steps, self.N))
        outputs = np.zeros((steps, self.dim_inp))
        states[0, :] = r[:, 0]

        thresholdmode = False
        normalisation = 1
        if data is not None and errorthreshold is not None:
            thresholdmode = True
            normalisation = np.sqrt(np.average(np.power(np.linalg.norm(data, axis=1), 2)))

        for i in range(steps):

            # output
            v = np.dot(self.Wout, self.routfunc(r)).astype(self.typefloat)
            outputs[i, :] = v[:, 0]

            # linear transformation
            x1 = np.dot(self.Win, v.reshape(self.dim_inp, 1)) + self.W.dot(r)

            # previous states memory leak and non-linear transformation
            r = ((1-self.lr)*r + self.lr*self.actfunc(x1))
            states[i, :] = r[:, 0]

            if thresholdmode:
                # compare data and prediction
                e = np.linalg.norm(data[i, :] - outputs[i, :])/ normalisation
                if e > errorthreshold:
                    break

        if thresholdmode:
            validtimesteps = i
            return outputs, states, validtimesteps

        return outputs, states

    def run_auto_parallel(self,
                            rank: int,
                            neighbours: list,
                            init_state: np.ndarray=None,
                            steps: int=100
                        ):
        '''
        Run the model autonomously starting from a given initial state of the reservoir, parallel version.
        Arguments:
            rank {int} -- rank of the process (MPI.COMM_WORLD.Get_rank())

            neighbours {list} -- list of neighbouring nodes of node_{rank}.
            
            init_state {np.ndarray} -- State initialization vector
                                       for the nodes of the reservoir. (default: {None})
            steps {int} -- Number of steps which the model is run for.
        '''
        if init_state is None:
            raise RuntimeError("Impossible to run autonomously without initial state of the reservoir.")
        r = init_state
        states = np.zeros((steps, self.N))
        outputs = np.zeros((steps, self.dim_out))
        states[0, :] = r[:, 0]

        for i in range(steps):
            #output for this node
            v = np.dot(self.Wout, self.routfunc(r)).astype(self.typefloat)
            outputs[i, :] = v[:, 0]
            #communicate the outputs
            for neighbour in neighbours:
                MPI.COMM_WORLD.send(v[:, 0], dest=neighbour)
            for neighbour in neighbours:
                neighbour_output = MPI.COMM_WORLD.recv(source=neighbour)
                v = np.append(v,neighbour_output)

            # linear transformation
            x1 = np.dot(self.Win, v.reshape(self.dim_inp, 1)) + self.W.dot(r)

            # previous states memory leak and non-linear transformation
            r = ((1-self.lr)*r + self.lr*np.tanh(x1))
            states[i, :] = r[:, 0]

        return outputs, states

    def rel_error(self,
                   testdata: np.ndarray,
                   lytime: float,
                   inputtime: float,
                   dt: float,
                   N: int=20,
                   init_state=None,
                   synchrodata: np.ndarray = None,
                   returnall: bool=False,
                   debugplots: bool=False ) -> float:
        """Calculates average relative error for short time prediction.

        Arguments:
            testdata {np.ndarray} -- The data to which the prediction of the reservoir is compared.
            lytime {float} -- systems lyapunov time
            inputtime {float} -- time used to initialize the reservoir state
                                by running the reservoir with data input
            dt{float} -- integration timestep

        Keyword Arguments:
            N {int} -- number of evenly spaced timepoints t_i within the testing data

        Returns:
            float -- Calculated short time prediction error.

        Notes:
        - First we form N evenly spaced timepoints t_i within the testing data
        - Then, before each t_i the reservoir is integrated with the data input
        - From each t_i onwards the reservoir runs autonomously and prediction error
          (tive avg. of relative error) is calculated for one lyapunov time. Lastly,
          the error is averaged over the trials (t_i's).
        """
        if synchrodata is None:
            synchrodata = testdata

        lysteps, inputsteps = round(lytime/dt), round(inputtime/dt)
        startind = inputsteps
        evaltimes = np.linspace(startind+1, testdata.shape[0]-lysteps, N, dtype=int) ## !!
        rstate = np.zeros((self.N,1))
        errors = np.zeros(N)
        i = 0

        if init_state is not None:
            # calculate first error
            outputs_pre, internal_states_pre = self.run_auto(init_state=init_state, steps=lysteps)
            e = np.average(np.linalg.norm((testdata[0:0+lysteps, :]-outputs_pre), axis=1)/np.linalg.norm(testdata[0:0+lysteps, :], axis=1))
            # save to errors
            errors[0] = e
            # remove first time from  evaltimes and increase i
            evaltimes = evaltimes[1:]
            i = i +1

        for t in evaltimes:
            # run reservoir with input from t-inputsteps to t
            inputs = synchrodata[t-inputsteps:t, :]
            output_tr, internal_tr = self.run(inputs=[inputs], init_state=rstate)

            # run auto from t to t+lysteps
            last_state = internal_tr[-1][-1:,:].T
            outputs_pre, internal_states_pre = self.run_auto(init_state=last_state, steps=lysteps)

            #debug plot
            if debugplots:
                plt.figure()
                plt.plot(testdata[t:t+lysteps, :])
                plt.plot(outputs_pre)
                plt.title(np.average(np.linalg.norm((testdata[t:t+lysteps, :]-outputs_pre), axis=1)/np.linalg.norm(testdata[t:t+lysteps, :], axis=1)))

            # calculate error
            e = np.average(np.linalg.norm((testdata[t:t+lysteps, :]-outputs_pre), axis=1)/np.linalg.norm(testdata[t:t+lysteps, :], axis=1))
            errors[i] = e
            i = i+1

        if returnall:
            return np.sqrt(1/N*np.sum(np.power(errors, 2))), errors

        return np.sqrt(1/N*np.sum(np.power(errors, 2)))


    def astp_error(self,
                   testdata: np.ndarray,
                   lytime: float,
                   inputtime: float,
                   dt: float,
                   N: int=20,
                   init_state = None,
                   synchrodata: np.ndarray = None,
                   returnall: bool=False) -> float:
        """Calculates average short time predidiction error of the reservoir.

        Arguments:
            testdata {np.ndarray} -- The data to which the prediction of the reservoir is compared.
            lytime {float} -- systems lyapunov time
            inputtime {float} -- time used to initialize the reservoir state
                                by running the reservoir with data input
            dt{float} -- integration timestep

        Keyword Arguments:
            N {int} -- number of evenly spaced timepoints t_i within the testing data

        Returns:
            float -- Calculated short time prediction error.

        Notes:
        - First we form N evenly spaced timepoints t_i within the testing data
        - Then, before each t_i the reservoir is integrated with the data input
        - From each t_i onwards the reservoir runs autonomously and prediction error
          (normalized RMSE) is calculated for one lyapunov time. Lastly, the error is averaged over
          the trials (t_i's).
        """

        if synchrodata is None:
            synchrodata = testdata

        lysteps, inputsteps = round(lytime/dt), round(inputtime/dt)
        startind = inputsteps
        evaltimes = np.linspace(startind+1, testdata.shape[0]-lysteps, N, dtype=int)  #!!
        #evaltimes = np.linspace(startind+1, testdata.shape[0]-inputsteps, N, dtype=int)
        rstate = np.zeros((self.N,1))
        errors = np.zeros(N)
        #print(evaltimes)
        i = 0

        if init_state is not None:
            # calculate first error
            outputs_pre, internal_states_pre = self.run_auto(init_state=init_state, steps=lysteps)
            e = np.sqrt(dt*1/lytime*np.sum(np.power(np.linalg.norm(testdata[0:0+lysteps, :]-outputs_pre), 2)))
            # save to errors
            errors[0] = e/np.std(testdata)
            # remove first time from  evaltimes and increase i
            evaltimes = evaltimes[1:]
            i = i +1

        for t in evaltimes:
            # run reservoir with input from t-inputsteps to t
            inputs = synchrodata[t-inputsteps:t, :]
            output_tr, internal_tr = self.run(inputs=[inputs], init_state=rstate)

            # run auto from t to t+lysteps
            last_state = internal_tr[-1][-1:,:].T
            outputs_pre, internal_states_pre = self.run_auto(init_state=last_state, steps=lysteps)

            #debug plot
            #plt.figure()
            #plt.plot(testdata[t:t+lysteps, :])
            #plt.plot(outputs_pre)
            #normalisation = np.sqrt(np.average(np.power(np.linalg.norm(testdata, axis=1), 2)))
            #print(testdata[t, :]-outputs_pre[0, :])
            #print(np.linalg.norm(testdata[t, :]-outputs_pre[0, :])/ normalisation)

            # calculate error
            e = np.sqrt(dt*1/lytime*np.sum(np.power(np.linalg.norm(testdata[t:t+lysteps, :]-outputs_pre), 2)))
            errors[i] = e/np.std(testdata)
            #errors[i] = e/np.sqrt(np.average(np.power(np.linalg.norm(testdata, axis=1), 2)))
            i = i+1

            #if init_state is not None:
            #    return np.sqrt(1/N*np.sum(np.power(errors, 2))), errors[0]
        if returnall:
            return np.sqrt(1/N*np.sum(np.power(errors, 2))), errors

        return np.sqrt(1/N*np.sum(np.power(errors, 2)))


    def validtime(self,
                  testdata: np.ndarray,
                  teststeps: int,
                  inputsteps: int,
                  dt: float,
                  N: int=10,
                  errorthr: float=0.4,
                  synchrodata: np.ndarray = None,
                  returnall: bool=False) -> float:
        '''
        Calcultes average valid time of the autonomous reservoir prediction. Valid time is
        time when prediction first exceeds given error threshold.

        Arguments:
            testdata {np.ndarray} -- The data to which the prediction of the reservoir is compared.
            teststeps {int} -- Number of datapoints the autonomous prediction maximally runs.
            inputsteps {int} -- Number of datapoints used to initialize the reservoir state
                                by running the reservoir with data input.
            dt {float} -- integration timestep

        Keyword Arguments:
            N {int} -- number of prediction trials
            errorthreshold {float} -- Threshold for the error between predicted and true state.

        Returns:
            float -- averaged valid time
        '''
        steps = testdata.shape[0] - teststeps
        test_ind = np.linspace(inputsteps+1, steps, N, dtype=int)

        validtimes = np.zeros(N)
        rstate = np.zeros((self.N,1))
        j = 0

        if synchrodata is None:
            synchrodata = testdata

        for i in test_ind:

            # run reservoir with input
            input_data = synchrodata[i-inputsteps:i, :]
            output_tr, internal_tr = self.run(inputs=[input_data], init_state=rstate)

            # run auto until error threshold is met or maximum number of steps
            last_state = internal_tr[-1][-1:,:].T
            testdataseg = testdata[i: i+teststeps, :]
            outputs_pre, internal_states_pre, validsteps = self.run_auto(init_state=last_state, steps=teststeps, data=testdataseg, errorthreshold = errorthr)

            # debug plots
#             plt.figure()
#             plt.plot(testdataseg)
#             plt.plot(outputs_pre)
#             print(testdataseg[0, :]-outputs_pre[0, :])
#             plt.figure()
#             err_norm = np.linalg.norm(testdataseg[:, :]-outputs_pre[:, :], axis=1)
#             plt.plot(err_norm)

            validtimes[j] = validsteps*dt
            j = j+1

        if returnall:
            return np.mean(validtimes), validtimes
        return np.mean(validtimes)


    def lyespectrum(self, rt, timestep, rout_=None, drout_=None, m=None):
        '''
        Calculates all Lyapunov exponents of the reservoir. There are as many of the exponents
        as reservoir has nodes. Exponents are calculated from the eigenvalues of osedelec? matrix,
        which is the product of jacobians and their transposes along the trajectory. This product
        becomes quickly too large. To cope with this, qr-decomposition is calculated iteratively.

        Arguments:
            rt {np.ndarray} -- The trajectory of the reservoir over which the lyapunow exponents
                               are calculated. Should be 1000 steps at least.
            dt {float} -- integration timestep of the reservoir

        Keyword Arguments:
            rout_ -- The output function applied to the values of reservoir before multiplying by Wout
                     Takes r(t) at single timestep and returns rout(r(t))
            drout_ -- Derivative of the output function. Takes r(t) at single timestep and
                      returns d/dr (rout(r(t)))
            m      -- Number of Lyapunov exponents to be calculated. Defaults to the number of
                      dimensions / nodes.

        Returns:
            ndarray -- Vector with the lyapunov exponents.

        NOTE: calculation takes a long time (compared to astperror or valid time),
        be prepared to give it 10 minutes at least.
        '''

        if (rout_ is None) != (drout_ is None):
            raise RuntimeError("Both rout-function and its derivative are needed.")
        if m is not None and m > rt.shape[0]:
            raise RuntimeError("The maximum number of the exponents to be calculated is the dimension of the system.")

        if m is None:
            m = rt.shape[0]

        N = rt.shape[0]
        T = rt.shape[1]
        # Q = np.identity(N)
        # log_r = np.zeros(N)
        Q0_ = np.random.random((N, m))
        Q, R = np.linalg.qr(Q0_)
        log_r = np.zeros(m)

        Wout = self.Wout
        Win = self.Win
        W = self.W
        lr = self.lr

        # help function to make sure that R is positive
        def qr_pos(Q, R):
            sgn = np.sign(np.diag(R))
            R_new = np.diag(sgn).dot(R)
            Q_new = Q.dot(np.diag(sgn))
            return Q_new, R_new

        def jacobian(rt, Win, Wout, W, lr=1):
            #Jacobian - DF() - at point rt
            J = np. zeros((N, N))
            for i in range(N):
                sech2 = lr*np.power(W[i, :]@rt + 1/np.cosh(Win[i, :]@Wout@rt), 2)
                for j in range(N):
                    J[i, j] = (1-lr)*(i==j) + sech2*Win[i, :]@Wout[:, j]
            return J

        # if rout_ != None:
        #     def jacobian(rt, Win, Wout, W, lr=1):
        #         #Jacobian - DF() - at point rt
        #         J = np. zeros((N, N))
        #         for i in range(N):
        #             rout = rout_(rt)
        #             drout = drout_(rt)
        #             sech2 = lr*np.power(1/np.cosh(W[i, :]@rt + Win[i, :]@Wout@rout), 2)
        #             for j in range(N):
        #                 J[i, j] = (1-lr)*(i==j) + sech2*(W[i,j] + Win[i, :]@Wout[:, j]*drout[j])
        #         return J

        if rout_ != None:  #v2
            def jacobian(rt, Win, Wout, W, lr=1):
                #Jacobian - DF() - at point rt
                J = np. zeros((N, N))
                rout = rout_(rt)
                drout = drout_(rt)
                for i in range(N):
                    WinWout = Win[i, :].dot(Wout)
                    sech2 = lr*np.power(1/np.cosh(W[i, :].dot(rt)+ WinWout.dot(rout)), 2)
                    J[i, :] =   sech2*(W[i,:] + np.multiply(WinWout, drout))
                    J[i, i] = J[i, i] + 1-lr
                return J

        #print('--starting to calculate jacobians--')
        for i in range(T):
            #if (T%10==0):
            #    print(T)
            #    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            DFL = jacobian(rt[:, i],Win, Wout, W, lr)
            Q, R = np.linalg.qr(DFL.dot(Q))
            Q, R = qr_pos(Q, R)
            log_r = log_r + np.log(np.diag(R))

        #print('--half point--')
        for i in range(T):
            #if (T%10==0):
            #    print(T)
            #    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            DFL_T = jacobian(rt[:, T-i-1],Win, Wout, W, lr).T
            Q, R = np.linalg.qr(DFL_T.dot(Q))
            Q, R = qr_pos(Q, R)
            log_r = log_r + np.log(np.diag(R))

        return 1/timestep*1/(2*T)*log_r


    def causal_dependence(self, trajectory, timestep):
        '''
        Calculates short term causal dependence.
        NOTE: rout(r) = r is assumed! Also leaking rate is assumed to be 1??

        Arguments:
            rt {np.ndarray} -- The trajectory of data over which the causal dependence
                               is calculated. 1000 steps is fine.
            dt {float} -- integration timestep of the reservoir

        Returns:
            ndarray -- Matrix dF_dz
        '''

        Wout = self.Wout
        Win = self.Win
        W = self.W

        steps = trajectory.shape[0]
        inv_tau = 1./timestep
        inv_Wout = linalg.pinv(Wout)
        P = W@inv_Wout + Win
        M = self.dim_inp
        R = self.N

        dH_dz = np.zeros((M,M))
        for t in range(0, steps):
            dH_dz += Wout[:, :] @ (1./pow(np.cosh(P@trajectory[t, :]), 2).reshape(R,1)*P[:,:])
        dH_dz = inv_tau/steps *dH_dz

        for j in range(0, M):  # This is supposedly taking care of the Kroenecker delta - delta_(i j).
            dH_dz[j][j] = 0

        return dH_dz


    def save(self, directory: str):
        """Save the ESN to disk.

        Arguments:
            directory {str or Path} -- Directory of the saved model.
        """
        _save(self, directory)


    def describe(self) -> Dict:
        """
        Provide descriptive stats about ESN matrices.

        Returns:
            Dict -- Descriptive data.
        """

        desc = {
            "Win": {
                "max": np.max(self.Win),
                "min": np.min(self.Win),
                "mean": np.mean(self.Win),
                "median": np.median(self.Win),
                "std": np.std(self.Win)
            },
            "W": {
                "max": np.max(self.W),
                "min": np.min(self.W),
                "mean": np.mean(self.W),
                "median": np.median(self.W),
                "std": np.std(self.W),
                "sr": max(abs(linalg.eig(self.W)[0]))
            }
        }
        if self.Wfb is not None:
            desc["Wfb"] = {
                "max": np.max(self.Wfb),
                "min": np.min(self.Wfb),
                "mean": np.mean(self.Wfb),
                "median": np.median(self.Wfb),
                "std": np.std(self.Wfb)
            }
        if self.Wout is not None:
            desc["Wout"] = {
                "max": np.max(self.Wout),
                "min": np.min(self.Wout),
                "mean": np.mean(self.Wout),
                "median": np.median(self.Wout),
                "std": np.std(self.Wout)
            }
        return desc
