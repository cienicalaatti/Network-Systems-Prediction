import numpy as np
from . import ESN
from typing import Optional, Callable, Tuple
import matplotlib.pyplot as plt


class HybridModel():
    def __init__(
            self,
            W: np.ndarray,
            Win: np.ndarray,
            knowledge_model: Optional[Callable[[np.ndarray], np.ndarray]],
            #different_W_in=False,
            integration_reset_func: Optional[Callable] = None,
            dimension_count: int = 3,
            D_r: int = 100,
            leak_rate: float = 0.3,
            lambda_reg: float = 0.000005,
            normalization_function: Optional[Callable] = None,
            inverse_norm: Optional[Callable] = None,
            km_output: Optional[int] = None
            ) -> None:
        '''
        An instance of the hybrid model.

        Arguments:
            knowledge_model (Callable): The integrator for the knowledge model. Takes an np.ndarray and returns np.ndarray.
            constant_object (dict or None): The constants used for generating W_in and reservoir matrices and also used in computing the states of the system.
            integration_reset_func (Callable or None): The function used to reset integration global variables, defaults to None
        '''

        self.model = knowledge_model
        self.reset_func = integration_reset_func
        self.dimension_count = dimension_count
        self.D_r = D_r
        self.leak_rate = leak_rate
        self.lambda_reg = lambda_reg
        self.W = W
        self.Win = Win
        self.normalization = normalization_function if normalization_function is not None else lambda x: x
        self.inverse_norm = inverse_norm if inverse_norm is not None else lambda x: x
        self.reservoir = ESN(lr=self.leak_rate, W=self.W, Win=self.Win, input_bias=False,
                             ridge=None, Wfb=None, fbfunc=None, routfunc=lambda x: x)
        
        self.km_output = km_output
        if km_output is None:
            self.km_output = self.dimension_count

        def rout(r):
            r_out = r.copy()
            r_out[:, 1::2] = np.power(r_out[:, 1::2], 2)
            return r_out

        self.rout = rout

    def _ridge_regression(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        '''
        Internal function used to compute a regularized linear regression.

        Arguments:
            X: The input datapoints used for linear regression. Should be a matrix of size n x d where n is the number of datapoints and d the number of dimensions.
            Y: The output values used for regression. Should be of the form n x 1.

        Returns a matrix of size d x d_out, where d_out is the number of dimensions in the system.
        '''

        regularization = self.lambda_reg
        moment_matrix = X.T @ X
        regularization_term = regularization * np.eye(moment_matrix.shape[0])
        inverse = np.linalg.inv(moment_matrix + regularization_term) @ X.T
        return inverse @ Y

    def train(self, inputs: np.ndarray, outputs: np.ndarray, wash_steps: int = 0) -> np.ndarray:
        '''
        Main training function, used to set W_out matrix using linear regression.

        Arguments:
            inputs (np.ndarray): Input data in the form of N x d, where N is the number of datapoints and d the number of system dimensions.
            outputs (np.ndarray): Output data used in training. The format is N x d.
            wash_steps (int): the number of steps forgotten at the start of training. Defaults to 0.

        Returns a matrix of size N x d which contains the predictions made by the generated W_out when considering the training input data, takes wash_steps into account.
        '''

        predictions, raw_states = self._calculate_states(inputs)
        state_star_data = self.rout(raw_states)
        combined_pred_star = np.concatenate([predictions, state_star_data], axis=1)

        # Linear regression
        X = combined_pred_star[wash_steps:]
        Y = outputs[wash_steps:]

        Wout = self._ridge_regression(X, Y)
        self.Wout = Wout
        self.final_train_state = raw_states[-1]
        return X @ self.Wout

    def _get_next_state(self, u, r) -> tuple([np.ndarray, np.ndarray]):
        '''
        Function to integrate the hybrid model by one step.
        Input: state in the system space (u), reservoir's internal state (r)
        Returns: u(t+dt), r(t+dt)
        '''
        # get knowledge model prediction
        unnormalized_u = self.inverse_norm(u)
        unnormalized_K_u = self.model(unnormalized_u)
        K_u = self.normalization(unnormalized_K_u)

        # Combine K[u(t)] and u(t)
        combined_input = np.concatenate([K_u, u])

        # get next reservoir state
        r_next = self.reservoir._get_next_state(
                        single_input=combined_input,
                        last_state=r.reshape(self.D_r, 1))
        # apply rout
        r_star = self.rout(r_next.T).flatten()

        # combine knowledge model prediction and rout(r(t))
        combined_pred = np.concatenate([K_u, r_star])
        # output by multiplying with Wout
        u_next = combined_pred @ self.Wout
        return (u_next, r_next.flatten())

    def _calculate_states(self, inputs: np.ndarray, initial_state: Optional[np.ndarray] = None, wash_steps: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Internal function for calculating the states of the hybrid reservoir.

        Arguments:
            inputs (np.ndarray): The input datapoints in a N x d matrix where N is the number of points and d the number of system dimensions.
            initial_state (Optional[np.ndarray]): The initial state of the reservoir, defaults to None.
            wash_steps (int): the number of ignored steps at the beginning. Defaults to 0.

        Returns a tuple which contains the knowledge model predictions and the reservoir states.
        '''
        # Calculate the predictions
        raw_inputs = self.inverse_norm(inputs)

        predictions = np.zeros((raw_inputs.shape[0], self.km_output))
        for i in range(raw_inputs.shape[0]):
            predictions[i, :] = self.model(raw_inputs[i])

        normalized_predictions = self.normalization(predictions)

        # Combine K[u(t)] and u(t)
        combined_input = np.concatenate([normalized_predictions, inputs], axis=1)

        # Flipped columns and rows?
        raw_states = self.reservoir.compute_all_states(
            inputs=[combined_input], init_state=initial_state, wash_nr_time_step=wash_steps)[0].T
        return (normalized_predictions, raw_states)

    def predict(self,
                initial_state: np.ndarray,
                initial_position: np.ndarray,
                N_steps=1000,
                threshold: Optional[float] = None,
                testdata: Optional[np.ndarray] = None,
                stop_when_invalid: bool = False,
                ) -> np.ndarray:
        '''
        Function for predicting the next N_steps of the series.

        Arguments:
            initial_state (np.ndarray): The initial reservoir state.
            initial_position (np.ndarray): The initial prediction position.
            N_steps (int): How many steps the prediction should last, defaults to 1000.
            threshold (Optional[float]): The threshold related to valid time, defaults to None which means the function does not consider valid time at all.
            correct_data (Optional[np.ndarray]): The data which is used to calculate valid time. Defaults to None.
            stop_when_invalid (bool): The flag to stop prediction if the prediction differs from correct by more than threshold. Defaults to False.

        Returns either just the predictions or a tuple containing the predictions and the number of valid timesteps.
        '''
        # Remember to reset the integration
        if self.reset_func is not None:
            self.reset_func()

        current_position = initial_position
        current_state = initial_state
        predicted_trajectory = np.zeros((N_steps, self.dimension_count))
        valid_steps = 0
        valid = True
        error = 0
        i = 0

        normalisation = 1
        if testdata is not None:
            normalisation = np.sqrt(np.average(np.power(np.linalg.norm(testdata, axis=1), 2)))

        should_stop = False
        while should_stop is False:
            current_position, current_state = self._get_next_state(current_position, current_state)
            predicted_trajectory[i, :] = current_position
            if threshold is not None and valid is True:
                diff = testdata[i, :] - predicted_trajectory[i, :]
                error_norm = np.linalg.norm(diff)
                # Normalize according to the hybrid paper:
                # normalization = np.linalg.norm(data[i, :])
                error = error_norm / normalisation
                if error < threshold:
                    valid_steps = valid_steps + 1
                else:
                    valid = False

            i = i + 1
            if stop_when_invalid is True:
                should_stop = (valid is False or i >= N_steps)
            else:
                should_stop = (i >= N_steps)

        if threshold is not None:
            return predicted_trajectory, valid_steps
        else:
            return predicted_trajectory

    def astp_error(self, testdata: np.ndarray, lytime: float, inputtime: float, N: int = 20, dt: float = 0.01) -> np.float:
        '''
        Average short term prediction error function. Computes the ASTP error from multiple (N) points using testdata.

        Arguments:
            testdata (np.ndarray): Testing data of shape N x d.
            lytime (float): The lyapynov time of the system (for lorenz around 0.91). This is the time taken to predict the system.
            inputtime (float): The time taken to integrate the system before trying to predict it.
            N (int): The number of prediction trials, defaults to 20.
            dt (float): The timestep of the system integration, defaults to 0.01.

        Returns a float corresponding to the mean squared prediction error.
        '''

        lysteps, inputsteps = round(lytime/dt), round(inputtime/dt)
        evaltimes = np.linspace(inputsteps, testdata.shape[0]-lysteps-1, N, dtype=int)
        errors = np.zeros(N)

        for i, t in enumerate(evaltimes):
            # run reservoir with input from t-inputsteps to t
            inputs = testdata[t-inputsteps:t, :]
            _, internal_tr = self._calculate_states(inputs)

            # run auto from t to t+lysteps
            last_state = internal_tr[-1:, :].T
            outputs_pre = self.predict(last_state, testdata[t], N_steps=lysteps)

#             #debug plots
#             plt.figure()
#             plt.plot(testdata[t+1:t+1+lysteps, :])
#             plt.plot(outputs_pre[:, :])
#             print(testdata[t+1, :]-outputs_pre[0, :])
#             plt.figure()
#             err_norm = np.linalg.norm(testdata[t+1:t+1+lysteps, :]-outputs_pre[:, :], axis=1)
#             plt.plot(err_norm)

            # calculate error
            e = np.sqrt(dt*1/lytime*np.sum(np.power(np.linalg.norm(testdata[t+1:t+1+lysteps, :]-outputs_pre), 2)))
            errors[i] = e/np.std(testdata)
        return np.sqrt(1/N*np.sum(np.power(errors, 2)))

    def validtime(self,
                  testdata: np.ndarray,
                  teststeps: int,
                  inputsteps: int,
                  dt: float,
                  N: int = 10,
                  errorthr: float = 0.4,
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
        steps = testdata.shape[0] - teststeps - 1
        test_ind = np.linspace(inputsteps+1, steps, N, dtype=int)

        validtimes = np.zeros(N)
        j = 0

        for i in test_ind:
            # run reservoir with input
            input_data = testdata[i-inputsteps:i, :]
            _, internal_tr = self._calculate_states(input_data)

            # run auto from t to t+lysteps
            last_state = internal_tr[-1:, :].T

            # run auto until error threshold is met or maximum number of steps
            testdataseg = testdata[i+1: i+1+teststeps, :]
            outputs_pre, validsteps = self.predict(last_state, testdata[i, :], N_steps=teststeps, threshold=errorthr,
                                                   testdata=testdataseg, stop_when_invalid=True)
#             # debug plots
#             plt.figure()
#             plt.plot(testdataseg[:200, :])
#             plt.plot(outputs_pre[:200, :])
#             print(testdataseg[0, :]-outputs_pre[0, :])
#             plt.figure()
#             err_norm = np.linalg.norm(testdataseg[:, :]-outputs_pre[:, :], axis=1)
#             #print(outputs_pre)
#             plt.plot(err_norm)
#             print(validsteps)
            validtimes[j] = validsteps*dt
            j = j+1
        
        if returnall:
            return np.mean(validtimes), validtimes
        return validtimes
