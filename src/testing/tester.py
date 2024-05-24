import os
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import torch


class Tester:
    def __init__(self, time_series, obs_window, nn_model=None):
        """
        Initializes a Tester object.

        Args:
            time_series (dict): A dictionary containing the time series data.
            obs_window (int): The size of the observation window.
            nn_model (nn.Module, optional): The neural network model to be used for prediction. Defaults to None.
        """

        # initialize attributes
        self.time_series = time_series
        self.obs_window = obs_window
        self.nn_model = nn_model
        self.predicted_states = None

    def load_model(self):
        """
        Load a neural network model from a file.

        Args:
            None

        Returns:
            None
        """

        nn_model_path = filedialog.askopenfilename(initialdir=os.getcwd())
        self.nn_model = torch.load(nn_model_path)

    def test_model(self, use_inputs=False):
        """
        Test the model by predicting states using the model forward propagation.

        Args:
            use_inputs (bool, optional): Whether to use inputs for prediction. Defaults to False.

        Returns:
            None
        """

        # initialize predicted states array
        self.predicted_states = np.empty(
            shape=(len(self.time_series["states"]), 0),
            dtype=np.float32,
        )

        # get initial input tensor
        observation_tensor = self._get_initial_observation(use_inputs)

        # set nn_model to evaluation mode
        self.nn_model.eval()

        for _ in range(len(self.time_series["time_steps"]) - self.obs_window):
            # predict next state using model forward propagation
            predicted_state_tensor = self.nn_model.forward(observation_tensor)
            predicted_state_array = predicted_state_tensor.data.cpu().numpy()

            # update array of predicted states
            self.predicted_states = np.concatenate(
                (
                    self.predicted_states,
                    np.expand_dims(predicted_state_array[0], axis=1),
                ),
                axis=1,
                dtype=np.float32,
            )

            # queue input tensor
            observation_tensor = self._queue_observations(
                observation_tensor=observation_tensor,
                predicted_state=predicted_state_array,
            )

    def _get_initial_observation(self, use_inputs):
        """
        Get the initial observation tensor for the model.

        Args:
            use_inputs (bool): Whether to use inputs for the observation.

        Returns:
            torch.Tensor: The initial observation tensor of shape (1, obs_window * (input_dim + state_dim)).
        """

        observation_list = list()

        for idx in range(self.obs_window):
            # get single input
            if use_inputs:
                input = [el[idx] for el in self.time_series["inputs"]]
            else:
                input = []

            # get single state
            state = [el[idx] for el in self.time_series["states"]]

            # aggregate observations
            observation_list.append(input + state)

        observation_tensor = torch.tensor([observation_list], dtype=torch.float32)

        return observation_tensor

    def _queue_observations(self, observation_tensor, predicted_state):
        """
        Queues the given observation tensor by removing the first row and concatenating it with the predicted state.

        Args:
            observation_tensor (torch.Tensor): The observation tensor to be queued.
            predicted_state (numpy.ndarray): The predicted state to be concatenated with the observation tensor.

        Returns:
            torch.Tensor: The queued observation tensor.
        """

        observation_array = observation_tensor.data.cpu().numpy()[0]
        observation_array = np.delete(observation_array, 0, axis=0)
        observation_array = np.concatenate((observation_array, predicted_state), axis=0)

        observation_tensor = torch.tensor(
            np.array([observation_array]), dtype=torch.float32
        )

        return observation_tensor

    def plot_results(self):
        """
        Plot the results of the prediction.

        Args:
            None

        Returns:
            None
        """

        # compute prediction error between actual and predicted states
        prediction_error = list()

        for idx in range(len(self.time_series["states"])):
            prediction_error.append(
                np.sqrt(
                    np.square(
                        np.subtract(
                            self.time_series["states"][idx][self.obs_window :],
                            self.predicted_states[idx],
                        )
                    )
                )
            )

        fig, axes = plt.subplots(
            len(self.time_series["states"]),
            2,
            figsize=(12, 3 * len(self.time_series["states"])),
        )

        for idx, state in enumerate(self.time_series["states"]):
            axes[idx, 0].plot(
                self.time_series["time_steps"],
                state,
                label=f"$x_{idx}$",
                linestyle="-",
                color="C0",
            )
            axes[idx, 0].plot(
                self.time_series["time_steps"][self.obs_window :],
                self.predicted_states[idx],
                label=f"$\hat{{x}}_{idx}$",
                linestyle="--",
                color="C1",
            )
            axes[idx, 0].grid()
            axes[idx, 0].set_xlabel("Time [s]")
            axes[idx, 0].set_ylabel(f"$x_{idx}$, $\hat{{x}}_{idx}$")
            axes[idx, 0].legend(loc="upper right")
            
            axes[idx, 1].plot(
                self.time_series["time_steps"][self.obs_window :],
                prediction_error[idx],
                label=f"||$x_{idx}$ - $\hat{{x}}_{idx}$||",
                linestyle="-",
                color="C2",
            )
            axes[idx, 1].grid()
            axes[idx, 1].set_xlabel("Time [s]")
            axes[idx, 1].set_ylabel(f"||$x_{idx}$ - $\hat{{x}}_{idx}$||")
            axes[idx, 1].legend(loc="upper right")

        fig.tight_layout()
        plt.show()

