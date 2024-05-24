import math
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


class DynamicSystem:
    def __init__(self):
        """
        Base class for all other dynamic systems classes.

        Args:
        None
        """

        # initialize attributes
        self.state_boundaries = None
        self.input_boundaries = None
        self.state_dim = None
        self.input_dim = None

    def set_initial_state(self, state):
        """
        Sets the initial state of the dynamic system.

        Args:
            state (list or None): The initial state of the system. If None, a random initial state is generated.

        Returns:
            list: The initial state of the system.
        """

        if not state:
            initial_state = list()
            for idx in range(self.state_dim):
                initial_state.append(
                    random.uniform(
                        self.state_boundaries[0][idx],
                        self.state_boundaries[1][idx],
                    )
                )

        else:
            initial_state = state

        return initial_state

    def generate_input_signals(self, use_inputs, inputs_shape, inputs_num):
        """
        Generates input signals for the dynamic system.

        Args:
            use_inputs (bool): Flag indicating whether to use inputs or not.
            inputs_shape (str): Shape of the inputs. If 'random', generates random inputs.
            inputs_num (int): Number of inputs to generate.

        Returns:
            list: List of input signals. Each input signal is a numpy array of shape (inputs_num,).
        """

        input_signals = list()

        if not use_inputs:
            for _ in range(self.input_dim):
                input_signals.append(np.zeros(inputs_num))

        elif inputs_shape == "random":
            for idx in range(self.input_dim):
                input_signals.append(
                    np.random.uniform(
                        self.input_boundaries[0][idx],
                        self.input_boundaries[1][idx],
                        inputs_num,
                    )
                )

        return input_signals

    def ode(self):
        pass

    def simulate_system(
        self,
        time_horizon,
        integration_step,
        initial_state=None,
        use_inputs=False,
        inputs_shape="random",
    ):
        """
        Simulates the dynamics of the dynamic system over a given time horizon and plots the results.

        Args:
            time_horizon (float): The duration of the simulation in seconds.
            integration_step (float): The time step for the integration in seconds.
            initial_state (list, optional): The initial state of the system. If None, a random initial state is generated. Defaults to None.
            use_inputs (bool, optional): Flag indicating whether to use inputs or not. Defaults to False.
            inputs_shape (str, optional): The shape of the inputs. If 'random', generates random inputs. Defaults to "random".
        """

        # propagate system dynamics
        data_dict = self.propagate_states(
            time_horizon,
            integration_step,
            initial_state,
            use_inputs,
            inputs_shape,
        )

        fig, axes = plt.subplots(len(data_dict["states"]) + (len(data_dict["inputs"]) if use_inputs else 0))

        for i, state in enumerate(data_dict["states"]):
            axes[i].plot(
                data_dict["time_steps"],
                state,
                label=f"$x_{i}$",
                color=f"C{i}",
            )
            axes[i].grid()
            axes[i].set_xlabel("Time [s]")
            axes[i].set_ylabel(f"$x_{i}$")
        if use_inputs:
            for j, input_signal in enumerate(data_dict["inputs"]):
                axes[len(data_dict["states"]) + j].plot(
                    data_dict["time_steps"],
                    input_signal,
                    label=f"$u_{j}$",
                    color=f"C{j+i+1}",
                )
                axes[len(data_dict["states"]) + j].grid()
                axes[len(data_dict["states"]) + j].set_xlabel("Time [s]")
                axes[len(data_dict["states"]) + j].set_ylabel(f"$u_{j}$")
        fig.tight_layout()
        fig.legend()
        plt.show()

    def propagate_states(
        self,
        time_horizon,
        integration_step,
        initial_state,
        use_inputs,
        inputs_shape,
    ):
        """
        Propagates the states of the dynamic system over a given time horizon.

        Args:
            time_horizon (float): The duration of the simulation in seconds.
            integration_step (float): The time step for the integration in seconds.
            initial_state (list): The initial state of the system.
            use_inputs (bool): Flag indicating whether to use inputs or not.
            inputs_shape (str): The shape of the inputs.

        Returns:
            dict: A dictionary containing the time steps, states, and inputs.
                - time_steps (ndarray): An array of time steps.
                - states (list): A list of state arrays.
                - inputs (list): A list of input arrays.
        """

        # define timing parameters
        num_steps = math.ceil(time_horizon / integration_step)
        time_horizon = num_steps * integration_step
        time_steps = np.linspace(0.0, time_horizon, num_steps + 1)

        # set dynamic system states and inputs
        current_state = self.set_initial_state(initial_state)
        states = [[state] for state in current_state]
        inputs = self.generate_input_signals(use_inputs, inputs_shape, num_steps + 1)

        # propagate system states
        for step in range(len(time_steps) - 1):
            # integration step
            ode_solution = solve_ivp(
                fun=self.ode,
                t_span=(time_steps[step], time_steps[step + 1]),
                y0=current_state,
                method="RK45",
                dense_output=False,
                args=([u[step] for u in inputs],),
            )

            # update current state
            current_state = [state[-1] for state in ode_solution.y]

            # update states list
            for idx, state in enumerate(states):
                state.append(current_state[idx])

        data_dict = dict(time_steps=time_steps, states=states, inputs=inputs)

        return data_dict
