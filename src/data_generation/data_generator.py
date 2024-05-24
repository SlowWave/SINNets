import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


class DataGenerator:
    def __init__(self, dynamic_system, integration_step, time_horizon):
        """
        Initialize a DataGenerator object.

        Args:
            dynamic_system (DynamicSystem): The dynamic system object.
            integration_step (float): The time step for integration.
            time_horizon (float): The duration of the simulation.
        """

        # initialize attributes
        self.dynamic_system = dynamic_system
        self.integration_step = integration_step
        self.time_horizon = time_horizon

    def generate_dataset(
        self,
        obs_num,
        obs_window,
        initial_state=None,
        use_inputs=False,
        inputs_shape="random",
        batch_size=64,
        shuffle=True,
        verbose=True,
    ):
        """
        Generates a dataset for training or validation.

        Args:
            obs_num (int): The number of observations to generate.
            obs_window (int): The size of the observation window.
            initial_state (list, optional): The initial state of the system. If None, a random initial state is generated. Defaults to None.
            use_inputs (bool, optional): Flag indicating whether to use inputs or not. Defaults to False.
            inputs_shape (str, optional): The shape of the inputs. If "random", generates random inputs. Defaults to "random".
            batch_size (int, optional): The batch size. Defaults to 64.
            shuffle (bool, optional): Flag indicating whether to shuffle the data. Defaults to True.
            verbose (bool, optional): Flag indicating whether to print progress. Defaults to True.

        Returns:
            torch.utils.data.DataLoader: The data loader for the dataset.
        """

        data_len = 0
        simulation_cnt = 0
        raw_data_container = list()

        if verbose:
            print("Dataset generation started...")
            print("- Simulation process started...")

        # generate new data
        while True:

            # propagate system states
            raw_data_dict = self.dynamic_system.propagate_states(
                time_horizon=self.time_horizon,
                integration_step=self.integration_step,
                initial_state=initial_state,
                use_inputs=use_inputs,
                inputs_shape=inputs_shape,
            )
            raw_data_container.append(raw_data_dict)
            
            simulation_cnt += 1
            
            # check generated data length
            data_len += len(raw_data_dict["time_steps"])
            if data_len - (obs_window) * len(raw_data_container) >= obs_num:
                break

        if verbose:
            print(f"- Simulation process completed. Total simulations: {simulation_cnt}")
            print("- Data aggregation started...")

        # reshape data
        data_container = dict(x=list(), y=list())
        for raw_data_dict in raw_data_container:

            inputs = np.array(raw_data_dict["inputs"])
            states = np.array(raw_data_dict["states"])
            single_obs = list()

            for i in range(raw_data_dict["time_steps"].shape[0] - 1):

                if use_inputs:
                    input = [el[i] for el in inputs]
                else:
                    input = []

                state = [el[i] for el in states]
                single_obs.append(input + state)

                if i >= obs_window - 1:
                    data_container["x"].append(np.copy(single_obs))
                    data_container["y"].append([el[i + 1] for el in states])
                    single_obs.pop(0)

        # delete leftover data
        data_container["x"] = data_container["x"][:obs_num]
        data_container["y"] = data_container["y"][:obs_num]

        # build tensor dataset
        data_container["x"] = torch.tensor(
            np.array(data_container["x"]),
            dtype=torch.float32
        )
        data_container["y"] = torch.tensor(
            np.array(data_container["y"]),
            dtype=torch.float32
        )
        dataset = TensorDataset(data_container["x"], data_container["y"])

        data_loader = DataLoader(dataset, batch_size, shuffle)

        if verbose:
            print("- Data aggregation completed.")
            print("Dataset generation completed.")

        return data_loader

    def generate_timeseries(
        self,
        initial_state=None,
        use_inputs=False,
        inputs_shape="random",
    ):
        """
        Generates a timeseries dictionary by propagating system states.

        Args:
            initial_state (list, optional): The initial state of the system. If None, a random initial state is generated. Defaults to None.
            use_inputs (bool, optional): Flag indicating whether to use inputs or not. Defaults to False.
            inputs_shape (str, optional): The shape of the inputs. If "random", generates random inputs. Defaults to "random".

        Returns:
            dict: A dictionary containing the timeseries data with the following keys:
                - time_steps (numpy.ndarray): An array of time steps.
                - inputs (numpy.ndarray): An array of input signals.
                - states (numpy.ndarray): An array of system states.
        """

        # propagate system states
        timeseries_dict = self.dynamic_system.propagate_states(
            time_horizon=self.time_horizon,
            integration_step=self.integration_step,
            initial_state=initial_state,
            use_inputs=use_inputs,
            inputs_shape=inputs_shape,
        )

        # convert data to numpy arrays
        timeseries_dict["time_steps"] = np.array(timeseries_dict["time_steps"])
        timeseries_dict["inputs"] = np.array(timeseries_dict["inputs"])
        timeseries_dict["states"] = np.array(timeseries_dict["states"])

        return timeseries_dict
