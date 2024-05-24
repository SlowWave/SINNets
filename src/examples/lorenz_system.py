import os
import sys

import torch
import torch.optim as optim

# add parent directory to "sys.path" to import modules from that path
sys.path.append(os.path.normpath(os.path.dirname(__file__) + os.sep + os.pardir))

from data_generation.data_generator import DataGenerator
from dynamic_systems.lorenz_system import LorenzSystem
from nn_models.simple_rnn import SimpleRNN
from nn_models.simple_lstm import SimpleLSTM
from nn_models.simple_gru import SimpleGRU
from testing.tester import Tester
from training.trainer import Trainer


# define dynamic system
dynamic_system_model = LorenzSystem(
    sigma=10,
    rho=28,
    beta=8/3,
)

# define timing parameters
time_horizon = 15
integration_step = 0.01

# simulate system
dynamic_system_model.simulate_system(
    time_horizon=time_horizon,
    integration_step=integration_step,
    initial_state=None,
)

# define data generator object
data_generator = DataGenerator(
    dynamic_system=dynamic_system_model,
    time_horizon=time_horizon,
    integration_step=integration_step,
)

# define the observation window
observation_window = 50

# generate training dataset
print("\nGenerating training dataset...")
training_data = data_generator.generate_dataset(
    obs_num=500000,
    obs_window=observation_window,
    batch_size=128,
    shuffle=True,
)

# generate validation dataset
print("\nGenerating validation dataset...")
validation_data = data_generator.generate_dataset(
    obs_num=50000,
    obs_window=observation_window,
    batch_size=128,
    shuffle=True,
)

# define RNN parameters
input_size = dynamic_system_model.state_dim
output_size = input_size

# define NN model (uncomment the model you want to use)
# nn_model = SimpleRNN(input_size=input_size, hidden_size=64, output_size=output_size, num_layers=1)
nn_model = SimpleLSTM(input_size=input_size, hidden_size=64, output_size=output_size, num_layers=1)
# nn_model = SimpleGRU(input_size=input_size, hidden_size=64, output_size=output_size, num_layers=1)

# define optimizer and loss function
optimizer = optim.Adam(nn_model.parameters(), lr=0.0001, weight_decay=1e-5)
loss_function = torch.nn.MSELoss()

# define number of training epochs
epochs_num = 30

# define trainer object
trainer = Trainer(
    nn_model=nn_model,
    optimizer=optimizer,
    loss_function=loss_function,
    dynamic_system_tag=dynamic_system_model.tag,
    epochs_num=epochs_num,
    validate_model=True,
)

# train NN model
trainer.train_model(
    training_dataloader=training_data,
    validation_dataloader=validation_data,
)

# save NN model
trainer.save_nn_model()

# generate new time series data
time_series = data_generator.generate_timeseries(initial_state=None)

# define tester object
tester = Tester(time_series=time_series, obs_window=observation_window, nn_model=nn_model)

# uncomment the following line to load saved model
# tester.load_model()

# test trained NN model on new time series data
tester.test_model()
tester.plot_results()
