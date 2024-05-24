import torch
import torch.nn as nn


class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        """
        Initializes a SimpleGRU object.

        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of hidden units in the recurrent layer.
            output_size (int): The number of output features.
            num_layers (int): The number of recurrent layers.
        """

        super(SimpleGRU, self).__init__()

        # initialize attributes
        self.tag = "SimpleGRU"
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # define recurrent layer
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # define linear layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Performs the forward pass of the neural network.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, output_size).
        """

        # initialize hidden state
        hidden_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # forward propagation
        x, _ = self.rnn(x, hidden_state)

        # decode hidden state of the last time step
        y = self.fc(x[:, -1, :])

        return y
