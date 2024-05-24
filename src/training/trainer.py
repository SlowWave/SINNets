import os
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(
        self,
        nn_model,
        optimizer,
        loss_function,
        dynamic_system_tag,
        epochs_num=50,
        validate_model=True,
    ):
        """
        Initializes a Trainer object.

        Args:
            nn_model (torch.nn.Module): The neural network model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            loss_function (torch.nn.Module): The loss function used for training.
            dynamic_system_tag (str): The tag for the dynamic system being trained.
            epochs_num (int, optional): The number of epochs to train for. Defaults to 50.
            validate_model (bool, optional): Whether to validate the model during training. Defaults to True.
        """

        # initialize attributes
        self.nn_model = nn_model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.dynamic_system_tag = dynamic_system_tag
        self.epochs_num = epochs_num
        self.validate_model = validate_model
        self.tb_logger = None
        self.tb_logs_path = None
        self.experiment_path = None

    def train_model(self, training_dataloader, validation_dataloader):
        """
        Trains the model using the provided training and validation dataloaders.

        Args:
            training_dataloader (torch.utils.data.DataLoader): The dataloader for the training data.
            validation_dataloader (torch.utils.data.DataLoader): The dataloader for the validation data.

        Returns:
            None
        """

        # create results folders and initialize tensorboard logger
        self._create_results_folders()
        self.tb_logger = SummaryWriter(log_dir=self.tb_logs_path)

        # display start training info
        start_training_info = f"| Start training and validation of {self.nn_model.tag} model on {self.dynamic_system_tag} system |"
        line_div = "-" * len(start_training_info)

        print(line_div)
        print(start_training_info)
        print(line_div)

        for epoch in range(self.epochs_num):
            # training loop
            current_training_loss = self._train_single_epoch(
                epoch, training_dataloader=training_dataloader
            )

            # validation loop
            current_validation_loss = self._validate_single_epoch(
                epoch, validation_dataloader=validation_dataloader
            )

            # display epoch info
            epoch_info = f"- Epoch: {epoch + 1} | Training loss: {round(current_training_loss, 5)} | Validation loss: {round(current_validation_loss, 5)}"
            print(epoch_info)

    def save_nn_model(self):
        """
        Save the neural network model to a file.

        Args:
            None

        Returns:
            None
        """

        # torch.save(self.nn_model.state_dict(), self.experiment_path)
        torch.save(self.nn_model, os.path.join(self.experiment_path, "nn_model.zip"))

    def _create_results_folders(self):
        """
        Creates the necessary folders for storing the results of the training process.

        Args:
            None

        Returns:
            None
        """

        # find "/results" folder
        root_dir = os.getcwd()
        results_path = False
        while not results_path:
            for root, dirs, _ in os.walk(root_dir):
                if "results" in dirs:
                    results_path = os.path.join(root, "results")
                    break

            root_dir = os.path.join(root_dir, os.pardir)

        # create results folder associated with the dynamic system under test
        dynamic_system_results_path = os.path.join(
            results_path, self.dynamic_system_tag
        )
        if not os.path.exists(dynamic_system_results_path):
            os.makedirs(dynamic_system_results_path)

        # create current experiment folder
        experiment_time = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
        experiment_folder = self.nn_model.tag + "_" + experiment_time
        self.experiment_path = os.path.join(
            dynamic_system_results_path, experiment_folder
        )
        os.makedirs(self.experiment_path)

        # create tensorboard log folder
        self.tb_logs_path = os.path.join(self.experiment_path, "tb_logs")
        os.makedirs(self.tb_logs_path)

    def _train_single_epoch(self, epoch_idx, training_dataloader):
        """
        Trains the model for a single epoch using the provided validation dataloader.

        Args:
            epoch_idx (int): The index of the current epoch.
            training_dataloader (torch.utils.data.DataLoader): The dataloader for the training data.

        Returns:
            float: The average training loss for the epoch.
        """

        # enable model training mode
        self.nn_model.train(True)

        # initialize loss value
        training_loss = 0.0

        # iterate over batch samples
        for idx, (inputs, targets) in enumerate(training_dataloader):

            # clear the gradients
            self.optimizer.zero_grad()

            # forward propagation
            outputs = self.nn_model(inputs)

            # compute loss value
            loss = self.loss_function(outputs, targets)
            training_loss += loss.item()

            # backward propagation
            loss.backward()

            # update model parameters
            self.optimizer.step()

        # compute average loss value
        average_loss = training_loss / len(training_dataloader)
        
        # update tensorboard data
        current_iteration = epoch_idx * len(training_dataloader) + idx + 1
        self.tb_logger.add_scalar(
            "Loss/Training",
            average_loss,
            current_iteration
        )        

        return average_loss

    def _validate_single_epoch(self, epoch_idx, validation_dataloader):
        """
        Validates the model on a single epoch using the provided validation dataloader.

        Args:
            epoch_idx (int): The index of the current epoch.
            validation_dataloader (torch.utils.data.DataLoader): The dataloader for the validation data.

        Returns:
            float: The average validation loss for the epoch.
        """

        # enable model validation mode
        self.nn_model.eval()

        # initialize loss value
        validation_loss = 0.0

        with torch.no_grad():

            # iterate over batch samples
            for idx, (inputs, targets) in enumerate(validation_dataloader):

                # forward propagation
                outputs = self.nn_model(inputs)

                # compute loss value
                loss = self.loss_function(outputs, targets)
                validation_loss += loss.item()

            # compute average loss
            average_loss = validation_loss / len(validation_dataloader)
            
            # update tensorboard data
            current_iteration = epoch_idx * len(validation_dataloader) + idx + 1
            self.tb_logger.add_scalar(
                "Loss/Validation",
                average_loss,
                current_iteration
            )

            return average_loss
