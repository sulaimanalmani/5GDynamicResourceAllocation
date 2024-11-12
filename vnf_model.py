import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_generator import DataGenerator
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VNF_Model(nn.Module):
    """
    A PyTorch neural network model for Throughput / Packet loss prediction.

    This model is designed to predict both the mean and standard deviation of the output,
    allowing for regression with uncertainty estimation. It consists of an input layer,
    multiple hidden layers, and separate output layers for mean and standard deviation.

    Args:
        vnf_typ (str): The type of VNF being modeled (e.g., 'upf', 'ovs', 'ran')
        n_inputs (int): The number of input features.
        n_hidden (list of int): A list specifying the number of neurons in each hidden layer.
        n_outputs (int): The number of output features.

    Methods:
        forward(x): Performs a forward pass through the network.
        log_prob_loss(x, y): Computes the negative log probability loss.
        load_weights(path): Loads model weights from a specified path.
        predict(x, mean_val=True): Makes predictions using the model.
        train(data_gen, batch_size=128, num_epochs=5000, learning_rate=0.0001, decay=0.00, save_model=True, save_loss=True):
            Trains the model using the provided data generator.
        save_loss(): Saves the training, validation, and test loss history to CSV files.
        save_model(): Saves the model's state dictionary to a file.
    """

    def __init__(self, vnf_typ, n_inputs, n_hidden, n_outputs):
        super().__init__()
        
        # Basic model properties ####################################################################################
        self.vnf_typ = vnf_typ
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.loss = self.log_prob_loss  # Negative log probability loss function

        # Model Architecture ####################################################################################
        
        # Input Layer
        self.fc = nn.Linear(n_inputs-1, n_hidden[0])
        
        # Hidden Layers (multiple layers based on n_hidden list)
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(n_hidden[i], n_hidden[i+1]) for i in range(len(n_hidden)-1)]
        )
        
        # Output Layers: Mean and Standard Deviation for regression with uncertainty
        self.fc_mean = nn.Linear(n_hidden[-1], n_outputs)  # Output layer for mean prediction
        self.fc_std = nn.Linear(n_hidden[-1], n_outputs)   # Output layer for standard deviation

        # Loss tracking ############################################################################################
        self.loss_array = []
        self.val_loss_array = []
        self.test_loss_array = []

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Mean and standard deviation predictions.
        """
        # # Drop irrelevant column (e.g., packet size) from input
        x = x[:, 1:]
        
        # Pass through Input Layer
        x = self.fc(x)
        x = nn.ReLU()(x)
        
        # Pass through Hidden Layers with ReLU activations
        for i, hidden_layer in enumerate(self.fc_hidden):
            x = hidden_layer(x)
            x = nn.ReLU()(x)
        
        # Output mean and std predictions
        mean = self.fc_mean(x)  # Mean output
        # log_std = self.fc_std(x) # Positive std output
        # std = torch.exp(log_std)
        std = nn.Softplus()(self.fc_std(x))
        return mean, std  # Return mean and std for uncertainty estimation

    # Negative Log Probability Loss Function #####################################################################
    def log_prob_loss(self, x, y):
        """
        Calculate the negative log probability loss.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target tensor.

        Returns:
            torch.Tensor: Computed loss.
        """
        mean, std = self.forward(x)
        normal_dist = torch.distributions.Normal(mean, std)
        loss = -normal_dist.log_prob(y).mean()  # Negative log likelihood loss
        return loss

    # Load model weights from a given path
    def load_weights(self, path):
        """
        Load model weights from a specified path.

        Args:
            path (str): Path to the weights file.
        """
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))

    # Prediction function, optionally returns mean or samples from the distribution
    def predict(self, x, mean_val=True):
        """
        Make predictions using the model.

        Args:
            x (np.ndarray or pd.DataFrame or torch.Tensor): Input data.
            mean_val (bool): If True, return mean predictions; otherwise, sample from the distribution.

        Returns:
            torch.Tensor: Predicted values.
        """
        # Convert input to tensor if in ndarray or DataFrame format
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        elif isinstance(x, pd.DataFrame):
            x = torch.tensor(x.values, dtype=torch.float)
        x = x.to(device)
        
        # Get predictions
        mean, std = self.forward(x)
        if mean_val:
            return mean  # Return mean for standard predictions
        else:
            dist = torch.distributions.Normal(mean, std)  # Sample from distribution
            return dist.rsample()

    # Training Loop ##############################################################################################
    def train(self, data_gen, batch_size=128, num_epochs=5000, learning_rate=0.0001, decay=0.00, save_model=True, save_loss=True):
        """
        Train the model using the provided data generator.

        Args:
            data_gen (DataGenerator): Data generator for sampling training, validation, and test data.
            batch_size (int): Number of samples per batch.
            num_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the optimizer.
            decay (float): Weight decay for the optimizer.
            save_model (bool): If True, save the model periodically.
            save_loss (bool): If True, save the loss history.

        Returns:
            tuple: Training and validation loss arrays.
        """
        self.to(device)  # Move model to GPU if available
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=decay)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # Sample training and validation data
            train_input, train_output = data_gen.sample('train', batch_size)
            val_input, val_output = data_gen.sample('val')
            
            # Remove unwanted columns from output data
            train_output = train_output.drop(columns=['time_in_sys'])
            val_output = val_output.drop(columns=['time_in_sys'])

            # Convert data to tensors and move to GPU
            train_input = torch.tensor(train_input.to_numpy(), dtype=torch.float).to(device)
            train_output = torch.tensor(train_output.to_numpy(), dtype=torch.float).to(device)
            val_input = torch.tensor(val_input.to_numpy(), dtype=torch.float).to(device)
            val_output = torch.tensor(val_output.to_numpy(), dtype=torch.float).to(device)

            # Forward pass and loss calculation for training
            loss = self.loss(train_input, train_output)
            self.loss_array.append(loss.item())
            
            # Forward pass and loss calculation for validation
            val_loss = self.loss(val_input, val_output)
            self.val_loss_array.append(val_loss.item())

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}', end='\r')

            # Save model and loss every 1000 epochs
            if (epoch + 1) % 1000 == 0 and save_model:
                self.save_loss()
                self.save_model()

        # Final model save after training
        if save_model:
            self.save_model()
        
        # Test the model and compute test loss
        test_input, test_output = data_gen.sample('test')
        test_output = test_output.drop(columns=['time_in_sys'])
        
        test_input = torch.tensor(test_input.to_numpy(), dtype=torch.float).to(device)
        test_output = torch.tensor(test_output.to_numpy(), dtype=torch.float).to(device)
        if test_output.ndim == 1:
            test_output = test_output.view(-1, 1)
        
        test_loss = self.loss(test_input, test_output)
        self.test_loss_array.append(test_loss.item())
        print(f'Test Loss: {test_loss.item()}')

        # Save test loss and plot loss history
        if save_loss:
            self.save_loss()

        return self.loss_array, self.val_loss_array

    def save_loss(self):
        """
        Save the training, validation, and test loss history to CSV files.
        """
        np.savetxt("./data/" + self.vnf_typ + '_loss.csv', self.loss_array, delimiter=',')
        np.savetxt("./data/" + self.vnf_typ + '_val_loss.csv', self.val_loss_array, delimiter=',')
        # save test loss in ./data/
        if self.test_loss_array != []:
            np.savetxt("./data/" + self.vnf_typ + '_test_loss.csv', self.test_loss_array, delimiter=',')

    def save_model(self):
        """
        Save the model's state dictionary to a file.
        """
        if not os.path.exists("./data/" + self.vnf_typ):
            os.makedirs("./data/" + self.vnf_typ)
        torch.save(self.state_dict(), "./data/" + self.vnf_typ + '/model.pth')

    def plot_loss(self):
        plt.plot(self.loss_array, label='Train Loss')
        plt.plot(self.val_loss_array, label='Val Loss')
        plt.legend()
        plt.title('Train and Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Log Prob. Loss')
        plt.grid()
        plt.show()