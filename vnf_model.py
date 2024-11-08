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

class VNF_Model(nn.Module):  # Inherit from nn.Module
    def __init__(self, vnf_typ, n_inputs, n_hidden, n_outputs, feature_model=False):
        super().__init__()  # Initialize the parent class
        # Architecture
        self.vnf_typ = vnf_typ
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.feature_model = feature_model
        self.loss = self.log_prob_loss

        # Model
        self.bn = nn.BatchNorm1d(n_inputs)
        self.fc = nn.Linear(n_inputs, n_hidden[0])
        
        self.fc_hidden = nn.ModuleList()
        for i in range(len(n_hidden)-1):
            self.fc_hidden.append(nn.Linear(n_hidden[i], n_hidden[i+1]))
            
        self.fc_mean = nn.Linear(n_hidden[-1], n_outputs)  # Mean output
        self.fc_std = nn.Linear(n_hidden[-1], n_outputs)   # Std output

        self.bn_hidden = nn.ModuleList()
        for i in range(len(n_hidden)):
            self.bn_hidden.append(nn.BatchNorm1d(n_hidden[i]))

        self.loss_array = []
        self.val_loss_array = []
        self.test_loss_array = []

    def forward(self, x):
        # Remove the 1st column (packet size)
        x = x[:, 1:]
        # x = self.bn(x)
        x = self.fc(x)
        x = nn.ReLU()(x)
        # x = self.bn_hidden[0](x)
        for i in range(len(self.n_hidden)-1):
            x = self.fc_hidden[i](x)
            x = nn.ReLU()(x)  
            # x = self.bn_hidden[i+1](x)
        mean = self.fc_mean(x)  # Mean output
        std = self.fc_std(x)  # Std output (positive)
        std = nn.Softplus()(std)
        return mean, std  # Return both mean and std

    def log_prob_loss(self, x, y):
        mean, std = self.forward(x)
        normal_dist = torch.distributions.Normal(mean, std)
        loss = -normal_dist.log_prob(y).mean()  # Negative log likelihood
        return loss


    def load_weights(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))

    def predict(self, x, mean_val=True):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        elif isinstance(x, pd.DataFrame):
            x = torch.tensor(x.values, dtype=torch.float)
        x = x.to(device)
        mean, std = self.forward(x)  # Get mean and std
        if mean_val:
            return mean  # Return only the mean value
        else:
            # rsample from normal distribution
            dist = torch.distributions.Normal(mean, std)
            return dist.rsample()

    def train(self, data_gen, batch_size=128, num_epochs=5000, learning_rate=0.0001, decay=0.00, save_model=True, save_loss=True):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
        self.to(device)  # Move model to GPU
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=decay)
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            train_input, train_output = data_gen.sample('train', batch_size)
            val_input, val_output = data_gen.sample('val')
            train_output = train_output.drop(columns=['time_in_sys'])
            val_output = val_output.drop(columns=['time_in_sys'])


            train_input, train_output = train_input.to_numpy(), train_output.to_numpy()
            val_input, val_output = val_input.to_numpy(), val_output.to_numpy()

            train_input = torch.tensor(train_input, dtype=torch.float).to(device)  # Move to GPU
            val_input = torch.tensor(val_input, dtype=torch.float).to(device)  # Move to GPU
            train_output = torch.tensor(train_output, dtype=torch.float).to(device)  # Move to GPU
            val_output = torch.tensor(val_output, dtype=torch.float).to(device)  # Move to GPU

            loss = self.loss(train_input, train_output)
            self.loss_array.append(loss.item())
            val_loss = self.loss(val_input, val_output)
            self.val_loss_array.append(val_loss.item())

            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}', end='\r')

            if (epoch+1) % 1000 == 0 and save_model:
                self.save_loss()
                self.save_model()

        # Save the model
        if save_model:
            self.save_model()
        
        # Test the model
        test_input, test_output = data_gen.sample('test')
        test_output = test_output.drop(columns=['time_in_sys'])
        test_input, test_output = test_input.to_numpy(), test_output.to_numpy()
        test_input = torch.tensor(test_input, dtype=torch.float).to(device)  # Move to GPU
        test_output = torch.tensor(test_output, dtype=torch.float).to(device)  # Move to GPU
        if test_output.ndim == 1:
            test_output = test_output.view(-1, 1)
        test_loss = self.loss(test_input, test_output)
        self.test_loss_array.append(test_loss.item())
        print(f'Test Loss: {test_loss.item()}')

        test_pred = self.predict(test_input)
        # Plot the loss
        if save_loss:
            self.save_loss()

        return self.loss_array, self.val_loss_array
        # return test_loss.item(), self.loss_array, self.val_loss_array, self.test_loss_array

    def save_loss(self):
        np.savetxt("./data/" + self.vnf_typ + '_loss.csv', self.loss_array, delimiter=',')
        np.savetxt("./data/" + self.vnf_typ + '_val_loss.csv', self.val_loss_array, delimiter=',')
        # save test loss in ./data/
        if self.test_loss_array != []:
            np.savetxt("./data/" + self.vnf_typ + '_test_loss.csv', self.test_loss_array, delimiter=',')

    def save_model(self):
        torch.save(self.state_dict(), "./" + self.vnf_typ + '/model.pth')