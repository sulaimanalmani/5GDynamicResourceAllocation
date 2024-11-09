import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import math
from scipy.interpolate import griddata

class DataGenerator:
    """
    A class to generate and preprocess resource allocation data for VNFs.

    This class handles loading, normalizing, and splitting the dataset into training, validation, 
    and test sets. The class provides methods for sampling data, normalizing and denormalizing values, 
    and creating 3D plots for data visualization.

    Attributes:
        input_dataset_file (str): Path to the input dataset file.
        output_dataset_file (str): Path to the output dataset file.
        vnf_type (str): Type of VNF, e.g., 'RAN', 'OvS', 'UPF', 'slice'.
        norm_type (str): Type of normalization to apply, e.g., 'None', 'minmax', 'std'.
        split (list): Ratios for splitting the dataset into train, validation, and test sets.
    """
    def __init__(self, input_dataset_file, output_dataset_file, vnf_type='ran', norm_type='None', split=[0.8, 0.9, 1.0]):
        if input_dataset_file.endswith('.pkl'):
            self.input_dataset = pd.read_pickle(input_dataset_file)
            self.output_dataset = pd.read_pickle(output_dataset_file)
        else:
            self.input_dataset = pd.read_csv(input_dataset_file)
            self.output_dataset = pd.read_csv(output_dataset_file)
        # Convert throughput from bytes to Mbps
        self.input_dataset['throughput'] = self.input_dataset['throughput'] * 8 / (1e6) 
        self.output_dataset['throughput'] = self.output_dataset['throughput'] * 8 / (1e6)        

        if vnf_type == 'UPF':
            # Replace the 'res' column with a constant value of 200
            self.input_dataset['res'] = 200

        # Drop 
        self.timestamp_arr = self.input_dataset['time_stamp_arr']
        self.inter_arrival_time_arr = self.input_dataset['time_stamp_arr'].apply(lambda times: [times[1]-times[0]] + [times[i+1] - times[i] for i in range(len(times) - 1)])
        self.time_in_sys_arr = self.output_dataset['time_in_sys_arr']

        self.norm_type = norm_type
        self.vnf_type = vnf_type
        resource_types = {'RAN': 'CPU (millicores)',
                         'OvS': 'Transport Bandwidth (Mbps)',
                         'UPF': 'CPU (millicores)',
                         'slice': ['CPU (millicores)', 'Transport Bandwidth (Mbps)', 'CPU (millicores)']
                         }
        self.resource_type = resource_types[self.vnf_type]
        if self.vnf_type.startswith('slice'):
            self.input_dataset['res_ran'] = self.input_dataset['res'].apply(lambda x: x[0])
            self.input_dataset['res_ran'] = self.input_dataset['res_ran'].astype(float)
            self.input_dataset['res_ovs'] = self.input_dataset['res'].apply(lambda x: x[1])
            self.input_dataset['res_ovs'] = self.input_dataset['res_ovs'].astype(float)
            self.input_dataset['res_upf'] = 1.0
            self.input_dataset['res_upf'] = self.input_dataset['res_upf'].astype(float)
            self.input_dataset.drop(columns=['res'], inplace=True)
        else:
            self.input_dataset['res'] = self.input_dataset['res'].astype(float)


        valid_indices = ~self.output_dataset['time_in_sys'].isna()  # Create a boolean mask
        self.input_dataset = self.input_dataset[valid_indices].reset_index(drop=True)  # Filter input dataset and reset index
        self.output_dataset = self.output_dataset[valid_indices].reset_index(drop=True)  # Filter output dataset and reset index
        self.input_dataset.drop(columns=['time_stamp_arr'], inplace=True)
        self.output_dataset.drop(columns=['time_in_sys_arr', 'time_stamp_arr'], inplace=True)
        self.input_feature_list = self.input_dataset.columns.tolist()
        self.output_feature_list = self.output_dataset.columns.tolist()

        # Shuffle the dataset but use same indices for input and output
        np.random.seed(42)  # Set a random seed for reproducibility
        self.indices = np.arange(len(self.input_dataset))
        np.random.shuffle(self.indices)
        self.input_dataset = self.input_dataset.iloc[self.indices]
        self.output_dataset = self.output_dataset.iloc[self.indices]
        self.timestamp_arr = self.timestamp_arr.iloc[self.indices]
        self.time_in_sys_arr = self.time_in_sys_arr.iloc[self.indices]
        self.inter_arrival_time_arr = self.inter_arrival_time_arr.iloc[self.indices]

        # Split the dataset into train, validation and test
        self.train_idx = int(len(self.input_dataset) * split[0])
        self.val_idx = int(len(self.input_dataset) * split[1])
        self.test_idx = int(len(self.input_dataset) * split[2])

        self.train_input = self.input_dataset[:self.train_idx]
        self.train_output = self.output_dataset[:self.train_idx]
        self.val_input = self.input_dataset[self.train_idx:self.val_idx]
        self.val_output = self.output_dataset[self.train_idx:self.val_idx]
        self.test_input = self.input_dataset[self.val_idx:self.test_idx]
        self.test_output = self.output_dataset[self.val_idx:self.test_idx]   
        self.train_time_in_sys = self.time_in_sys_arr[:self.train_idx]
        self.val_time_in_sys = self.time_in_sys_arr[self.train_idx:self.val_idx]
        self.test_time_in_sys = self.time_in_sys_arr[self.val_idx:self.test_idx]

        self.input_min_scalars = np.min(self.train_input.values, axis=0)
        self.input_max_scalars = np.max(self.train_input.values, axis=0)
        self.output_min_scalars = np.min(self.train_output.values, axis=0)
        self.output_max_scalars = np.max(self.train_output.values, axis=0)

        self.input_mean_scalars = np.mean(self.train_input.values, axis=0)
        self.input_std_scalars = np.std(self.train_input.values, axis=0)
        self.output_mean_scalars = np.mean(self.train_output.values, axis=0)
        self.output_std_scalars = np.std(self.train_output.values, axis=0)

        if norm_type != 'None':
            self.train_input = self.normalize(self.train_input, 'input')
            self.train_output = self.normalize(self.train_output, 'output')
            self.val_input = self.normalize(self.val_input, 'input')
            self.val_output = self.normalize(self.val_output, 'output')
            self.test_input = self.normalize(self.test_input, 'input')
            self.test_output = self.normalize(self.test_output, 'output')  

    
    def get_norm_params(self, norm_type, feature_type, feature_idx):
        """
        Retrieve normalization parameters for a given feature.

        Depending on the normalization type ('minmax' or 'std'), this function returns
        the appropriate parameters (min and max values for 'minmax', mean and standard deviation for 'std')
        for the specified feature index and type (input or output).

        Args:
            norm_type (str): The type of normalization ('minmax' or 'std').
            feature_type (str): The type of feature ('input' or 'output').
            feature_idx (int): The index of the feature for which to retrieve parameters.

        Returns:
            tuple: A tuple containing the normalization parameters.
        """
        if norm_type == 'minmax':
            if feature_type == 'input':
                return self.input_min_scalars[feature_idx], self.input_max_scalars[feature_idx]
            else:
                return self.output_min_scalars[feature_idx], self.output_max_scalars[feature_idx]
        elif norm_type == 'std':
            if feature_type == 'input':
                return self.input_mean_scalars[feature_idx], self.input_std_scalars[feature_idx]
            else:
                return self.output_mean_scalars[feature_idx], self.output_std_scalars[feature_idx]

    def normalize_val(self, data, norm_type, feature, feature_type):
        """
        Normalize a single feature value or a series of values.

        This function applies the specified normalization technique ('minmax' or 'std') 
        to a given feature, either by its name or index, for the specified feature type 
        ('input' or 'output'). It returns the normalized value(s).

        Args:
            data (float or pd.Series): The data to be normalized.
            norm_type (str): The type of normalization ('minmax' or 'std').
            feature (str or int): The feature name or index to normalize.
            feature_type (str): The type of feature ('input' or 'output').

        Returns:
            float or pd.Series: The normalized value(s).
        """
        if isinstance(feature, str):
            if feature_type == 'input':
                feature_idx = self.input_feature_list.index(feature)
            else:
                feature_idx = self.output_feature_list.index(feature)
        else:
            feature_idx = feature
        params = self.get_norm_params(norm_type, feature_type, feature_idx)

        if norm_type == 'minmax':
            min_val = params[0]
            max_val = params[1]
            range_val = max_val - min_val
            if range_val != 0:
                return (data - min_val) / range_val
            else:
                return 0.5
        elif norm_type == 'std':
            mean_val = params[0]
            std_val = params[1]
            if std_val != 0:
                return (data - mean_val) / std_val
            else:
                return 0

    def denormalize_val(self, data, norm_type, feature, feature_type):
        """
        Denormalize a single feature value or a series of values.

        This function reverses the specified normalization technique ('minmax' or 'std') 
        applied to a given feature, either by its name or index, for the specified feature type 
        ('input' or 'output'). It returns the denormalized value(s).

        Args:
            data (float, pd.Series, or pd.DataFrame): The data to be denormalized.
            norm_type (str): The type of normalization ('minmax' or 'std').
            feature (str or int): The feature name or index to denormalize.
            feature_type (str): The type of feature ('input' or 'output').

        Returns:
            float, pd.Series, or pd.DataFrame: The denormalized value(s).
        """
        if isinstance(data, pd.DataFrame):
            data = data.copy()
        if isinstance(feature, str):
            if feature_type == 'input':
                feature_idx = self.input_feature_list.index(feature)
            else:
                feature_idx = self.output_feature_list.index(feature)
        else:
            feature_idx = feature

        params = self.get_norm_params(norm_type, feature_type, feature_idx)

        if norm_type == 'minmax':
            min_val = params[0]
            max_val = params[1]
            range_val = max_val - min_val
            if range_val != 0:
                return (data * range_val) + min_val
            else:
                return (min_val + max_val) / 2
        elif norm_type == 'std':
            mean_val = params[0]
            std_val = params[1]
            if std_val != 0:
                return (data * std_val) + mean_val
            else:
                return mean_val

    def normalize(self, data, feature_type='input', feature=None):
        """
        Normalize the dataset or a specific feature.

        This function normalizes the entire dataset or a specific feature using the 
        normalization type specified during the initialization of the DataGenerator class.
        It supports both 'minmax' and 'std' normalization techniques.

        Args:
            data (pd.DataFrame or np.ndarray): The data to be normalized.
            feature_type (str): The type of feature ('input' or 'output').
            feature (str or int, optional): The feature name or index to normalize. 
                                            If None, all features are normalized.

        Returns:
            pd.DataFrame or np.ndarray: The normalized data.
        """
        if isinstance(data, pd.DataFrame):
            data = data.copy()       
            data = data.astype(float)  # Add this line to cast to float
        if feature is not None:
            return self.normalize_val(data, self.norm_type, feature, feature_type)
        for i in range(data.shape[1]):
            feature_idx = i
            if isinstance(data, pd.DataFrame):
                data.iloc[:, i] = self.normalize_val(data.iloc[:, i], self.norm_type, feature_idx, feature_type)
            else:
                data[:, i] = self.normalize_val(data[:, i], self.norm_type, feature_idx, feature_type)
        return data  # Return the modified copy

    def denormalize(self, data, feature_type='input', feature=None):
        """
        Denormalize the dataset or a specific feature.

        This function denormalizes the entire dataset or a specific feature using the 
        normalization type specified during the initialization of the DataGenerator class.
        It supports both 'minmax' and 'std' normalization techniques.

        Args:
            data (pd.DataFrame or np.ndarray): The data to be denormalized.
            feature_type (str): The type of feature ('input' or 'output').
            feature (str or int, optional): The feature name or index to denormalize. 
                                            If None, all features are denormalized.

        Returns:
            pd.DataFrame or np.ndarray: The denormalized data.
        """
        if isinstance(data, pd.DataFrame):
            data = data.copy()
        if feature is not None:
            return self.denormalize_val(data, self.norm_type, feature, feature_type)
        else:
            for i in range(data.shape[1]):
                feature_idx = i
                if isinstance(data, pd.DataFrame):
                    data.iloc[:, i] = self.denormalize_val(data.iloc[:, i], self.norm_type, feature_idx, feature_type)
                else:
                    data[:, i] = self.denormalize_val(data[:, i], self.norm_type, feature_idx, feature_type)
        return data  # Return the modified copy

    def sample(self, typ='train', size=None):
        """
        Sample data from the dataset.

        This function returns random samples from the specified dataset type (train, val, or test).
        If the size is not specified or exceeds the dataset length, it returns the entire dataset.

        Args:
            typ (str): The type of dataset to sample from ('train', 'val', 'test', or 'test_all').
            size (int, optional): The number of samples to return. If None, returns the entire dataset.

        Returns:
            tuple: A tuple containing the sampled input and output data.
        """
        # Return random samples from the dataset of size 'size' but use same idxs
        if typ == 'train':
            if size is None or size > len(self.train_input):
                return self.train_input, self.train_output
            indices = np.random.choice(len(self.train_input), size, replace=False)
            return self.train_input.iloc[indices], self.train_output.iloc[indices]  # Use .iloc for indexing
        elif typ == 'val':
            if size is None or size > len(self.val_input):
                return self.val_input, self.val_output
            indices = np.random.choice(len(self.val_input), size, replace=False)
            return self.val_input.iloc[indices], self.val_output.iloc[indices]  # Use .iloc for indexing
        elif typ == 'test' or typ == 'test_all':
            if size is None:
                return self.test_input, self.test_output
            indices = np.random.choice(len(self.test_input), size, replace=False)
            return self.test_input.iloc[indices], self.test_output.iloc[indices]  # Use .iloc for indexing
        else:
            raise ValueError(f"Invalid type: {typ}")

    def create_3d_plot(self, res=None, input_throughput=None, output_throughput=None):
        """
        Create a 3D plot for data visualization.

        This function generates a 3D surface plot to visualize the relationship between 
        resource allocation, input throughput, and output throughput. It uses interpolation 
        to create a smooth surface.

        Args:
            res (pd.Series, optional): Resource allocation data. Defaults to training data if None.
            input_throughput (pd.Series, optional): Input throughput data. Defaults to training data if None.
            output_throughput (pd.Series, optional): Output throughput data. Defaults to training data if None.

        Returns:
            None
        """
        pred = 1
        # Creating a DataFrame to simulate grouped statistics
        if res is None:
            res = self.train_input['res']
        if input_throughput is None:
            input_throughput = self.train_input['throughput']
        if output_throughput is None:
            pred = 0
            output_throughput = self.train_output['throughput']
        
        if self.norm_type != 'None':
            res = self.denormalize_val(res, self.norm_type, 'res', 'input')
            input_throughput = self.denormalize_val(input_throughput, self.norm_type, 'throughput', 'input')
            output_throughput = self.denormalize_val(output_throughput, self.norm_type, 'throughput', 'output')

        data = pd.DataFrame({
            'res': res,
            'input_throughput': input_throughput,
            'output_throughput': output_throughput
        })
        
        # Creating a grid for interpolation using all data points
        res_grid_all, input_grid_all = np.meshgrid(
            np.linspace(data['res'].min(), data['res'].max(), 50),
            np.linspace(data['input_throughput'].min(), data['input_throughput'].max(), 50)
        )
        
        # Interpolating output_throughput values on the grid for all data points
        output_grid_all = griddata(
            (data['res'], data['input_throughput']),
            data['output_throughput'],
            (res_grid_all, input_grid_all),
            method='linear'
        )
        
        # Plotting the surface plot with the interpolated grid of all data points
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf_all = ax.plot_surface(res_grid_all, input_grid_all, output_grid_all, cmap='viridis')
        
        print(self.vnf_type)
        ax.set_title(f"3D Surface Plot for {self.vnf_type} Throughput")
        ax.set_xlabel(self.resource_type)
        ax.set_ylabel('Input Throughput (Mbps)')
        if pred == 0:
            ax.set_zlabel('Output Throughput (Mbps)')
        else:
            ax.set_zlabel('Predicted Output Throughput (Mbps)')
        
        # Adding a color bar
        fig.colorbar(surf_all, ax=ax, shrink=0.75, aspect=7)
        
        plt.show()

        

    def get_nearest_neighbor(self, input_throughput, resource_allocation):
        """
        Find the nearest neighbor in the dataset.

        This function identifies the nearest neighbor in the training dataset based on 
        input throughput and resource allocation. It returns a dictionary with the nearest 
        neighbor's input throughput, resource allocation, output throughput, time in system, 
        and packet loss.

        Args:
            input_throughput (float): The input throughput to find the nearest neighbor for.
            resource_allocation (float): The resource allocation to find the nearest neighbor for.

        Returns:
            dict or None: A dictionary with the nearest neighbor's data or None if no neighbor is found.
        """
        if input_throughput > self.train_input['throughput'].max() or input_throughput < self.train_input['throughput'].min() \
            or resource_allocation > self.train_input['res'].max() or resource_allocation < self.train_input['res'].min():
            # print(f"Input throughput must be between {self.train_input['throughput'].min()} and {self.train_input['throughput'].max()}")
            # print(f"Resource allocation must be between {self.train_input['res'].min()} and {self.train_input['res'].max()}")
            return None
       
        input_data = self.train_input.copy()
        output_data = self.train_output.copy()
        merged_data = pd.merge(input_data, output_data, left_index=True, right_index=True, suffixes=('_input', '_output'))
        # Aggregate by input throughput (of intervals of 5) and mean the other features
        merged_data['throughput_input'] = merged_data['throughput_input'].apply(lambda x: round(x/5)*5)
        merged_data = merged_data.groupby(['throughput_input', 'res']).mean().reset_index()

        # Filter the next lowest resource allocation
        min_res = merged_data[merged_data['res'] >= resource_allocation]['res'].min()
        filter_data = merged_data[merged_data['res'] == min_res]
        # Now filter the next lowest input throughput
        filter_data = filter_data[filter_data['throughput_input'] >= input_throughput]
        filter_data = filter_data.sort_values(by=['throughput_input'], ascending=[True])
        if len(filter_data) == 0:
            # print(f"No data found for input throughput {input_throughput} and resource allocation {resource_allocation}")
            # print(f"Please try a different input throughput and resource allocation")
            return None
        nearest_neighbor = filter_data.iloc[0]
        return_df = {
            'input_throughput': float(nearest_neighbor['throughput_input']),
            'resource_allocation': float(nearest_neighbor['res']),
            'output_throughput': min(float(nearest_neighbor['throughput_output']), float(nearest_neighbor['throughput_input'])),
            'time_in_sys': float(nearest_neighbor['time_in_sys']) * 1e3,
            'packet_loss': max(0, (float(nearest_neighbor['throughput_input'] - float(nearest_neighbor['throughput_output'])) / float(nearest_neighbor['throughput_input'])) * 100)
        }
        return return_df
        

if __name__ == '__main__':
    ran_data_gen = DataGenerator("./ran/input_dataset.pkl", "./ran/output_dataset.pkl", vnf_type='ran')
    print(ran_data_gen.train_input.columns)
    print(ran_data_gen.train_output.columns)