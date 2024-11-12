from scipy.stats import wasserstein_distance
import time
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

pd.set_option('display.float_format', '{:.10f}'.format)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SliceModel:
    def __init__(self, vnf_models, data_gens):
        self.vnf_models = vnf_models
        self.data_gens = data_gens
        
    def predict_slice_data_gen(self, slice_data_gen):
        res_arr = np.vstack([
            slice_data_gen.input_dataset.res_upf.values, 
            slice_data_gen.input_dataset.res_ovs.values, 
            slice_data_gen.input_dataset.res_ran.values
        ]).T
        vnf_output = slice_data_gen.input_dataset.copy()
        
        for vnf_num, (vnf_model, data_gen) in enumerate(zip(self.vnf_models, self.data_gens)):
            # Preparing input data for the current VNF model
            if vnf_num == 0:
                vnf_output = vnf_output.loc[:, data_gen.input_feature_list[:-1]]
                vnf_output['res'] = res_arr[:, vnf_num]
            else:
                res = torch.tensor(res_arr[:, vnf_num].astype(np.float32)).unsqueeze(1).to(device)
                vnf_output = torch.cat((vnf_output, res), dim=1)

            # Normalize, predict, and denormalize through the current VNF model
            vnf_output = data_gen.normalize(vnf_output, feature_type='input')
            vnf_output = vnf_model.predict(vnf_output, mean_val=True)
            vnf_output = data_gen.denormalize(vnf_output, feature_type='output')

        return vnf_output

    def predict_throughput(self, res, input_throughput, differentiable=False, res_normalized=True):
        # Ensure 'res' is a list of resource values, denormalizing if needed
        if isinstance(res, torch.Tensor):
            res = res.clone().squeeze()
        if res_normalized:
            # Convert each resource in `res` to a denormalized float
            res = [float(self.data_gens[i].denormalize(res[i], feature_type='input', feature='res')) for i in range(3)]
        # print(input_throughput)
        
        # Initialize throughput for the first VNF processing
        predicted_output = None
        throughput = input_throughput
        
        # Iterate through each VNF model and corresponding data generator
        for vnf_num, (vnf_model, data_gen) in enumerate(zip(self.vnf_models, self.data_gens)):
            # Prepare the input data for this VNF based on its resource allocation and throughput
            input_data = self._prepare_input_data(predicted_output, vnf_num, data_gen, throughput, res)

            # Pass the data through normalization, prediction, and denormalization stages
            normalized_input = data_gen.normalize(input_data, feature_type='input')
            predicted_output = vnf_model.predict(normalized_input, mean_val=True)
            denormalized_output = data_gen.denormalize(predicted_output, feature_type='output')

            # Update throughput with the output of this VNF, used as input for the next VNF
            throughput = denormalized_output[0, 2]  # Selecting throughput value in the output
            # print(f"Throughput after VNF {vnf_num}: {throughput}")
        # print("--------------------------------")
        # Return the final throughput, applying a cap if not differentiable
        return min(throughput.item(), input_throughput) if not differentiable else throughput


    def _prepare_input_data(self, output_features, vnf_num, data_gen, throughput, res):
        """Helper function to prepare input data for a specific VNF model."""
        if vnf_num == 0:
            data_sample = data_gen.get_nearest_neighbor(throughput)
            if data_sample is None:
                return None
            data_sample['res'] = res[vnf_num]
            data_sample = [
                data_sample['packet_size'],
                (throughput * 1e6) / (8 * data_sample['packet_size']),
                throughput,
                data_sample['inter_arrival_time_mean'],
                data_sample['inter_arrival_time_std'],
                res[vnf_num]
            ]
            return torch.tensor(data_sample).to(device).unsqueeze(0)
        else:
            res_tensor = torch.tensor([res[vnf_num]]).unsqueeze(1).to(device)
            return torch.cat((output_features.to(device), res_tensor), dim=1)
