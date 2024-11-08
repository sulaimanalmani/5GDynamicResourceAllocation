from scipy.stats import wasserstein_distance
import time
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
pd.set_option('display.float_format', '{:.10f}'.format)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Slice_Model:
    def __init__(self, vnf_models, data_gens):
        self.vnf_models = vnf_models
        self.data_gens = data_gens
        
    def predict_slice_data_gen(self, slice_data_gen):
        res = np.array([slice_data_gen.input_dataset.res_upf.values, slice_data_gen.input_dataset.res_ovs.values, slice_data_gen.input_dataset.res_ran.values])
        res_arr = res.T
        vnf_output = slice_data_gen.input_dataset
        vnf_num = 0
        for vnf_model, data_gen in zip(self.vnf_models, self.data_gens):
            if vnf_num == 0:
                vnf_output = vnf_output.loc[:, data_gen.input_feature_list[:-1]]  
                vnf_output.loc[:, 'res'] = res_arr[:, vnf_num] 
            else:
                res = res_arr[:, vnf_num]
                res = np.expand_dims(res, axis=1).astype(np.float32)
                vnf_output = torch.cat((vnf_output, torch.tensor(res).to(device)), dim=1)
            vnf_output = data_gen.normalize(vnf_output, feature_type='input')
            vnf_output = vnf_model.predict(vnf_output, mean_val=True)  
            vnf_output = data_gen.denormalize(vnf_output, feature_type='output')
            vnf_num += 1
        return vnf_output
    
    def predict_throughput(self, res, input_throughput, differentiable=0, res_normalized=True):
        if isinstance(res, torch.Tensor):
            res = res.clone().squeeze()
        if res_normalized:
            res[1] = self.data_gens[1].denormalize(res[1], feature_type='input', feature='res')
            res[2] = self.data_gens[2].denormalize(res[2], feature_type='input', feature='res')
            res = [float(res[0]), float(res[1]), float(res[2])]
        init_throughput = input_throughput
        vnf_num = 0
        for vnf_model, data_gen in zip(self.vnf_models, self.data_gens):
            if vnf_num == 0:
                input_dataset = data_gen.input_dataset.copy()
                # Extract the input dataset where 'throughput' is +-5 of the input_throughput
                input_dataset = input_dataset[input_dataset['throughput'].between(input_throughput-2, input_throughput+2)]
                mean_input_dataset = pd.DataFrame(input_dataset.mean()).T
                mean_input_dataset = torch.tensor(mean_input_dataset.values.astype(np.float32)).to(device)
                mean_input_dataset[:, -1] = float(res[vnf_num])
                # Add a duplicate row to the mean_input_dataset
                mean_input_dataset = torch.cat((mean_input_dataset, mean_input_dataset), dim=0)
            else: 
                mean_input_dataset = torch.cat((mean_input_dataset.to(device), torch.tensor([res[vnf_num]]).unsqueeze(1).repeat(2, 1).to(device)), dim=1)
            mean_input_dataset = data_gen.normalize(mean_input_dataset, feature_type='input')
            mean_input_dataset = vnf_model.predict(mean_input_dataset, mean_val=True)
            mean_input_dataset = data_gen.denormalize(mean_input_dataset, feature_type='output')
            vnf_num += 1
            input_throughput = mean_input_dataset[0, 2]

        output_throughput = input_throughput
        if not differentiable:
            return min(output_throughput.item(), init_throughput)
        else:
            return output_throughput
