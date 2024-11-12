import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
torch.autograd.set_detect_anomaly(True)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class ResAllocModel(nn.Module):
    def __init__(self, init_weights=None):
        super(ResAllocModel, self).__init__()
        self.sigmoid = nn.Sigmoid()        
        self.fc_op = nn.Linear(1, 3, bias=False)
        if init_weights is not None:
            self.fc_op.weight.data = torch.tensor([[init_weights[0]], [init_weights[1]], [init_weights[2]]], dtype=torch.float32)
        self.fc_op.weight.requires_grad = True

    def forward(self, x):
        x = self.fc_op(x)
        x = self.sigmoid(x)
        return x

def MicroOpt(slice_model, input_throughput, qos_thresh, lr=0.001, init_weights=None, verbose=0):
    """
    Micro-optimization for resource allocation.
    """
    res_alloc_init = get_gridsearch_solution(slice_model, input_throughput, qos_thresh)
    res_alloc_init = np.array(res_alloc_init)
    print(f"Init weights: {res_alloc_init}")
    epochs = 50
    max_time = 10
    max_iter = 10
    penalty = 1.0
    penalty_step = 0.5
    feasible_res_alloc = None
    feasible_qos = None
    UB = None
    LB = None
    n_sol = 0
    min_UB = 2.0
    iters = 0
    iters_since_optim = 0

    start_time = time.time()
    # Do inverse sigmoid on the res_alloc_init
    init_weights = torch.tensor(np.log(res_alloc_init/(1-res_alloc_init+1e-6)))

    while True:
        losses = []
        res_alloc_model = ResAllocModel(init_weights=init_weights)
        epoch = 0
        optimizer = optim.Adam(res_alloc_model.parameters(), lr=lr) # reset optimizer
        iters += 1
        epoch_losses = []
        cost = []
        if verbose: print(f"################\nIteration: {iters}")

        while True:
            epoch += 1
            res_alloc_model.zero_grad()
            res_alloc = res_alloc_model(torch.ones((1, 1)))
            qos = slice_model.predict_throughput(res_alloc, input_throughput, differentiable=1)
            constr_viol = qos_thresh - qos

            loss = res_alloc.sum() + (penalty * constr_viol)
            if verbose: print(f"Loss: {loss}, QoS: {qos}, Penalty: {penalty}, Constrain Violation: {constr_viol}")
            loss.backward(retain_graph=True)
            epoch_losses.append(loss.item())
            if iters > 1:
                optimizer.step()
            else:
                start_time = time.time()

            grad_norm = 0.0
            for param in res_alloc_model.parameters():
                grad_norm += torch.norm(param.grad.data, p=2) ** 2
            grad_norm = grad_norm.sqrt().item()

            if len(cost) > 20:
                if all(cost[-20] > 0) : 
                    break
            stop = False            
            if grad_norm < 0.01 or stop or epoch > epochs:
                if verbose: print(f"Gradient norm: {grad_norm}, Epoch: {epoch}, Stop: {stop}")
                if verbose: print("Converged after {} epochs".format(epoch))
                break
            losses.append(loss.item())
        
        iters_since_optim += 1
        
        if verbose: print(f"Current UB: {min_UB}, time: {time.time() - start_time}")
        if constr_viol <= 0 or iters == 1:
            UB = res_alloc.mean().detach().numpy()
            LB = loss.item()
            
            if UB < min_UB:
                if verbose: print(f"New feasible solution found with UB: {UB}")
                n_sol += 1
                min_UB = UB
                feasible_res_alloc = res_alloc.detach().numpy()
                feasible_qos = qos.item()
                if iters > 1:
                    init_weights = res_alloc_model.fc_op.weight.detach().numpy().flatten()
                iters_since_optim = 0

        penalty += penalty_step * constr_viol
        #reduce lr
        # lr = lr * 0.99
        if verbose: print(f"Constr viol: {constr_viol}")
        penalty = max(0, penalty)
        if verbose: print(f"Penalty: {penalty}")

        if time.time() - start_time > max_time or iters > max_iter:
            if abs(feasible_qos - qos_thresh) < 1:
                if verbose: print("Max iterations reached")
                break

    return feasible_res_alloc, feasible_qos, UB, LB, time.time() - start_time

def get_gridsearch_solution(slice_model, input_throughput, qos_thresh):
    """
    Get the gridsearch solution for initial resource allocation.
    """
    min_res_alloc_sum = math.inf
    res_alloc = None
    for res_alloc_ovs in np.arange(0, 1, 0.1):
        for res_alloc_ran in np.arange(0, 1, 0.1):
            res_alloc = torch.tensor([1, res_alloc_ovs, res_alloc_ran])
            qos = slice_model.predict_throughput(res_alloc, input_throughput, differentiable=1)
            if qos > qos_thresh and res_alloc.sum() < min_res_alloc_sum:
                min_res_alloc_sum = res_alloc.sum()
                min_res_alloc = res_alloc
    return min_res_alloc