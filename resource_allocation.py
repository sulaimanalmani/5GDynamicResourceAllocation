import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResAllocModel(nn.Module):
    def __init__(self, init_weights=None):
        super().__init__()
        self.fc_layer = nn.Linear(1, 3, bias=False)
        if init_weights is not None:
            self.fc_layer.weight.data = torch.tensor(
                [[init_weights[0]], [init_weights[1]], [init_weights[2]]], dtype=torch.float32
            )
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.fc_layer(x))

def microopt(slice_model, input_throughput, qos_threshold, learning_rate=0.001, init_weights=None, verbose=False):
    # Initial solution
    res_alloc_init = get_initial_solution(slice_model, input_throughput, qos_threshold)
    init_weights = torch.tensor(np.log(res_alloc_init / (1 - res_alloc_init + 1e-6)))

    # Initialize variables
    epochs, max_time, max_iterations = 50, 20, 50  # Increased max_time and max_iterations
    penalty, penalty_step, min_upper_bound = 1.0, 0.1, 2.0  # Reduced penalty_step for finer adjustments
    feasible_allocation, feasible_qos, upper_bound, lower_bound = None, None, None, None
    iteration, iterations_since_optimal = 0, 0
    start_time = time.time()

    # Outer loop for primal-dual optimization
    while True:
        model = ResAllocModel(init_weights=init_weights).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        iteration += 1
        if verbose: print(f"Iteration {iteration}")

        # Perform inner loop (gradient descent) within primal-dual algorithm
        res_alloc, qos, loss = inner_loop(model, optimizer, slice_model, input_throughput, qos_threshold, penalty, epochs, verbose)
        
        # Update feasibility bounds
        feasible_allocation, feasible_qos, upper_bound, lower_bound, min_upper_bound, iterations_since_optimal = \
            update_bounds_if_feasible(
                model, res_alloc, qos, qos_threshold, loss, penalty, upper_bound, min_upper_bound, feasible_allocation,
                feasible_qos, init_weights, iterations_since_optimal, verbose
            )

        # Adjust penalty and check stopping conditions
        penalty = adjust_penalty(penalty, qos, qos_threshold, penalty_step)
        if stop_conditions_met(feasible_qos, qos_threshold, start_time, max_time, iteration, max_iterations):
            if verbose: print("Optimization complete")
            break

    return feasible_allocation, feasible_qos, upper_bound, lower_bound, time.time() - start_time

# Inner Loop for Gradient Descent #############################################################

def inner_loop(model, optimizer, slice_model, input_throughput, qos_threshold, penalty, epochs, verbose):
    for epoch in range(epochs):
        optimizer.zero_grad()
        res_alloc = model(torch.ones((1, 1)).to(device))
        qos = slice_model.predict_throughput(res_alloc, input_throughput, differentiable=True)
        constraint_violation = qos_threshold - qos
        loss = calculate_loss(res_alloc, penalty, constraint_violation)
        loss.backward(retain_graph=True)

        # Update weights after the first epoch
        if epoch >= 1:
            optimizer.step()

        # Reduced gradient norm threshold for finer convergence
        grad_norm = torch.sqrt(sum(torch.norm(p.grad) ** 2 for p in model.parameters())).item()
        if grad_norm < 0.001 or epoch == epochs - 1:  # Lowered threshold from 0.01 to 0.001
            if verbose: print("Converged early")
            break

    return res_alloc.detach().cpu().numpy(), qos.item(), loss.item()

# Feasibility Check and Bounds Update #########################################################

def update_bounds_if_feasible(model, res_alloc, qos, qos_threshold, loss, penalty, upper_bound, min_upper_bound, 
                              feasible_allocation, feasible_qos, init_weights, iterations_since_optimal, verbose):
    constraint_violation = qos_threshold - qos
    current_upper_bound = upper_bound  # Default value in case the feasible condition is not met

    if constraint_violation <= 0:
        current_upper_bound = res_alloc.mean()
        if current_upper_bound < min_upper_bound:
            feasible_allocation = res_alloc
            feasible_qos = qos
            # Update init_weights directly as a torch.FloatTensor
            init_weights[:] = torch.tensor(model.fc_layer.weight.detach().cpu().numpy().flatten(), dtype=torch.float32)
            min_upper_bound = current_upper_bound
            iterations_since_optimal = 0
            if verbose: print(f"Feasible allocation: {feasible_allocation}, Feasible QoS: {feasible_qos}, Constraint violation: {constraint_violation}")
    else:
        iterations_since_optimal += 1
    
    return feasible_allocation, feasible_qos, current_upper_bound, loss, min_upper_bound, iterations_since_optimal

# Calculate Loss for Resource Allocation ######################################################

def calculate_loss(res_alloc, penalty, constraint_violation):
    return res_alloc.sum() + penalty * constraint_violation

# Penalty Adjustment ##########################################################################

def adjust_penalty(penalty, qos, qos_threshold, penalty_step):
    constraint_violation = qos_threshold - qos if qos else 0
    return max(0, penalty + penalty_step * constraint_violation)

# Stop Condition Check ########################################################################

def stop_conditions_met(feasible_qos, qos_threshold, start_time, max_time, iteration, max_iterations):
    return (abs(feasible_qos - qos_threshold) < 0.5 or  # Reduced QoS tolerance for finer results
            time.time() - start_time > max_time or 
            iteration > max_iterations)

# Slice Model Interaction: Grid Search for Initialization #####################################

def get_initial_solution(slice_model, input_throughput, qos_threshold):
    best_allocation, min_res_alloc_sum = None, float('inf')
    for ovs in np.arange(0, 1, 0.1):
        for ran in np.arange(0, 1, 0.1):
            res_alloc = torch.tensor([1, ovs, ran], dtype=torch.float32)
            qos = slice_model.predict_throughput(res_alloc, input_throughput, differentiable=True)
            if qos > qos_threshold and res_alloc.sum() < min_res_alloc_sum:
                best_allocation, min_res_alloc_sum = res_alloc, res_alloc.sum()
    return best_allocation.detach().cpu().numpy()
