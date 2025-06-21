import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from load_data import load_hvac_data, data_split, normalize_tensor
device = 'cpu'

import os
choice = input("Choose dataset [filtered/raw]: ").strip().lower()
train_amount = int(input("Enter number of training samples (e.g., 100): "))

if choice == "filtered":
    relative_path = os.path.join("filtered_room_104.csv")
elif choice == "original":
    relative_path = os.path.join("HVAC_B90_r104_exp_30m_20210727_15_min.csv")

data = pd.read_csv(relative_path)
room, airflow, supply = data["r104_room_temp"], data["r104_airflow_current"], data["r104_supply_discharge_temp"]
start = 1
temp = torch.tensor(np.array(room[start:-1]).reshape(-1, 1), device=device, dtype=torch.float32)
output = torch.tensor(np.array(room[start + 1:]).reshape(-1, 1), device=device, dtype=torch.float32)
air = torch.tensor(np.array(airflow[start + 1:]).reshape(-1, 1), device=device, dtype=torch.float32)
supply = torch.tensor(np.array(supply[start + 1:]).reshape(-1, 1), device=device, dtype=torch.float32)

temp, output, air, supply = map(normalize_tensor, [temp, output, air, supply])
N_start, N_end = 1, train_amount  
Ns_t, Ne_t = 530, 560

T_tr, sup_tr, air_tr, output_tr = data_split(N_start, N_end, temp, supply, air, output)
T_t, sup_t, air_t, output_t = data_split(Ns_t, Ne_t, temp, supply, air, output)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, X_tr, output_tr, likelihood):
        super(ExactGPModel, self).__init__(X_tr, output_tr, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=3)  # ARD for 3 inputs
        )
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

X_tr = torch.cat((T_tr, sup_tr, air_tr), dim=1)
X_t = torch.cat((T_t, sup_t, air_t), dim=1)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(X_tr, output_tr.squeeze(-1), likelihood)

model.train()
likelihood.train()

optimizer = torch.optim.Rprop([
    {'params': model.parameters()},
])


mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iterations = 100
for i in range(training_iterations):
    optimizer.zero_grad()
    pred_output = model(X_tr) 
    loss = -mll(pred_output, output_tr.squeeze(-1)) 
    loss.backward()
    if i % 10 == 0:
        print(f"Iteration {i + 1}/{training_iterations} - Loss: {loss.item():.4f}")
    optimizer.step()

model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(X_t)).mean.reshape(-1,1)



def evaluate_test_data(model, likelihood, X_test, y_test, num=1):
    model.eval()
    likelihood.eval()
    temp, sup, air = X_test[:, :1], X_test[:, 1:2], X_test[:, 2:]
    rmse_list = []
    mae_list = []
    r2_list = []
    # Initial prediction using the first batch
    inputs = torch.cat([temp[:-num], sup[:-num], air[:-num]], dim=1)
    t_next = likelihood(model(inputs)).mean.reshape(-1,1)
    t_np = t_next.detach().cpu().numpy()
    y_test = y_test.detach().cpu().numpy()
    
    rmse_list.append(np.sqrt(mean_squared_error(y_test[:-num], t_np)) * 5.9)
    mae_list.append(mean_absolute_error(y_test[:-num], t_np) * 5.9)
    r2_list.append(r2_score(y_test[:-num], t_np))

    for i in range(1, num):
        model.eval()
        inputs = torch.cat([t_next, sup[i:i-num], air[i:i-num]], dim=1)
        print(inputs.size())
        t_next = likelihood(model(inputs)).mean.reshape(-1,1)
        t_np = t_next.detach().cpu().numpy()
        true_output = y_test[i:i-num]
        rmse_list.append(np.sqrt(mean_squared_error(true_output, t_np)) * 5.9)
        mae_list.append(mean_absolute_error(true_output, t_np) * 5.9)
        r2_list.append(r2_score(true_output, t_np))
    print("Baseline:")
    print(f"\nRMSE: [{', '.join(f'{x:.4f}' for x in rmse_list)}]")
    print(f"\nMAE: [{', '.join(f'{x:.4f}' for x in mae_list)}]")
    print(f"\nR²: [{', '.join(f'{x:.4f}' for x in r2_list)}]")
    return rmse_list, mae_list, r2_list

rmse_list, mae_list, r2_list = evaluate_test_data(model, likelihood, X_t, output_t, num=3)
df = pd.DataFrame({'rmse': rmse_list,'mae': mae_list,'r2': r2_list})
df = df.T
df.to_csv(f'results_GP_{choice}_test_{N_end}.csv', index=True, header=True)

rmse_list, mae_list, r2_list = evaluate_test_data(model, likelihood, X_tr, output_tr, num=1)
df = pd.DataFrame({'rmse': rmse_list,'mae': mae_list,'r2': r2_list})
df = df.T
df.to_csv(f'results_GP_{choice}_train_{N_end}.csv', index=True, header=True)