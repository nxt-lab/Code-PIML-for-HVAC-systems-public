import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, Subset, DataLoader
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
import os
from load_data import load_hvac_data, data_split, generate_param_grid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

ini_weights = os.path.join("warmup_model_min_max.pth")

choice = input("Choose dataset [filtered/original]: ").strip().lower()
train_amount = int(input("Enter number of training samples (e.g., 100): "))

temp, supply, air, output, N_end, Ns_t, Ne_t = load_hvac_data(choice=choice, start=1, train_amount=train_amount)

N_start = 1
T_tr, sup_tr, air_tr, output_tr = data_split(N_start, N_end, temp, supply, air, output)
T_t, sup_t, air_t, output_t = data_split(Ns_t, Ne_t, temp, supply, air, output)

X_tr = torch.concat((T_tr, sup_tr, air_tr), dim=1)
X_t = torch.concat((T_t, sup_t, air_t), dim=1)

dataset_tr = TensorDataset(X_tr, output_tr)
patience = 30000
step_size = 20
gamma = 0.9
epoch = 500

class ExponentialParameterization(nn.Module):
    def forward(self, X):
        return torch.exp(X)
class Max(nn.Module):
    def __init__(self):
        super(Max, self).__init__()
    def forward(self, x):
        return torch.max(x, dim=-1, keepdim=True)[0]
class Min(nn.Module):
    def __init__(self):
        super(Min, self).__init__()
    def forward(self, x):
        return torch.min(x, dim=-1)[0]

class min_max(nn.Module):
    def __init__(self, num_neuron=3, dropout_p=0):
        super(min_max, self).__init__()
        self.fc1_1 = nn.Linear(1, num_neuron)  # Input layer to hidden layer 1
        self.fc1_2 = nn.Linear(1, num_neuron)
        self.fc1_3 = nn.Linear(1, num_neuron)
        self.fc2 = nn.ModuleList([Max() for _ in range(3)])  
        self.fc3 = Min()  
        parametrize.register_parametrization(self.fc1_2, "weight", ExponentialParameterization())
    def forward(self, x):
        x, y, z = x[:, :1], x[:, 1:2], x[:, 2:]
        x_1 = self.fc1_1(x)  
        y_1 = self.fc1_2(y)
        z_1 = self.fc1_3(z)
        x = x_1 + y_1 + z_1
        split_size = x.size(1) // 3
        x_splits = [x[:, i*split_size:(i+1)*split_size] for i in range(3)]
        x = torch.cat([fc(split) for fc, split in zip(self.fc2, x_splits)], dim=-1)
        output = self.fc3(x).reshape(-1, 1)
        return output
    

def choose_lr(dataset, param_grid, n_splits=5):
    lr_results = {}
    loss_fn = nn.MSELoss()
    kf = KFold(n_splits, shuffle=False)
    for params in param_grid:
        num_neuron, lr = params['num_neuron'], params['lr']
        save_path = ini_weights
        print(f"Evaluating num_neuron={num_neuron}, lr={lr}")
        avg_val_losses, avg_train_losses = [], []

        for train_idx, val_idx in kf.split(dataset):
            train_loader = DataLoader(Subset(dataset, train_idx), batch_size=len(train_idx), shuffle=False)
            val_loader = DataLoader(Subset(dataset, val_idx), batch_size=len(val_idx), shuffle=False)
            model = min_max(num_neuron=num_neuron).to(device)
            model.load_state_dict(torch.load(save_path))  # Load the pre-trained model
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            for epoch in range(params['epochs']):
                model.train()
                x_train, y_train = next(iter(train_loader))  # Only 1 batch
                x_train, y_train = x_train.to(device), y_train.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(x_train), y_train)
                loss.backward()
                optimizer.step()
                scheduler.step()
            model.eval()
            
            with torch.no_grad():
                x_val, y_val = next(iter(val_loader))
                x_val, y_val = x_val.to(device), y_val.to(device)
                val_loss = loss_fn(model(x_val), y_val).item()
            avg_val_losses.append(val_loss)
            with torch.no_grad():
                train_loss = loss_fn(model(x_train), y_train).item()
            avg_train_losses.append(train_loss)
        avg_val_loss = np.mean(avg_val_losses)
        avg_train_loss = np.mean(avg_train_losses)
        lr_results[(num_neuron, lr)] = avg_val_loss
        print(f"Neuron: {num_neuron}; Learning Rate: {lr:.4f}, Avg Validation Loss: {avg_val_loss:.6f}, Avg Training Loss: {avg_train_loss:.6f}")
    best_neuron, best_lr = min(lr_results, key=lr_results.get)
    print(f"\nBest Hyperparameters: num_neuron={best_neuron}, lr={best_lr:.4f} with Validation Loss: {lr_results[(best_neuron, best_lr)]:.6f}")
    return best_neuron, best_lr



param_grid = generate_param_grid()
best_neuron, best_lr = choose_lr(dataset_tr, param_grid, n_splits=10)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

param_grid = [{'num_neuron': 3, 'epochs': epoch, 'lr': best_lr}]
model = min_max(num_neuron=param_grid[0]['num_neuron']).to(device)
model.load_state_dict(torch.load(ini_weights))
optimizer = optim.AdamW(model.parameters(), lr=best_lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

for i in range(epoch):
    model.train()
    optimizer.zero_grad()
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    loss_fn = nn.MSELoss() 
    loss = loss_fn(model(X_tr), output_tr) 
    loss.backward()
    optimizer.step()

model.eval()
predictions = model(X_t).detach().to("cpu").numpy().reshape(-1,1)
output_t_np = output_t.detach().to("cpu").numpy().reshape(-1,1)

rmse = np.sqrt(mean_squared_error(output_t_np, predictions))*7
print("RMSE:", rmse)
r2 = r2_score(output_t_np, predictions)
print("R2:", r2)

def evaluate_test_data(model, X_test, y_test, num=1):
    model.eval()
    temp, sup, air = X_test[:, :1], X_test[:, 1:2], X_test[:, 2:]
    rmse_list = []
    mae_list = []
    r2_list = []
    inputs = torch.cat([temp[:-num], sup[:-num], air[:-num]], dim=1)
    t_next = model(inputs)
    t_np = t_next.detach().cpu().numpy()
    y_test = y_test.detach().cpu().numpy()
    rmse_list.append(np.sqrt(mean_squared_error(y_test[:-num], t_np)) * 5.9)
    mae_list.append(mean_absolute_error(y_test[:-num], t_np) * 5.9)
    r2_list.append(r2_score(y_test[:-num], t_np))
    # plt.plot(t_np, label ="prediction")
    # plt.plot(y_test[:-num], label = "ground truth")
    # plt.grid()
    # plt.legend()
    # plt.show()
    # Multi-step recursive predictions
    for i in range(1, num):
        model.eval()
        inputs = torch.cat([t_next, sup[i:i-num], air[i:i-num]], dim=1)
        print(inputs.size())
        t_next = model(inputs)
        t_np = t_next.detach().cpu().numpy()
        true_output = y_test[i:i-num]
        # Calculate and store metrics
        rmse_list.append(np.sqrt(mean_squared_error(true_output, t_np)) * 5.9)
        mae_list.append(mean_absolute_error(true_output, t_np) * 5.9)
        r2_list.append(r2_score(true_output, t_np))
    print("min_maxline:")
    print(f"\nRMSE: [{', '.join(f'{x:.4f}' for x in rmse_list)}]")
    print(f"\nMAE: [{', '.join(f'{x:.4f}' for x in mae_list)}]")
    print(f"\nR²: [{', '.join(f'{x:.4f}' for x in r2_list)}]")
    return rmse_list, mae_list, r2_list


rmse_list, mae_list, r2_list = evaluate_test_data(model, X_t, output_t, num=3)
df = pd.DataFrame({'rmse': rmse_list,'mae': mae_list,'r2': r2_list})
df = df.T
df.to_csv(f'results_min_max_{choice}_test_{N_end}.csv', index=True, header=True)


rmse_list, mae_list, r2_list = evaluate_test_data(model, X_tr, output_tr, num=1)
df = pd.DataFrame({'rmse': rmse_list,'mae': mae_list,'r2': r2_list})
df = df.T
df.to_csv(f'results_min_max_{choice}_train_{N_end}.csv', index=True, header=True)