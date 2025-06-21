import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, Subset, DataLoader
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
from load_data import load_hvac_data, data_split, generate_param_grid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

ini_weights = os.path.join("warmup_model_neu_relu_3.pth")

choice = input("Choose dataset [filtered/original]: ").strip().lower()
train_amount = int(input("Enter number of training samples (e.g., 100): "))

temp, supply, air, output, N_end = load_hvac_data(choice=choice, start=1, train_amount=train_amount)

N_start = 1
Ns_t, Ne_t = 530, 560
T_tr, sup_tr, air_tr, output_tr = data_split(N_start, N_end, temp, supply, air, output)
T_t, sup_t, air_t, output_t = data_split(Ns_t, Ne_t, temp, supply, air, output)

X_tr = torch.concat((T_tr, sup_tr, air_tr), dim=1)
X_t = torch.concat((T_t, sup_t, air_t), dim=1)

dataset_tr = TensorDataset(X_tr, output_tr)
patience = 30000
step_size = 20
gamma = 0.9
epoch = 500

class NODE(nn.Module):
    def __init__(self, num_neuron=8):
        super(NODE, self).__init__()
        self.fc1 = nn.Linear(3, num_neuron)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_neuron, 1)
    def _func(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def _rk4(self, x, dt=0.25):
        y0 = x[:, 0:1]     
        y  = x[:, 1:3]        
        half_dt = 0.5 * dt
        one_sixth = 1.0 / 6.0
        cat = torch.cat((y0, y), dim=1)  
        k1 = self._func(cat)
        cat_k2 = torch.cat((y0 + half_dt*k1, y), dim=1)
        k2 = self._func(cat_k2)
        cat_k3 = torch.cat((y0 + half_dt*k2, y), dim=1)
        k3 = self._func(cat_k3)
        cat_k4 = torch.cat((y0 + dt*k3, y), dim=1)
        k4 = self._func(cat_k4)
        return (k1 + 2*(k2 + k3) + k4) * dt * one_sixth

    def forward(self, x):
        y0 = x[:, 0:1] 
        dy = self._rk4(x)
        y1 = y0 + dy
        return y1
    

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
            model = NODE(num_neuron=num_neuron).to(device)
            model.load_state_dict(torch.load(save_path)) 
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            for epoch in range(params['epochs']):
                model.train()
                x_train, y_train = next(iter(train_loader)) 
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
model = NODE(num_neuron=param_grid[0]['num_neuron']).to(device)
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
    plt.plot(t_np, label ="prediction")
    plt.plot(y_test[:-num], label = "ground truth")
    plt.grid()
    plt.legend()
    plt.show()
    for i in range(1, num):
        model.eval()
        inputs = torch.cat([t_next, sup[i:i-num], air[i:i-num]], dim=1)
        print(inputs.size())
        t_next = model(inputs)
        t_np = t_next.detach().cpu().numpy()
        true_output = y_test[i:i-num]
        rmse_list.append(np.sqrt(mean_squared_error(true_output, t_np)) * 5.9)
        mae_list.append(mean_absolute_error(true_output, t_np) * 5.9)
        r2_list.append(r2_score(true_output, t_np))
    print("NODE: ")
    print(f"\nRMSE: [{', '.join(f'{x:.4f}' for x in rmse_list)}]")
    print(f"\nMAE: [{', '.join(f'{x:.4f}' for x in mae_list)}]")
    print(f"\nR²: [{', '.join(f'{x:.4f}' for x in r2_list)}]")
    return rmse_list, mae_list, r2_list

rmse_list, mae_list, r2_list = evaluate_test_data(model, X_t, output_t, num=3)
df = pd.DataFrame({
    'rmse': rmse_list,
    'mae': mae_list,
    'r2': r2_list
})
df = df.T
df.to_csv(f'results_node_{choice}_test_{N_end}.csv', index=True, header=True)

rmse_list, mae_list, r2_list = evaluate_test_data(model, X_tr, output_tr, num=1)
df = pd.DataFrame({
    'rmse': rmse_list,
    'mae': mae_list,
    'r2': r2_list
})
df = df.T
df.to_csv(f'results_node_{choice}_train_{N_end}.csv', index=True, header=True)