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

class Base(nn.Module):
    def __init__(self, num_neuron=4, dropout_p=0):
        super(Base, self).__init__()
        self.fc1 = nn.Linear(3, num_neuron)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(num_neuron, 1)
        self.initialize_weights()
    def initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    def forward(self, x):
        cur = x[:, :1].reshape(-1, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x + cur

def choose_lr(dataset, param_grid, n_splits=5):
    lr_results = {}
    loss_fn = nn.MSELoss()
    warmed_up_neurons = {} 
    kf = KFold(n_splits, shuffle=False)
    for params in param_grid:
        num_neuron, lr = params['num_neuron'], params['lr']
        save_path = ini_weights
        print(f"Evaluating num_neuron={num_neuron}, lr={lr}")
        avg_val_losses, avg_train_losses = [], []

        for train_idx, val_idx in kf.split(dataset):
            train_loader = DataLoader(Subset(dataset, train_idx), batch_size=len(train_idx), shuffle=False)
            val_loader = DataLoader(Subset(dataset, val_idx), batch_size=len(val_idx), shuffle=False)
            model = Base(num_neuron=num_neuron).to(device)
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
model = Base(num_neuron=param_grid[0]['num_neuron']).to(device)
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
    # Initial prediction using the first batch
    inputs = torch.cat([temp[:-num], sup[:-num], air[:-num]], dim=1)
    t_next = model(inputs)
    t_np = t_next.detach().cpu().numpy()
    y_test = y_test.detach().cpu().numpy()
    rmse_list.append(np.sqrt(mean_squared_error(y_test[:-num], t_np)) * 5.9)
    mae_list.append(mean_absolute_error(y_test[:-num], t_np) * 5.9)
    r2_list.append(r2_score(y_test[:-num], t_np))
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
    print("Residual Neural Network:")
    print(f"\nRMSE: [{', '.join(f'{x:.4f}' for x in rmse_list)}]")
    print(f"\nMAE: [{', '.join(f'{x:.4f}' for x in mae_list)}]")
    print(f"\nR²: [{', '.join(f'{x:.4f}' for x in r2_list)}]")
    return rmse_list, mae_list, r2_list


rmse, mae, r2 = evaluate_test_data(model, X_t, output_t, num=3)
df = pd.DataFrame({
    'rmse': rmse,
    'mae': mae,
    'r2': r2
})
df = df.T
df.to_csv(f'results_res_{choice}_test_{N_end}.csv', index=True, header=True)
rmse, mae, r2 = evaluate_test_data(model, X_tr, output_tr, num=1)
df = pd.DataFrame({
    'rmse': rmse,
    'mae': mae,
    'r2': r2
})
df = df.T
df.to_csv(f'results_res_{choice}_train_{N_end}.csv', index=True, header=True)