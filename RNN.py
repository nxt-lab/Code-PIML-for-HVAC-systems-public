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
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
number = 4   

def warm_up_training(dataset, num_neuron=4, num_epochs=5, lr=0.01, batch_size=32, save_path="warmup_model.pth"):
    warmup_model = Base(num_neuron=num_neuron).to(device)
    optimizer = torch.optim.AdamW(warmup_model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    for epoch in range(num_epochs):
        warmup_model.train()
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            warmup_model.train()
            x1, x2 , x3 = x[:, :1], x[:, 1:2], x[:, 2:]
            y1 = warmup_model(torch.cat([x1[0:0-number], x2[0:0-number], x3[0:0-number]], dim=1))
            y2 = warmup_model(torch.cat([y1, x2[1:1-number], x3[1:1-number]], dim=1))
            y3 = warmup_model(torch.cat([y2, x2[2:2-number], x3[2:2-number]], dim=1))
            y4 = warmup_model(torch.cat([y3, x2[3:3-number], x3[3:3-number]], dim=1))
            y5 = warmup_model(torch.cat([y4, x2[4:], x3[4:]], dim=1))
            
            loss_1 = loss_fn(y1, y[0:0-number])
            loss_2 = loss_fn(y2, y[1:1-number])
            loss_3 = loss_fn(y3, y[2:2-number])
            loss_4 = loss_fn(y4, y[3:3-number])
            loss_5 = loss_fn(y5, y[4:])
            loss = 1/5*(loss_1+loss_2+loss_3+loss_4+loss_5)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Warm-Up Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.6f}")
    torch.save(warmup_model.state_dict(), save_path)
    print(f"\nWarm-up complete. Model saved to {save_path}")

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
            model.load_state_dict(torch.load(save_path))  # Load the pre-trained model
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            for epoch in range(params['epochs']):
                model.train()
                x_train, y_train = next(iter(train_loader))  # Only 1 batch
                x_train, y_train = x_train.to(device), y_train.to(device)
                optimizer.zero_grad()
                y = y_train
                x1, x2 , x3 = x_train[:, :1], x_train[:, 1:2], x_train[:, 2:]
                y1 = model(torch.cat([x1[0:0-number], x2[0:0-number], x3[0:0-number]], dim=1))
                y2 = model(torch.cat([y1, x2[1:1-number], x3[1:1-number]], dim=1))
                y3 = model(torch.cat([y2, x2[2:2-number], x3[2:2-number]], dim=1))
                y4 = model(torch.cat([y3, x2[3:3-number], x3[3:3-number]], dim=1))
                y5 = model(torch.cat([y4, x2[4:], x3[4:]], dim=1))
                
                loss_1 = loss_fn(y1, y[0:0-number])
                loss_2 = loss_fn(y2, y[1:1-number])
                loss_3 = loss_fn(y3, y[2:2-number])
                loss_4 = loss_fn(y4, y[3:3-number])
                loss_5 = loss_fn(y5, y[4:])
                
                loss = 1/5*(loss_1+loss_2+loss_3+loss_4+loss_5)
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
number = 4
for i in range(epoch):
    model.train()
    optimizer.zero_grad()
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    loss_fn = nn.MSELoss() 
    x = X_tr
    y = output_tr
    x1, x2 , x3 = x[:, :1], x[:, 1:2], x[:, 2:]
    y1 = model(torch.cat([x1[0:0-number], x2[0:0-number], x3[0:0-number]], dim=1))
    y2 = model(torch.cat([y1, x2[1:1-number], x3[1:1-number]], dim=1))
    y3 = model(torch.cat([y2, x2[2:2-number], x3[2:2-number]], dim=1))
    y4 = model(torch.cat([y3, x2[3:3-number], x3[3:3-number]], dim=1))
    y5 = model(torch.cat([y4, x2[4:], x3[4:]], dim=1))
    
    loss_fn = nn.MSELoss() 
    loss_1 = loss_fn(y1, y[0:0-number])
    loss_2 = loss_fn(y2, y[1:1-number])
    loss_3 = loss_fn(y3, y[2:2-number])
    loss_4 = loss_fn(y4, y[3:3-number])
    loss_5 = loss_fn(y5, y[4:])
    loss = 1/5*(loss_1+loss_2+loss_3+loss_4+loss_5)
    loss = loss + param_grid[0]['l1_lambda'] * l1_norm
    
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
        rmse_list.append(np.sqrt(mean_squared_error(true_output, t_np)) * 5.9)
        mae_list.append(mean_absolute_error(true_output, t_np) * 5.9)
        r2_list.append(r2_score(true_output, t_np))
    print("Baseline:")
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
df.to_csv(f'results_rnn_{choice}_test_{N_end}.csv', index=True, header=True)


rmse_list, mae_list, r2_list = evaluate_test_data(model, X_tr, output_tr, num=1)
df = pd.DataFrame({
    'rmse': rmse_list,
    'mae': mae_list,
    'r2': r2_list
})
df = df.T
df.to_csv(f'results_rnn_{choice}_train_{N_end}.csv', index=True, header=True)
