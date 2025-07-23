import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
import torch.nn.functional as F
from scipy.integrate import odeint
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device

start = 1
choice = input("Choose dataset [filtered/raw]: ").strip().lower()
train_amount = int(input("Enter number of training samples (e.g., 100): "))

if choice == "filtered":
    relative_path = os.path.join("filtered_room_104.csv")
    Ns_t, Ne_t = 180, 190
elif choice == "original":
    relative_path = os.path.join("HVAC_B90_r104_exp_30m_20210727_15_min.csv")
    Ns_t, Ne_t = 530, 560

data = pd.read_csv(relative_path)
room = data["r104_room_temp"]
room = (room - 32)*5/9
airflow = data["r104_airflow_current"]*0.0283168
supply = (data["r104_supply_discharge_temp"]- 32)*5/9
temp = np.array(room[start:-1]).reshape(-1,1)
output = np.array(room[start+1:]).reshape(-1,1)
air = np.array(airflow[start+1:]).reshape(-1,1)
supply = np.array(supply[start+1:]).reshape(-1,1)

air_ori = torch.tensor(air).to(device).reshape(-1,1).float()
supply_ori = torch.tensor(supply).to(device).reshape(-1,1).float()
temp_ori = torch.tensor(temp).to(device).reshape(-1,1).float()
output_ori = torch.tensor(output).to(device).reshape(-1,1).float()

# Output
output_numpy = output.reshape(-1, 1)
output = torch.tensor(output_numpy).to(device).float()

air = air_ori.requires_grad_(True)
supply = supply_ori.requires_grad_(True)
temp = temp_ori.requires_grad_(True)


def data_split(N: int, N_e: int, temp: torch.Tensor, supply: torch.Tensor, air: torch.Tensor, output: torch.Tensor):
    temp_tr = temp[N:N_e]
    supply_tr = supply[N:N_e]
    air_tr = air[N:N_e]
    output_tr = output[N:N_e]
    return temp_tr, supply_tr, air_tr, output_tr


N_s, N_e = 1, train_amount  # Training range

temp_tr, supply_tr, air_tr, output_tr =  data_split(N_s, N_e, temp, supply, air, output)
temp_t, supply_t, air_t, output_t = data_split(Ns_t, Ne_t, temp, supply, air, output)

inputs_tr = torch.cat((supply_tr, air_tr), dim = 1)
inputs_t = torch.cat((supply_t, air_t), dim = 1)

def RC_func(y, t, u):
    C = 30.0218
    K = 1.4850
    Ta =  21.3846
    ca = 0.0847
    d = 2.0809
    u1 = u[0]
    u2 = u[1]
    dT = 1/C *(K * (Ta - y) + u2 * ca * (u1 - y) + d)
    return dT

T_next = []
T_solv_0 = []

for i in range(len(temp_tr)):
    x0 = temp_tr[i].to("cpu").detach().numpy() 
    t = [0, 15]                  
    u = inputs_tr[i].to("cpu").detach().numpy() 
    sol_2 = odeint(RC_func, x0, t, args=(u,))
    T_next.append(sol_2[1])  
    T_solv_0.append(sol_2[0])


T_next = np.array(T_next)
T_solv_0 = np.array(T_solv_0)


num = 3
def eval_func(RC_func, temp_t, inputs_t, output_t):
    r2_loop = []
    rmse_loop = []
    mae_loop = []
    for j in range(num):
        T_next = []
        for i in range(len(temp_t[j:j-num])):
            if j == 0:
                x0 = temp_t[i+j].to("cpu").detach().numpy()
            else:
                x0 = T_next_after[i]
            t = [0, 15]                 
            u = inputs_t[i+j].to("cpu").detach().numpy() 
            
            sol_2 = odeint(RC_func, x0, t, args=(u,))
            T_next.append(sol_2[1])   
        T_next_after = np.array(T_next)
       
        r2_loop.append(r2_score(output_t[j:j-num].to("cpu").detach().numpy(), T_next))
        rmse_loop.append(np.sqrt(mean_squared_error(output_t[j:j-num].to("cpu").detach().numpy(), T_next))*9/5)
        mae_loop.append(mean_absolute_error(output_t[j:j-num].to("cpu").detach().numpy(), T_next)*9/5)
    return rmse_loop, mae_loop, r2_loop


rmse, mae, r2 = eval_func(RC_func, temp_t, inputs_t, output_t)
df = pd.DataFrame({
    'rmse': rmse,
    'mae': mae,
    'r2': r2
})

df = df.T
df.to_csv(f'results_rc_{choice}_test_{N_e}.csv', index=True, header=True)
print(df)
num = 1
rmse, mae, r2 = eval_func(RC_func, temp_tr, inputs_tr, output_tr)
df = pd.DataFrame({
    'rmse': rmse,
    'mae': mae,
    'r2': r2
})
df = df.T

df.to_csv(f'results_rc_{choice}_train_{N_e}.csv', index=True, header=True)
print(df)