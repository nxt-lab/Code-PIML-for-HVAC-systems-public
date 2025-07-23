import torch
import numpy as np
import pandas as pd
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

def data_split(N, N_end, temp, supply, air, output):
    return temp[N:N_end], supply[N:N_end], air[N:N_end], output[N:N_end]
def normalize_tensor(t):
    return (t - t.min()) / (t.max() - t.min() + 1e-8)

def data_split(N, N_end, temp, supply, air, output):
    return temp[N:N_end], supply[N:N_end], air[N:N_end], output[N:N_end]

def load_hvac_data(choice="filtered", start=1, train_amount=100):
    if choice == "filtered":
        relative_path = os.path.join("filtered_room_104.csv")
        Ns_t, Ne_t = 180, 190
    elif choice == "original":
        relative_path = os.path.join("HVAC_B90_r104_exp_30m_20210727_15_min.csv")
        Ns_t, Ne_t = 530, 560
    else:
        raise ValueError("Invalid choice: should be 'filtered' or 'original'")

    data = pd.read_csv(relative_path)
    room = data["r104_room_temp"]
    airflow = data["r104_airflow_current"]
    supply_temp = data["r104_supply_discharge_temp"]

    temp = torch.tensor(np.array(room[start:-1]).reshape(-1, 1), device=device, dtype=torch.float32)
    output = torch.tensor(np.array(room[start + 1:]).reshape(-1, 1), device=device, dtype=torch.float32)
    air = torch.tensor(np.array(airflow[start + 1:]).reshape(-1, 1), device=device, dtype=torch.float32)
    supply = torch.tensor(np.array(supply_temp[start + 1:]).reshape(-1, 1), device=device, dtype=torch.float32)

    # Normalize all tensors
    temp, output, air, supply = map(normalize_tensor, [temp, output, air, supply])

    return temp, supply, air, output, train_amount, Ns_t, Ne_t

def generate_param_grid():
    param_grid = []
    for lr in np.arange(0.01, 0.4, 0.01):
        for num_neuron in [3]:
            param_grid.append({'num_neuron': num_neuron, 'epochs': 500, 'lr': lr})
    return param_grid

def normalize_tensor(t):
    return (t - t.min()) / (t.max() - t.min() + 1e-8)