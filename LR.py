import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

choice = input("Choose dataset [filtered/original]: ").strip().lower()

if choice == "filtered":
    relative_path = os.path.join("filtered_room_104.csv")
    Ns_t, Ne_t = 180, 190
elif choice == "original":
    relative_path = os.path.join("HVAC_B90_r104_exp_30m_20210727_15_min.csv")
    Ns_t, Ne_t = 530, 560

data = pd.read_csv(relative_path)
start = 1
room = data["r104_room_temp"]
airflow = data["r104_airflow_current"]
supply = data["r104_supply_discharge_temp"]
temp = np.array(room[start:-1]).reshape(-1,1)
temp_out = data["r104_thermostat_outside_temp"]
temp_out = np.zeros_like(temp)
output = np.array(room[start+1:]).reshape(-1,1)
air = np.array(airflow[start+1:]).reshape(-1,1)
supply = np.array(supply[start+1:]).reshape(-1,1)

air = (air - min(air))/(max(air)- min(air))
supply = (supply - min(supply))/(max(supply) - min(supply))
temp = (temp - min(temp))/(max(temp) - min(temp))
output = (output - min(output))/(max(output) - min(output))


def data_split(N, N_end, temp, supply, air, temp_out, output):
    temp_tr = temp[N:N_end]
    supply_tr = supply[N:N_end]
    air_tr = air[N:N_end]
    temp_o_tr = temp_out[N:N_end]
    output_tr = output[N:N_end]
    return temp_tr, supply_tr, air_tr, temp_o_tr, output_tr

N_end = int(input("Enter number of training samples (e.g., 100): "))
N_start = 1 
temp_tr, supply_tr, air_tr, temp_o_tr, output_tr =  data_split(N_start, N_end, temp, supply, air, temp_out, output)
temp_t, supply_t, air_t, temp_o_t, output_t = data_split(Ns_t, Ne_t, temp, supply, air, temp_out, output)

def cal_rmse(mse):
    if isinstance(mse, list):
        mse = np.array(mse)
    mse_F = np.sqrt(mse) * 5.9
    if isinstance(mse, np.ndarray):
        return mse_F.tolist()
    return mse_F

in_tr = np.concatenate((temp_tr, supply_tr, air_tr, temp_o_tr), axis=1)
model = LinearRegression()
model.fit(in_tr, output_tr)
print("model coeficient:", model.coef_)

def evaluate(num, temp_t, supply_t, air_t, temp_o_t, output_t):
    in_t_next = np.concatenate((temp_t[:-num].reshape(-1, 1), supply_t[:-num].reshape(-1, 1), air_t[:-num].reshape(-1, 1), temp_o_t[:-num].reshape(-1, 1)), axis=1)
    t_next = model.predict(in_t_next)
    r2 = []
    mse = []
    mae = [] 
    r2.append(r2_score(output_t[:-num], t_next))
    mse.append(mean_squared_error(output_t[:-num], t_next))
    mae.append(mean_absolute_error(output_t[:-num], t_next)*5.9)
    
    for i in range(1, num):
        in_t_next = np.concatenate((t_next.reshape(-1, 1), supply_t[i:i-num].reshape(-1, 1), air_t[i:i-num].reshape(-1, 1), temp_o_t[i:i-num].reshape(-1, 1)), axis=1)
        t_next =  model.predict(in_t_next)
        R2 = r2_score(output_t[i:i-num], t_next)
        MSE = mean_squared_error(output_t[i:i-num], t_next)
        MAE = mean_absolute_error(output_t[i:i-num], t_next)* 5.9
        r2.append(R2)
        mse.append(MSE)
        mae.append(MAE)
    rmse = cal_rmse(mse)
    return rmse, mae, r2

num = 3
rmse, mae, r2 = evaluate(num, temp_t, supply_t, air_t, temp_o_t, output_t)
df = pd.DataFrame({
    'rmse': rmse,
    'mae': mae,
    'r2': r2
})
# Save the DataFrame to a CSV file
df = df.T
df.to_csv(f'results_lr_{choice}_test_{N_end}.csv', index=True, header=True)
print(df)

rmse, mae, r2 = evaluate(1, temp_tr, supply_tr, air_tr, temp_o_tr, output_tr)
df = pd.DataFrame({
    'rmse': rmse,
    'mae': mae,
    'r2': r2
})
df = df.T
df.to_csv(f'results_lr_{choice}_train_{N_end}.csv', index=True, header=True)