import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import random

# ========================
# Setup
# ========================
SEED = 88
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)

# ========================
# Helper Functions
# ========================
def compute_rmse(y_pred, y_true):
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))

def create_IO_data(u, y, na, nb):
    X, Y = [], []
    for k in range(max(na, nb), len(y)):
        X.append(np.concatenate([u[k-nb:k], y[k-na:k]]))
        Y.append(y[k])
    return np.array(X), np.array(Y)

def use_NARX_model_in_simulation(ulist, ylist, model, na, nb):
    ylist = ylist[:50]
    upast = ulist[50-nb:50]
    ypast = ylist[-na:]

    for unow in ulist[50:]:
        x = torch.tensor(upast + ypast, dtype=torch.float32).unsqueeze(0)
        out = model(x)
        y_new = out.squeeze().cpu().item()
        upast.append(unow)
        upast.pop(0)
        ypast.append(y_new)
        ypast.pop(0)
        ylist.append(y_new)

    return torch.tensor(ylist)

def normalize(x, mean, std):
    return (x - mean) / std

def unnormalize(x, mean, std):
    return x * std + mean

# ========================
# NARX Model Definition
# ========================
class NARX(nn.Module):
    def __init__(self, input_dim, hidden_neuron, hidden_layers, activation):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_neuron] * hidden_layers + [1]
        self.layers = nn.ModuleList([
            nn.Linear(in_f, out_f) for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:])
        ])
        self.act = activation()

    def forward(self, x):
        for lin in self.layers[:-1]:
            x = self.act(lin(x))
        return self.layers[-1](x)

# ========================
# LSTM Model Definition
# ========================
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional=False, dropout=0.0):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0).double()
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction_factor, output_size).double()

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ========================
# Load and Preprocess Data
# ========================
print("Loading and preprocessing data...")
data = np.load('disc-benchmark-files/training-val-test-data.npz')
th = data['th']
u = data['u']

dt = 0.025
omega = np.gradient(th, dt)

theta_mean, theta_std = th.mean(), th.std()
omega_mean, omega_std = omega.mean(), omega.std()
u_mean, u_std = u.mean(), u.std()

th_norm = normalize(th, theta_mean, theta_std)
omega_norm = normalize(omega, omega_mean, omega_std)
u_norm = normalize(u, u_mean, u_std)

# ========================
# Train NARX Models
# ========================
print("\n=== Training NARX Models ===")
batch_size = 256
epochs = 400
learning_rate = 1e-3

best_rmse_narx = float("inf")
best_model_narx = None
best_config_narx = None

for na in [5, 8]:
    for nb in [3, 5]:
        X_train_np, y_train_np = create_IO_data(u[:int(0.8*len(th))], th[:int(0.8*len(th))], na, nb)
        X_val_np, y_val_np = create_IO_data(u[int(0.8*len(th)):int(0.9*len(th))], th[int(0.8*len(th)):int(0.9*len(th))], na, nb)

        X_train = torch.tensor(X_train_np, dtype=torch.float32)
        y_train = torch.tensor(y_train_np, dtype=torch.float32)
        X_val = torch.tensor(X_val_np, dtype=torch.float32)
        y_val = torch.tensor(y_val_np, dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

        for act_name, activation in [("tanh", nn.Tanh), ("relu", nn.ReLU)]:
            for hl in [1, 2]:
                for hn in [32, 64]:
                    model_narx = NARX(input_dim=na+nb, hidden_neuron=hn, hidden_layers=hl, activation=activation).to(device)
                    opt = optim.Adam(model_narx.parameters(), lr=learning_rate)
                    criterion = nn.MSELoss()
                    for epoch in range(epochs):
                        model_narx.train()
                        for xb, yb in train_loader:
                            xb, yb = xb.to(device), yb.to(device)
                            opt.zero_grad()
                            loss = criterion(model_narx(xb).squeeze(), yb)
                            loss.backward()
                            opt.step()
                    model_narx.eval()
                    with torch.no_grad():
                        preds = model_narx(X_val.to(device)).squeeze()
                        rmse_narx = compute_rmse(preds, y_val.to(device)).item()
                    print(f"NARX na:{na} nb:{nb} hl:{hl} hn:{hn} act:{act_name} --> RMSE: {rmse_narx:.5f}")
                    if rmse_narx < best_rmse_narx:
                        best_rmse_narx = rmse_narx
                        best_model_narx = model_narx
                        best_config_narx = (na, nb)

torch.save(best_model_narx.state_dict(), "disc-submission-files/ann-narx-model.pth")

# ========================
# Simulate NARX Model
# ========================
print("\nSimulating best NARX model...")
na, nb = best_config_narx
model = best_model_narx
model.eval()
test_data = np.load('disc-benchmark-files/hidden-test-simulation-submission-file.npz')
u_test = list(test_data['u'])
th_test = list(test_data['th'])

th_test_sim = use_NARX_model_in_simulation(u_test, th_test, model, na, nb).numpy()
np.savez('disc-submission-files/ann-narx-hidden-test-simulation-submission-file.npz', th=th_test_sim, u=np.array(u_test))

# ========================
# Train LSTM Model
# ========================
print("\n=== Training LSTM Models ===")

def create_lstm_sequences(theta, omega, u, seq_len):
    X, Y = [], []
    for i in range(len(theta) - seq_len):
        x_seq = np.stack([theta[i:i+seq_len], omega[i:i+seq_len], u[i:i+seq_len]], axis=1)
        y_target = [theta[i+seq_len], omega[i+seq_len]]
        X.append(x_seq)
        Y.append(y_target)
    return np.array(X), np.array(Y)

X_lstm, Y_lstm = create_lstm_sequences(th_norm, omega_norm, u_norm, seq_len=20)
total = len(X_lstm)
X_train, Y_train = X_lstm[:int(0.8*total)], Y_lstm[:int(0.8*total)]
X_test, Y_test = X_lstm[int(0.8*total):], Y_lstm[int(0.8*total):]

X_train = torch.tensor(X_train).double().to(device)
Y_train = torch.tensor(Y_train).double().to(device)
X_test = torch.tensor(X_test).double().to(device)
Y_test = torch.tensor(Y_test).double().to(device)

best_rmse_lstm = float("inf")
best_model_lstm = None
best_config_lstm = None

for hidden_size in [32, 64, 128]:
    for num_layers in [1, 2]:
        for bidirectional in [False, True]:
            model_lstm = LSTM(input_size=3, hidden_size=hidden_size, num_layers=num_layers,
                                 output_size=2, bidirectional=bidirectional).to(device)
            optimizer = optim.Adam(model_lstm.parameters(), lr=1e-3)
            criterion = nn.MSELoss()

            for epoch in range(100):
                model_lstm.train()
                optimizer.zero_grad()
                output = model_lstm(X_train)
                loss = criterion(output, Y_train)
                loss.backward()
                optimizer.step()

            model_lstm.eval()
            with torch.no_grad():
                pred = model_lstm(X_test)
                rmse_lstm = compute_rmse(pred, Y_test).item()

            print(f"hs:{hidden_size} nl:{num_layers} bi:{int(bidirectional)} → RMSE: {rmse_lstm:.5f}")

            if rmse_lstm < best_rmse_lstm:
                best_rmse_lstm = rmse_lstm
                best_model_lstm = model_lstm
                best_config_lstm = (hidden_size, num_layers, bidirectional)

print(f"Best LSTM → hs:{best_config_lstm[0]} nl:{best_config_lstm[1]} bi:{int(best_config_lstm[2])} | RMSE: {best_rmse_lstm:.5f}")
torch.save(best_model_lstm.state_dict(), "disc-submission-files/ann-lstm-model.pth")

# ========================
# Simulate LSTM Model
# ========================
print("Saving LSTM .npz output...")
input_data = np.load("disc-benchmark-files/hidden-test-simulation-submission-file.npz")
u_hidden = input_data["u"]
th_hidden = input_data["th"]
omega_hidden = np.gradient(th_hidden, dt)

th_hidden_norm = normalize(th_hidden, theta_mean, theta_std)
omega_hidden_norm = normalize(omega_hidden, omega_mean, omega_std)
u_hidden_norm = normalize(u_hidden, u_mean, u_std)

X_hidden, _ = create_lstm_sequences(th_hidden_norm, omega_hidden_norm, u_hidden_norm, seq_len=20)
X_hidden_tensor = torch.tensor(X_hidden).double().to(device)

with torch.no_grad():
    preds = model_lstm(X_hidden_tensor).cpu().numpy()
    preds_unnorm = np.empty_like(preds)
    preds_unnorm[:, 0] = unnormalize(preds[:, 0], theta_mean, theta_std)
    preds_unnorm[:, 1] = unnormalize(preds[:, 1], omega_mean, omega_std)

pred_theta = np.concatenate([th_hidden[:20], preds_unnorm[:, 0]])
np.savez("disc-submission-files/ann-lstm-hidden-test-simulation-submission-file.npz", th=pred_theta, u=u_hidden)