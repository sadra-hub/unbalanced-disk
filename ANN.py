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

os.makedirs("disc-submission-files", exist_ok=True)

# ========================
# Helper Functions
# ========================
def compute_rmse(y_pred, y_true):
    y_pred = torch.as_tensor(y_pred, dtype=torch.float32, device="cpu")
    y_true = torch.as_tensor(y_true, dtype=torch.float32, device="cpu")
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()

def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def compute_wrapped_rmse(y_pred, y_true):
    yp = np.asarray(y_pred).reshape(-1)
    yt = np.asarray(y_true).reshape(-1)
    err = wrap_angle(yp - yt)
    return float(np.sqrt(np.mean(err ** 2)))

def create_IO_data_sincos(u, y, na, nb):
    """Build NARX IO with sin/cos(theta) to avoid wrap discontinuity."""
    X, Y = [], []
    s, c = np.sin(y), np.cos(y)
    for k in range(max(na, nb), len(y)):
        u_block = u[k - nb:k]
        s_block = s[k - na:k]
        c_block = c[k - na:k]
        X.append(np.concatenate([u_block, s_block, c_block]))
        Y.append(y[k])  # keep target as angle; use wrapped RMSE for evaluation
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def use_NARX_model_in_simulation(ulist, ylist, model, na, nb):
    """
    Free-run simulation with sin/cos features.
    - Warm-up with first 50 y samples from ground truth.
    - Model predicts angle; we feed sin/cos of predicted angle back.
    """
    ylist = list(ylist[:50])
    upast = list(ulist[50 - nb:50])
    ypast = list(ylist[-na:])

    model.eval()
    with torch.no_grad():
        for unow in ulist[50:]:
            sin_block = np.sin(np.array(ypast[-na:], dtype=np.float32))
            cos_block = np.cos(np.array(ypast[-na:], dtype=np.float32))
            x_vec = np.concatenate([np.array(upast[-nb:], dtype=np.float32), sin_block, cos_block], axis=0)
            x = torch.tensor(x_vec, dtype=torch.float32, device=device).unsqueeze(0)

            y_new = model(x).squeeze().float().cpu().item()  # predicted angle
            upast.append(float(unow)); upast.pop(0)
            ypast.append(y_new);       ypast.pop(0)
            ylist.append(y_new)

    return torch.tensor(ylist, dtype=torch.float32)

def normalize(x, mean, std):
    return (x - mean) / (std + 1e-12)

def unnormalize(x, mean, std):
    return x * (std + 1e-12) + mean

def try_load(path):
    return np.load(path) if os.path.exists(path) else None

# Hidden helpers (robust keys)
def get_first_key(dct, candidates):
    for k in candidates:
        if k in dct:
            return k
    return None

def narx_hidden_prediction_rmse(model, na, nb, pred_npz):
    """
    hidden-test-prediction-submission-file.npz must contain:
      - upast: (N, 15)
      - thpast: (N, 15)
      - ground truth: one of ['thnow', 'thnow_true', 'y', 'y_true', 'target']
    Returns WRAPPED RMSE.
    """
    if pred_npz is None:
        return None
    if ('upast' not in pred_npz) or ('thpast' not in pred_npz):
        return None

    upast = pred_npz['upast'].astype(np.float32)
    thpast = pred_npz['thpast'].astype(np.float32)
    gt_key = get_first_key(pred_npz, ['thnow_true', 'thnow', 'y_true', 'y', 'target'])
    if gt_key is None:
        return None
    y_true = pred_npz[gt_key].astype(np.float32).reshape(-1)

    if upast.shape[1] < nb or thpast.shape[1] < na:
        return None  # not enough lags

    u_slice  = upast[:, -nb:] if nb > 0 else np.zeros((upast.shape[0], 0), dtype=np.float32)
    s_slice  = np.sin(thpast[:, -na:]) if na > 0 else np.zeros((upast.shape[0], 0), dtype=np.float32)
    c_slice  = np.cos(thpast[:, -na:]) if na > 0 else np.zeros((upast.shape[0], 0), dtype=np.float32)
    X_hidden = np.concatenate([u_slice, s_slice, c_slice], axis=1).astype(np.float32)

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_hidden, dtype=torch.float32, device=device)).squeeze().float().cpu().numpy()
    return compute_wrapped_rmse(preds, y_true)

def narx_hidden_simulation_rmse(model, na, nb, sim_npz):
    """
    hidden-test-simulation-submission-file.npz must contain:
      - u: (T,)
      - th: (T,) ground-truth trajectory
    Returns WRAPPED RMSE.
    """
    if sim_npz is None:
        return None
    # Accept alternate keys
    u_key  = get_first_key(sim_npz, ['u','u_sequence','u_valid','u_test'])
    th_key = get_first_key(sim_npz, ['th','th_true','y','y_true'])
    if u_key is None or th_key is None:
        return None

    u_seq = sim_npz[u_key].astype(np.float32).reshape(-1)
    th_true = sim_npz[th_key].astype(np.float32).reshape(-1)
    if len(u_seq) < 50 or len(th_true) < 50:
        return None

    th_sim = use_NARX_model_in_simulation(list(u_seq), list(th_true), model, na, nb).cpu().numpy()
    L = min(len(th_sim), len(th_true))
    return compute_wrapped_rmse(th_sim[:L], th_true[:L])

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
th = data['th'].astype(np.float32)
u = data['u'].astype(np.float32)

dt = 0.025
omega = np.gradient(th, dt).astype(np.float32)

theta_mean, theta_std = th.mean(), th.std()
omega_mean, omega_std = omega.mean(), omega.std()
u_mean, u_std = u.mean(), u.std()

th_norm = normalize(th, theta_mean, theta_std)
omega_norm = normalize(omega, omega_mean, omega_std)
u_norm = normalize(u, u_mean, u_std)

# Hidden files (for per-iteration RMSE)
hidden_pred_npz = try_load('disc-benchmark-files/hidden-test-prediction-submission-file.npz')
hidden_sim_npz  = try_load('disc-benchmark-files/hidden-test-simulation-submission-file.npz')
if hidden_pred_npz is None:
    print("⚠ Hidden prediction file not found; will skip hidden prediction RMSE.")
if hidden_sim_npz is None:
    print("⚠ Hidden simulation file not found; will skip hidden simulation RMSE.")

# ========================
# Train NARX Models (sin/cos inputs + wrapped RMSE)
# ========================
print("\n=== Training NARX Models (with per-iteration RMSE logs) ===")
batch_size = 256
epochs = 400
learning_rate = 1e-3

best_rmse_narx = float("inf")
best_model_narx = None
best_config_narx = None

for na in [5, 8]:
    for nb in [3, 5]:
        X_train_np, y_train_np = create_IO_data_sincos(u[:int(0.8*len(th))], th[:int(0.8*len(th))], na, nb)
        X_val_np, y_val_np = create_IO_data_sincos(u[int(0.8*len(th)):int(0.9*len(th))], th[int(0.8*len(th)):int(0.9*len(th))], na, nb)

        X_train = torch.tensor(X_train_np, dtype=torch.float32)
        y_train = torch.tensor(y_train_np, dtype=torch.float32)
        X_val = torch.tensor(X_val_np, dtype=torch.float32)
        y_val = torch.tensor(y_val_np, dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

        for act_name, activation in [("tanh", nn.Tanh), ("relu", nn.ReLU)]:
            for hl in [1, 2]:
                for hn in [32, 64]:
                    input_dim = nb + 2 * na  # sin/cos features
                    model_narx = NARX(input_dim=input_dim, hidden_neuron=hn, hidden_layers=hl, activation=activation).to(device)
                    opt = optim.Adam(model_narx.parameters(), lr=learning_rate)
                    criterion = nn.MSELoss()
                    # Train
                    for epoch in range(epochs):
                        model_narx.train()
                        for xb, yb in train_loader:
                            xb, yb = xb.to(device), yb.to(device)
                            opt.zero_grad()
                            loss = criterion(model_narx(xb).squeeze(), yb)
                            loss.backward()
                            opt.step()
                    # Validation prediction RMSE (plain + wrapped)
                    model_narx.eval()
                    with torch.no_grad():
                        preds_val = model_narx(X_val.to(device)).squeeze().float().cpu().numpy()
                        rmse_val_pred_plain = compute_rmse(preds_val, y_val.numpy())
                        rmse_val_pred_wrap  = compute_wrapped_rmse(preds_val, y_val.numpy())

                    # Hidden prediction RMSE (one-step, WRAPPED)
                    rmse_hidden_pred = narx_hidden_prediction_rmse(model_narx, na, nb, hidden_pred_npz)

                    # Hidden simulation RMSE (free-run, WRAPPED)
                    rmse_hidden_sim = narx_hidden_simulation_rmse(model_narx, na, nb, hidden_sim_npz)

                    print(f"NARX na:{na} nb:{nb} hl:{hl} hn:{hn} act:{act_name} "
                          f"→ ValPredRMSE: {rmse_val_pred_plain:.5f} (wrap:{rmse_val_pred_wrap:.5f}) "
                          f"| HiddenPredRMSE(wrap): {('NA' if rmse_hidden_pred is None else f'{rmse_hidden_pred:.5f}')} "
                          f"| HiddenSimRMSE(wrap): {('NA' if rmse_hidden_sim is None else f'{rmse_hidden_sim:.5f}')}")

                    # Track best by hidden simulation if available, else by wrapped val pred
                    score = rmse_hidden_sim if rmse_hidden_sim is not None else rmse_val_pred_wrap
                    if score < best_rmse_narx:
                        best_rmse_narx = score
                        best_model_narx = model_narx
                        best_config_narx = (na, nb, hl, hn, act_name)

# Save best NARX
if best_model_narx is not None:
    torch.save(best_model_narx.state_dict(), "disc-submission-files/ann-narx-model.pth")
    print(f"\nBest NARX config: na:{best_config_narx[0]} nb:{best_config_narx[1]} hl:{best_config_narx[2]} "
          f"hn:{best_config_narx[3]} act:{best_config_narx[4]} | Score (prefers HiddenSim wrap): {best_rmse_narx:.5f}")

# ========================
# Simulate & export best NARX on hidden set
# ========================
if best_model_narx is not None and hidden_sim_npz is not None:
    print("\nSimulating best NARX model on hidden simulation set...")
    na, nb = best_config_narx[0], best_config_narx[1]
    model = best_model_narx
    model.eval()
    # choose robust keys
    u_key  = get_first_key(hidden_sim_npz, ['u','u_sequence','u_valid','u_test'])
    th_key = get_first_key(hidden_sim_npz, ['th','th_true','y','y_true'])
    u_test = list(hidden_sim_npz[u_key])
    th_test = list(hidden_sim_npz[th_key])
    th_test_sim = use_NARX_model_in_simulation(u_test, th_test, model, na, nb).numpy()
    np.savez('disc-submission-files/ann-narx-hidden-test-simulation-submission-file.npz',
             th=th_test_sim, u=np.array(u_test))
    print("Saved NARX hidden simulation submission npz.")

# ========================
# Train LSTM Model (SEQ_LEN=15; with per-iteration HiddenPredRMSE + HiddenSimRMSE)
# ========================
print("\n=== Training LSTM Models (SEQ_LEN=15) ===")

def create_lstm_sequences(theta, omega, u, seq_len):
    X, Y = [], []
    for i in range(len(theta) - seq_len):
        x_seq = np.stack([theta[i:i+seq_len], omega[i:i+seq_len], u[i:i+seq_len]], axis=1)
        y_target = [theta[i+seq_len], omega[i+seq_len]]
        X.append(x_seq)
        Y.append(y_target)
    return np.array(X, dtype=np.float64), np.array(Y, dtype=np.float64)

SEQ_LEN = 15
X_lstm, Y_lstm = create_lstm_sequences(th_norm, omega_norm, u_norm, seq_len=SEQ_LEN)
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

def lstm_hidden_prediction_rmse(model_lstm):
    """Use hidden-test-prediction-submission-file.npz for one-step LSTM prediction."""
    if hidden_pred_npz is None:
        return None
    if ('upast' not in hidden_pred_npz) or ('thpast' not in hidden_pred_npz):
        return None
    upast = hidden_pred_npz['upast'].astype(np.float32)  # (N, 15)
    thpast = hidden_pred_npz['thpast'].astype(np.float32)  # (N, 15)
    gt_key = get_first_key(hidden_pred_npz, ['thnow_true', 'thnow', 'y_true', 'y', 'target'])
    if gt_key is None:
        return None
    th_now_true = hidden_pred_npz[gt_key].astype(np.float32).reshape(-1)

    # Build omega from thpast (finite difference)
    omega_past = np.gradient(thpast, axis=1) / dt  # same shape (N, 15)

    # Normalize using training stats
    th_past_n = normalize(thpast, theta_mean, theta_std)
    om_past_n = normalize(omega_past, omega_mean, omega_std)
    u_past_n  = normalize(upast, u_mean, u_std)

    # LSTM expects (batch, seq_len, features=3)
    Xh = np.stack([th_past_n, om_past_n, u_past_n], axis=2).astype(np.float64)  # (N, 15, 3)
    Xh_t = torch.tensor(Xh).double().to(device)

    model_lstm.eval()
    with torch.no_grad():
        pred = model_lstm(Xh_t).double().cpu().numpy()
    pred_theta = pred[:, 0]  # normalized
    pred_theta_unnorm = unnormalize(pred_theta, theta_mean, theta_std)
    # Use wrapped RMSE for angle
    return compute_wrapped_rmse(pred_theta_unnorm, th_now_true)

def lstm_hidden_sim_rmse(model_lstm, seq_len):
    """Teacher-forced next-step over hidden simulation set."""
    if hidden_sim_npz is None:
        return None
    u_key  = get_first_key(hidden_sim_npz, ['u','u_sequence','u_valid','u_test'])
    th_key = get_first_key(hidden_sim_npz, ['th','th_true','y','y_true'])
    if u_key is None or th_key is None:
        return None
    u_h = hidden_sim_npz[u_key].astype(np.float32)
    th_h = hidden_sim_npz[th_key].astype(np.float32)
    if len(th_h) <= seq_len:
        return None

    omega_h = np.gradient(th_h, dt).astype(np.float32)
    th_hn = normalize(th_h, theta_mean, theta_std)
    om_hn = normalize(omega_h, omega_mean, omega_std)
    u_hn = normalize(u_h, u_mean, u_std)

    Xh, Yh = create_lstm_sequences(th_hn, om_hn, u_hn, seq_len=seq_len)
    Xh_t = torch.tensor(Xh).double().to(device)

    model_lstm.eval()
    with torch.no_grad():
        pred = model_lstm(Xh_t).double().cpu().numpy()
    pred_theta = unnormalize(pred[:, 0], theta_mean, theta_std)
    gt_theta = th_h[seq_len:seq_len + len(pred_theta)]
    return compute_wrapped_rmse(pred_theta, gt_theta)

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
                rmse_lstm_test = compute_rmse(pred, Y_test)

            rmse_hidden_pred_lstm = lstm_hidden_prediction_rmse(model_lstm)
            rmse_hidden_sim_lstm = lstm_hidden_sim_rmse(model_lstm, SEQ_LEN)

            print(f"LSTM hs:{hidden_size} nl:{num_layers} bi:{int(bidirectional)} "
                  f"→ TestRMSE: {rmse_lstm_test:.5f} "
                  f"| HiddenPredRMSE(wrap): {('NA' if rmse_hidden_pred_lstm is None else f'{rmse_hidden_pred_lstm:.5f}')} "
                  f"| HiddenSimRMSE(wrap): {('NA' if rmse_hidden_sim_lstm is None else f'{rmse_hidden_sim_lstm:.5f}')}")

            # Prefer hidden sim if available, else hidden pred, else test
            if rmse_hidden_sim_lstm is not None:
                score = rmse_hidden_sim_lstm
            elif rmse_hidden_pred_lstm is not None:
                score = rmse_hidden_pred_lstm
            else:
                score = rmse_lstm_test

            if score < best_rmse_lstm:
                best_rmse_lstm = score
                best_model_lstm = model_lstm
                best_config_lstm = (hidden_size, num_layers, bidirectional)

print(f"\nBest LSTM → hs:{best_config_lstm[0]} nl:{best_config_lstm[1]} bi:{int(best_config_lstm[2])} "
      f"| Score (prefers HiddenSim > HiddenPred > Test): {best_rmse_lstm:.5f}")
torch.save(best_model_lstm.state_dict(), "disc-submission-files/ann-lstm-model.pth")

# ========================
# Simulate LSTM Model (export like before)
# ========================
print("Saving LSTM .npz output...")
input_data = try_load("disc-benchmark-files/hidden-test-simulation-submission-file.npz")
if input_data is not None:
    u_key  = get_first_key(input_data, ['u','u_sequence','u_valid','u_test'])
    th_key = get_first_key(input_data, ['th','th_true','y','y_true'])
    if (u_key is not None) and (th_key is not None):
        u_hidden = input_data[u_key].astype(np.float32)
        th_hidden = input_data[th_key].astype(np.float32)
        omega_hidden = np.gradient(th_hidden, dt).astype(np.float32)

        th_hidden_norm = normalize(th_hidden, theta_mean, theta_std)
        omega_hidden_norm = normalize(omega_hidden, omega_mean, omega_std)
        u_hidden_norm = normalize(u_hidden, u_mean, u_std)

        X_hidden, _ = create_lstm_sequences(th_hidden_norm, omega_hidden_norm, u_hidden_norm, seq_len=SEQ_LEN)
        X_hidden_tensor = torch.tensor(X_hidden).double().to(device)

        model_lstm = best_model_lstm  # use best
        with torch.no_grad():
            preds = model_lstm(X_hidden_tensor).cpu().numpy()
            preds_unnorm = np.empty_like(preds)
            preds_unnorm[:, 0] = unnormalize(preds[:, 0], theta_mean, theta_std)
            preds_unnorm[:, 1] = unnormalize(preds[:, 1], omega_mean, omega_std)

        pred_theta = np.concatenate([th_hidden[:SEQ_LEN], preds_unnorm[:, 0]])
        np.savez("disc-submission-files/ann-lstm-hidden-test-simulation-submission-file.npz",
                 th=pred_theta, u=u_hidden)
        print("Saved LSTM hidden simulation submission npz.")
    else:
        print("⚠ Hidden simulation file lacks expected keys — skipped LSTM export.")
else:
    print("⚠ Hidden simulation file missing — skipped LSTM export.")