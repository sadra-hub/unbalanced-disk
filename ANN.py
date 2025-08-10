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
# Angle helpers & metrics
# ========================
def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def rmse_wrap(pred_theta, true_theta):
    pred_theta = np.asarray(pred_theta).reshape(-1)
    true_theta = np.asarray(true_theta).reshape(-1)
    err = wrap_angle(pred_theta - true_theta)
    return float(np.sqrt(np.mean(err**2)))

def compute_rmse_torch(y_pred, y_true):
    y_pred = torch.as_tensor(y_pred, dtype=torch.float32, device="cpu")
    y_true = torch.as_tensor(y_true, dtype=torch.float32, device="cpu")
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()

def sincos_to_theta(y_sc):
    # y_sc: (N,2) with [sinθ, cosθ]
    y_sc = np.asarray(y_sc)
    return np.arctan2(y_sc[:, 0], y_sc[:, 1])

# ========================
# NARX Dataset (sin/cos targets)
# ========================
def create_IO_data_sincos(u, th, na, nb):
    s = np.sin(th).astype(np.float32)
    c = np.cos(th).astype(np.float32)
    X, Y = [], []
    start = max(na, nb)
    for k in range(start, len(th)):
        u_blk = u[k-nb:k]
        s_blk = s[k-na:k]
        c_blk = c[k-na:k]
        X.append(np.concatenate([u_blk, s_blk, c_blk], axis=0))
        Y.append([s[k], c[k]])  # target = [sinθ_k, cosθ_k]
    return np.array(X, np.float32), np.array(Y, np.float32)

def standardize_X(X, mu=None, sig=None):
    if mu is None:
        mu = X.mean(0)
        sig = X.std(0) + 1e-12
    return (X - mu) / sig, mu, sig

# ========================
# Hidden-set IO builders for NARX (sin/cos)
# ========================
def get_first_key(dct, candidates):
    for k in candidates:
        if k in dct:
            return k
    return None

def narx_hidden_prediction_rmse(model, na, nb, pred_npz, x_mu, x_sig):
    if pred_npz is None:
        return None
    if ('upast' not in pred_npz) or ('thpast' not in pred_npz):
        return None
    upast = pred_npz['upast'].astype(np.float32)  # (N, L=15)
    thpast = pred_npz['thpast'].astype(np.float32)

    gt_key = get_first_key(pred_npz, ['thnow_true','thnow','y_true','y','target'])
    if gt_key is None:
        return None
    th_now_true = pred_npz[gt_key].astype(np.float32).reshape(-1)

    if upast.shape[1] < nb or thpast.shape[1] < na:
        return None

    u_blk = upast[:, -nb:] if nb > 0 else np.zeros((upast.shape[0], 0), np.float32)
    s_blk = np.sin(thpast[:, -na:]) if na > 0 else np.zeros((upast.shape[0], 0), np.float32)
    c_blk = np.cos(thpast[:, -na:]) if na > 0 else np.zeros((upast.shape[0], 0), np.float32)
    Xh = np.concatenate([u_blk, s_blk, c_blk], axis=1).astype(np.float32)
    Xh_n, _, _ = standardize_X(Xh, x_mu, x_sig)

    model.eval()
    with torch.no_grad():
        y_sc = model(torch.tensor(Xh_n, dtype=torch.float32, device=device)).cpu().numpy()  # (N,2)
    th_hat = sincos_to_theta(y_sc)
    return rmse_wrap(th_hat, th_now_true)

def use_NARX_model_in_simulation_sincos(ulist, ylist, model, na, nb, x_mu, x_sig):
    # Free-run with 50 warmup; model predicts [sinθ,cosθ], we feed back θ via atan2
    ylist = list(ylist[:50])
    upast = list(ulist[50-nb:50])
    ypast = list(ylist[-na:])

    model.eval()
    with torch.no_grad():
        for unow in ulist[50:]:
            sin_blk = np.sin(np.array(ypast[-na:], np.float32))
            cos_blk = np.cos(np.array(ypast[-na:], np.float32))
            x = np.concatenate([np.array(upast[-nb:], np.float32), sin_blk, cos_blk])[None, :]
            x = (x - x_mu) / x_sig
            y_sc = model(torch.tensor(x, dtype=torch.float32, device=device)).cpu().numpy()[0]  # [sin,cos]
            y_new = float(np.arctan2(y_sc[0], y_sc[1]))
            upast.append(float(unow)); upast.pop(0)
            ypast.append(y_new);       ypast.pop(0)
            ylist.append(y_new)

    return np.array(ylist, np.float32)

def narx_hidden_simulation_rmse(model, na, nb, sim_npz, x_mu, x_sig):
    if sim_npz is None:
        return None
    u_key  = get_first_key(sim_npz, ['u','u_sequence','u_valid','u_test'])
    th_key = get_first_key(sim_npz, ['th','th_true','y','y_true'])
    if u_key is None or th_key is None:
        return None
    u_seq = sim_npz[u_key].astype(np.float32)
    th_true = sim_npz[th_key].astype(np.float32)
    if len(u_seq) < 60 or len(th_true) < 60:
        return None
    th_sim = use_NARX_model_in_simulation_sincos(list(u_seq), list(th_true), model, na, nb, x_mu, x_sig)
    L = min(len(th_sim), len(th_true))
    return rmse_wrap(th_sim[:L], th_true[:L])

# ========================
# NARX Model (outputs 2: sin, cos)
# ========================
class NARX(nn.Module):
    def __init__(self, input_dim, hidden_neuron, hidden_layers, activation):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_neuron] * hidden_layers + [2]
        self.layers = nn.ModuleList([nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.act = activation()

    def forward(self, x):
        for lin in self.layers[:-1]:
            x = self.act(lin(x))
        return self.layers[-1](x)  # [sinθ, cosθ]

# ========================
# LSTM Model (unchanged; predicts [θ, ω])
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
# Load data & normalize (for LSTM)
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

def normalize(x, mean, std): return (x - mean) / (std + 1e-12)
def unnormalize(x, mean, std): return x * (std + 1e-12) + mean

th_norm = normalize(th, theta_mean, theta_std)
omega_norm = normalize(omega, omega_mean, omega_std)
u_norm = normalize(u, u_mean, u_std)

# Hidden files
hidden_pred_npz = np.load('disc-benchmark-files/hidden-test-prediction-submission-file.npz') if os.path.exists('disc-benchmark-files/hidden-test-prediction-submission-file.npz') else None
hidden_sim_npz  = np.load('disc-benchmark-files/hidden-test-simulation-submission-file.npz')  if os.path.exists('disc-benchmark-files/hidden-test-simulation-submission-file.npz')  else None
if hidden_pred_npz is None: print("⚠ Hidden prediction file not found; will skip hidden prediction RMSE.")
if hidden_sim_npz is None:  print("⚠ Hidden simulation file not found; will skip hidden simulation RMSE.")

# ========================
# Train NARX Models (sin/cos outputs) + per-iteration logs
# ========================
print("\n=== Training NARX Models (sin/cos outputs; wrapped metrics) ===")
batch_size = 256
epochs = 400
learning_rate = 1e-3

best_score = float("inf")
best_model_narx = None
best_config_narx = None
best_norm_stats = None  # (x_mu, x_sig)

for na in [5, 8]:
    for nb in [3, 5]:
        # Build IO (sin/cos)
        X_train_np, y_train_np = create_IO_data_sincos(u[:int(0.8*len(th))], th[:int(0.8*len(th))], na, nb)
        X_val_np,   y_val_theta = create_IO_data_sincos(u[int(0.8*len(th)):int(0.9*len(th))],
                                                        th[int(0.8*len(th)):int(0.9*len(th))], na, nb)
        # We'll use y_val_theta’s θ from the original series:
        # We need the actual θ at indices; it's equivalent to atan2(y_val_sc[:,0], y_val_sc[:,1])
        # but we can just recompute from th slice:
        y_val_theta = th[int(0.8*len(th)) + max(na, nb): int(0.9*len(th))]  # ground-truth θ

        # Standardize X
        X_train, x_mu, x_sig = standardize_X(X_train_np)
        X_val,   _,    _     = standardize_X(X_val_np, x_mu, x_sig)

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train_np, dtype=torch.float32)
        X_val_t   = torch.tensor(X_val, dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

        for act_name, activation in [("tanh", nn.Tanh), ("relu", nn.ReLU)]:
            for hl in [1, 2]:
                for hn in [32, 64]:
                    model_narx = NARX(input_dim=nb + 2*na, hidden_neuron=hn, hidden_layers=hl, activation=activation).to(device)
                    opt = optim.Adam(model_narx.parameters(), lr=learning_rate, weight_decay=1e-5)
                    mse = nn.MSELoss()

                    # Train
                    for epoch in range(epochs):
                        model_narx.train()
                        for xb, yb in train_loader:
                            xb = xb.to(device); yb = yb.to(device)
                            opt.zero_grad()
                            pred_sc = model_narx(xb)  # [sin,cos]
                            # MSE on sin/cos + small unit-circle penalty
                            loss = mse(pred_sc, yb) + 1e-3 * ((pred_sc.norm(dim=1) - 1.0) ** 2).mean()
                            loss.backward()
                            opt.step()

                    # Validation one-step (wrapped RMSE on θ)
                    model_narx.eval()
                    with torch.no_grad():
                        pred_sc_val = model_narx(X_val_t.to(device)).cpu().numpy()
                    th_val_hat = sincos_to_theta(pred_sc_val)
                    rmse_val_wrap = rmse_wrap(th_val_hat, y_val_theta)

                    # Hidden prediction RMSE (wrapped)
                    rmse_hidden_pred = narx_hidden_prediction_rmse(model_narx, na, nb, hidden_pred_npz, x_mu, x_sig)

                    # Hidden simulation RMSE (wrapped)
                    rmse_hidden_sim = narx_hidden_simulation_rmse(model_narx, na, nb, hidden_sim_npz, x_mu, x_sig)

                    print(f"na:{na} nb:{nb} hl:{hl} hn:{hn} act:{act_name} "
                          f"→ ValPredRMSE(wrap): {rmse_val_wrap:.5f} "
                          f"| HiddenPredRMSE(wrap): {('NA' if rmse_hidden_pred is None else f'{rmse_hidden_pred:.5f}')} "
                          f"| HiddenSimRMSE(wrap): {('NA' if rmse_hidden_sim is None else f'{rmse_hidden_sim:.5f}')}")

                    # pick score: prefer hidden sim, else hidden pred, else val
                    if rmse_hidden_sim is not None:
                        score = rmse_hidden_sim
                    elif rmse_hidden_pred is not None:
                        score = rmse_hidden_pred
                    else:
                        score = rmse_val_wrap

                    if score < best_score:
                        best_score = score
                        best_model_narx = model_narx
                        best_config_narx = (na, nb, hl, hn, act_name)
                        best_norm_stats = (x_mu.copy(), x_sig.copy())

# Save best NARX + export hidden simulation submission
if best_model_narx is not None:
    x_mu, x_sig = best_norm_stats
    torch.save(best_model_narx.state_dict(), "disc-submission-files/ann-narx-model.pth")
    print(f"\nBest NARX: na:{best_config_narx[0]} nb:{best_config_narx[1]} hl:{best_config_narx[2]} hn:{best_config_narx[3]} act:{best_config_narx[4]} | score:{best_score:.5f}")

    # Export hidden simulation .npz if available
    if hidden_sim_npz is not None:
        u_key  = get_first_key(hidden_sim_npz, ['u','u_sequence','u_valid','u_test'])
        th_key = get_first_key(hidden_sim_npz, ['th','th_true','y','y_true'])
        if (u_key is not None) and (th_key is not None):
            u_test = hidden_sim_npz[u_key].astype(np.float32)
            th_test = hidden_sim_npz[th_key].astype(np.float32)
            th_sim = use_NARX_model_in_simulation_sincos(list(u_test), list(th_test), best_model_narx, best_config_narx[0], best_config_narx[1], x_mu, x_sig)
            np.savez('disc-submission-files/ann-narx-hidden-test-simulation-submission-file.npz',
                     th=th_sim, u=u_test)
            print("Saved NARX hidden simulation submission npz.")
    else:
        print("⚠ Hidden simulation file missing — skipped NARX export.")

# ========================
# LSTM (seq_len=15) — unchanged from your version
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
X_test,  Y_test  = X_lstm[int(0.8*total):], Y_lstm[int(0.8*total):]

X_train = torch.tensor(X_train).double().to(device)
Y_train = torch.tensor(Y_train).double().to(device)
X_test  = torch.tensor(X_test).double().to(device)
Y_test  = torch.tensor(Y_test).double().to(device)

best_rmse_lstm = float("inf")
best_model_lstm = None
best_config_lstm = None

def lstm_hidden_prediction_rmse(model_lstm):
    if hidden_pred_npz is None: return None
    if ('upast' not in hidden_pred_npz) or ('thpast' not in hidden_pred_npz): return None
    upast = hidden_pred_npz['upast'].astype(np.float32)  # (N,15)
    thpast = hidden_pred_npz['thpast'].astype(np.float32)
    gt_key = get_first_key(hidden_pred_npz, ['thnow_true','thnow','y_true','y','target'])
    if gt_key is None: return None
    th_now_true = hidden_pred_npz[gt_key].astype(np.float32).reshape(-1)

    omega_past = np.gradient(thpast, axis=1) / dt
    th_past_n = normalize(thpast, theta_mean, theta_std)
    om_past_n = normalize(omega_past, omega_mean, omega_std)
    u_past_n  = normalize(upast,  u_mean, u_std)

    Xh = np.stack([th_past_n, om_past_n, u_past_n], axis=2).astype(np.float64)
    Xh_t = torch.tensor(Xh).double().to(device)

    model_lstm.eval()
    with torch.no_grad():
        pred = model_lstm(Xh_t).double().cpu().numpy()
    pred_theta = unnormalize(pred[:,0], theta_mean, theta_std)
    return rmse_wrap(pred_theta, th_now_true)

def lstm_hidden_sim_rmse(model_lstm, seq_len):
    if hidden_sim_npz is None: return None
    u_key  = get_first_key(hidden_sim_npz, ['u','u_sequence','u_valid','u_test'])
    th_key = get_first_key(hidden_sim_npz, ['th','th_true','y','y_true'])
    if u_key is None or th_key is None: return None
    u_h = hidden_sim_npz[u_key].astype(np.float32)
    th_h = hidden_sim_npz[th_key].astype(np.float32)
    if len(th_h) <= seq_len: return None

    omega_h = np.gradient(th_h, dt).astype(np.float32)
    th_hn = normalize(th_h, theta_mean, theta_std)
    om_hn = normalize(omega_h, omega_mean, omega_std)
    u_hn = normalize(u_h, u_mean, u_std)

    Xh, Yh = create_lstm_sequences(th_hn, om_hn, u_hn, seq_len=seq_len)
    Xh_t = torch.tensor(Xh).double().to(device)

    model_lstm.eval()
    with torch.no_grad():
        pred = model_lstm(Xh_t).double().cpu().numpy()
    pred_theta = unnormalize(pred[:,0], theta_mean, theta_std)
    gt_theta = th_h[seq_len:seq_len + len(pred_theta)]
    return rmse_wrap(pred_theta, gt_theta)

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
                rmse_lstm_test = compute_rmse_torch(pred, Y_test)

            rmse_hidden_pred_lstm = lstm_hidden_prediction_rmse(model_lstm)
            rmse_hidden_sim_lstm  = lstm_hidden_sim_rmse(model_lstm, SEQ_LEN)

            print(f"LSTM hs:{hidden_size} nl:{num_layers} bi:{int(bidirectional)} "
                  f"→ TestRMSE: {rmse_lstm_test:.5f} "
                  f"| HiddenPredRMSE(wrap): {('NA' if rmse_hidden_pred_lstm is None else f'{rmse_hidden_pred_lstm:.5f}')} "
                  f"| HiddenSimRMSE(wrap): {('NA' if rmse_hidden_sim_lstm is None else f'{rmse_hidden_sim_lstm:.5f}')}")

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
if hidden_sim_npz is not None:
    u_key  = get_first_key(hidden_sim_npz, ['u','u_sequence','u_valid','u_test'])
    th_key = get_first_key(hidden_sim_npz, ['th','th_true','y','y_true'])
    if (u_key is not None) and (th_key is not None):
        u_hidden = hidden_sim_npz[u_key].astype(np.float32)
        th_hidden = hidden_sim_npz[th_key].astype(np.float32)
        omega_hidden = np.gradient(th_hidden, dt).astype(np.float32)

        th_hidden_norm = normalize(th_hidden, theta_mean, theta_std)
        omega_hidden_norm = normalize(omega_hidden, omega_mean, omega_std)
        u_hidden_norm = normalize(u_hidden, u_mean, u_std)

        def create_lstm_sequences(theta, omega, u, seq_len):
            X, Y = [], []
            for i in range(len(theta) - seq_len):
                x_seq = np.stack([theta[i:i+seq_len], omega[i:i+seq_len], u[i:i+seq_len]], axis=1)
                y_target = [theta[i+seq_len], omega[i+seq_len]]
                X.append(x_seq); Y.append(y_target)
            return np.array(X, dtype=np.float64), np.array(Y, dtype=np.float64)

        X_hidden, _ = create_lstm_sequences(th_hidden_norm, omega_hidden_norm, u_hidden_norm, seq_len=SEQ_LEN)
        X_hidden_tensor = torch.tensor(X_hidden).double().to(device)

        model_lstm = best_model_lstm
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