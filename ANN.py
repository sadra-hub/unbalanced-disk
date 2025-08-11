# ANN_grid_export_exact_example.py
# - Matches course examples IO:
#   * Training prediction errors (exact prints)
#   * Training simulation errors (exact prints)
#   * Hidden PREDICTION export (upast, thpast -> thnow) respecting (na, nb) or seq_len=15
#   * Hidden SIMULATION export (free-run from skip=50)
#
# Saved files (per config):
#   disc-submission-files/grid/ann-narx_hidden-prediction_*.npz      (upast, thpast, thnow)
#   disc-submission-files/grid/ann-narx_hidden-simulation_*.npz      (th, u)
#   disc-submission-files/grid/ann-lstm_hidden-prediction_*.npz      (upast, thpast, thnow)
#   disc-submission-files/grid/ann-lstm_hidden-simulation_*.npz      (th, u)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# -----------------------
# Setup
# -----------------------
SEED = 88
torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WARMUP = 50           # hidden simulation warm-up
HIDDEN_PRED_L = 15    # hidden prediction history length

os.makedirs("disc-submission-files/grid", exist_ok=True)

# -----------------------
# IO helpers (match examples)
# -----------------------
def create_IO_data(u, y, na, nb):
    X, Y = [], []
    start = max(na, nb)
    for k in range(start, len(y)):
        X.append(np.concatenate([u[k-nb:k], y[k-na:k]]))
        Y.append(y[k])
    return np.array(X, np.float32), np.array(Y, np.float32)

def simulation_IO_model(predict_fn, ulist, ylist, skip=50, na=1, nb=1):
    # Exactly like the example: warm up with 'skip' true outputs, then free-run
    upast = ulist[skip-nb:skip].tolist()
    ypast = ylist[skip-na:skip].tolist()
    Y = ylist[:skip].tolist()
    for u in ulist[skip:]:
        x = np.concatenate([upast, ypast], axis=0)
        ypred = float(predict_fn(x[None, :]))
        Y.append(ypred)
        upast.append(float(u)); upast.pop(0)
        ypast.append(ypred);   ypast.pop(0)
    return np.array(Y, np.float32)

# -----------------------
# Models
# -----------------------
class NARX_MLP(nn.Module):
    def __init__(self, input_dim, hidden_neuron=64, hidden_layers=1, activation="relu"):
        super().__init__()
        act = nn.ReLU if activation == "relu" else nn.Tanh
        layers = []
        in_f = input_dim
        for _ in range(hidden_layers):
            layers += [nn.Linear(in_f, hidden_neuron), act()]
            in_f = hidden_neuron
        layers += [nn.Linear(in_f, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # (N,1)

class LSTM_Seq2One(nn.Module):
    # Input sequence: (batch, seq_len, 2) with features [u, y]
    def __init__(self, input_size=2, hidden_size=64, num_layers=1, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            bidirectional=bidirectional)
        d = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size*d, 1)

    def forward(self, x):
        out, _ = self.lstm(x)           # (B, T, H*d)
        y = self.fc(out[:, -1, :])      # (B, 1)
        return y

def create_lstm_sequences(u, y, seq_len):
    X, Y = [], []
    for i in range(len(y) - seq_len):
        X.append(np.stack([u[i:i+seq_len], y[i:i+seq_len]], axis=1))  # (seq_len, 2)
        Y.append(y[i+seq_len])
    return np.array(X, np.float32), np.array(Y, np.float32)

def simulate_lstm_free_run(model, ulist, ylist, seq_len, skip):
    # Use 'skip' real samples to seed (skip=seq_len for training sim; skip=50 for hidden sim)
    u = np.array(ulist, np.float32); y = np.array(ylist, np.float32)
    Xwin = np.stack([u[skip-seq_len:skip], y[skip-seq_len:skip]], axis=1)  # (seq_len, 2)
    traj = list(y[:skip])
    model.eval()
    with torch.no_grad():
        for t in range(skip, len(u)):
            x_t = torch.tensor(Xwin[None, ...], dtype=torch.float32, device=device)
            y_pred = float(model(x_t).cpu().numpy().reshape(-1)[0])
            traj.append(y_pred)
            Xwin = np.roll(Xwin, shift=-1, axis=0)
            Xwin[-1, 0] = u[t]
            Xwin[-1, 1] = y_pred
    return np.array(traj, np.float32)

# -----------------------
# Load data
# -----------------------
print("Loading data...")
droot = "disc-benchmark-files"
base = np.load(os.path.join(droot, "training-val-test-data.npz"))
th = base["th"].astype(np.float32)
u  = base["u"].astype(np.float32)

# Hidden files (templates)
hidden_pred_path = os.path.join(droot, "hidden-test-prediction-submission-file.npz")
hidden_sim_path  = os.path.join(droot, "hidden-test-simulation-submission-file.npz")
hidden_pred = np.load(hidden_pred_path) if os.path.exists(hidden_pred_path) else None
hidden_sim  = np.load(hidden_sim_path)  if os.path.exists(hidden_sim_path)  else None
if hidden_pred is None: print("⚠ hidden-test-prediction-submission-file.npz not found; will skip hidden prediction export.")
if hidden_sim  is None: print("⚠ hidden-test-simulation-submission-file.npz not found; will skip hidden simulation export.")

# -----------------------
# Training utils
# -----------------------
def train_mlp(model, X, Y, epochs=200, bs=256, lr=1e-3):
    model = model.to(device)
    ds = TensorDataset(torch.tensor(X), torch.tensor(Y).unsqueeze(1))
    loader = DataLoader(ds, batch_size=bs, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    return model

def train_lstm(model, X, Y, epochs=80, bs=128, lr=1e-3):
    model = model.to(device)
    ds = TensorDataset(torch.tensor(X), torch.tensor(Y).unsqueeze(1))
    loader = DataLoader(ds, batch_size=bs, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    return model

def print_train_pred_metrics(y_pred, y_true):
    err = y_pred.reshape(-1) - y_true.reshape(-1)
    RMS = float(np.sqrt(np.mean(err**2)))
    print('train prediction errors:')
    print('RMS:', RMS, 'radians')
    print('RMS:', RMS/(2*np.pi)*360, 'degrees')
    print('NRMS:', RMS/np.std(y_true)*100, '%')

def print_train_sim_metrics(th_sim, th_true, skip):
    err = th_sim[skip:] - th_true[skip:]
    RMS = float(np.sqrt(np.mean(err**2)))
    print('train simulation errors:')
    print('RMS:', RMS, 'radians')
    print('RMS:', RMS/(2*np.pi)*360, 'degrees')
    print('NRMS:', RMS/np.std(th_true)*100, '%')

# ============================================================
# NARX GRID
# ============================================================
narx_na_grid = [5, 8]
narx_nb_grid = [3, 5]
narx_hl_grid = [1, 2]
narx_hn_grid = [32, 64]
narx_act_grid = ["relu", "tanh"]

print("\n=== NARX (MLP) ===")
for na in narx_na_grid:
    for nb in narx_nb_grid:
        # Build TRAIN IO dataset from the whole data (as in the example)
        Xtrain, Ytrain = create_IO_data(u, th, na, nb)
        for hl in narx_hl_grid:
            for hn in narx_hn_grid:
                for act in narx_act_grid:
                    tag = f"na{na}_nb{nb}_hl{hl}_hn{hn}_act{act}"
                    print(f"\n[NARX] Config: {tag}")

                    model = NARX_MLP(input_dim=Xtrain.shape[1], hidden_neuron=hn, hidden_layers=hl, activation=act)
                    model = train_mlp(model, Xtrain, Ytrain, epochs=200, bs=256, lr=1e-3)

                    # (1) TRAIN prediction metrics
                    with torch.no_grad():
                        y_pred_train = model(torch.tensor(Xtrain, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)
                    print_train_pred_metrics(y_pred_train, Ytrain)

                    # (2) TRAIN simulation metrics (skip = max(na, nb))
                    skip_train = max(na, nb)
                    def _mlp_fn(x_np):
                        x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
                        return model(x_t).cpu().numpy().reshape(-1)[0]
                    th_train_sim = simulation_IO_model(_mlp_fn, list(u), list(th), skip=skip_train, na=na, nb=nb)
                    print_train_sim_metrics(th_train_sim, th, skip_train)

                    # (3) Hidden PREDICTION export
                    if hidden_pred is not None and "upast" in hidden_pred and "thpast" in hidden_pred:
                        up = hidden_pred["upast"].astype(np.float32)   # (N, 15)
                        tp = hidden_pred["thpast"].astype(np.float32)  # (N, 15)
                        Xh = np.concatenate([up[:, HIDDEN_PRED_L-nb:], tp[:, HIDDEN_PRED_L-na:]], axis=1).astype(np.float32)
                        with torch.no_grad():
                            y_hat = model(torch.tensor(Xh, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1,1)
                        out_pred = f"disc-submission-files/grid/ann-narx_hidden-prediction_{tag}.npz"
                        np.savez(out_pred, upast=up, thpast=tp, thnow=y_hat)
                        print("  ↳ wrote", out_pred)

                    # (4) Hidden SIMULATION export (skip=50)
                    if hidden_sim is not None and "u" in hidden_sim and "th" in hidden_sim:
                        u_test = hidden_sim["u"].astype(np.float32)
                        th_test_init = hidden_sim["th"].astype(np.float32)
                        th_sim = simulation_IO_model(_mlp_fn, list(u_test), list(th_test_init), skip=WARMUP, na=na, nb=nb)
                        out_sim = f"disc-submission-files/grid/ann-narx_hidden-simulation_{tag}.npz"
                        np.savez(out_sim, th=th_sim, u=u_test)
                        print("  ↳ wrote", out_sim)

# ============================================================
# LSTM GRID  (seq_len fixed to 15 to match hidden prediction histories)
# ============================================================
lstm_seq_len = 15  # hidden prediction has 15 past samples
lstm_hs_grid  = [32, 64]
lstm_nl_grid  = [1, 2]
lstm_bi_grid  = [False, True]

print("\n=== LSTM (seq_len=15, features=[u,y]) ===")
Xtrain_seq, Ytrain_seq = create_lstm_sequences(u, th, lstm_seq_len)

for hs in lstm_hs_grid:
    for nl in lstm_nl_grid:
        for bi in lstm_bi_grid:
            tag = f"seq15_hs{hs}_nl{nl}_bi{int(bi)}"
            print(f"\n[LSTM] Config: {tag}")
            model = LSTM_Seq2One(input_size=2, hidden_size=hs, num_layers=nl, bidirectional=bi)
            model = train_lstm(model, Xtrain_seq, Ytrain_seq, epochs=80, bs=128, lr=1e-3)

            # (1) TRAIN prediction metrics
            with torch.no_grad():
                y_pred_train = model(torch.tensor(Xtrain_seq, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)
            print_train_pred_metrics(y_pred_train, Ytrain_seq)

            # (2) TRAIN simulation metrics (skip = seq_len)
            th_train_sim = simulate_lstm_free_run(model, list(u), list(th), seq_len=lstm_seq_len, skip=lstm_seq_len)
            print_train_sim_metrics(th_train_sim, th, lstm_seq_len)

            # (3) Hidden PREDICTION export
            if hidden_pred is not None and "upast" in hidden_pred and "thpast" in hidden_pred:
                up = hidden_pred["upast"].astype(np.float32)   # (N, 15)
                tp = hidden_pred["thpast"].astype(np.float32)  # (N, 15)
                Xh = np.stack([up, tp], axis=2)                # (N, 15, 2)
                with torch.no_grad():
                    y_hat = model(torch.tensor(Xh, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1,1)
                out_pred = f"disc-submission-files/grid/ann-lstm_hidden-prediction_{tag}.npz"
                np.savez(out_pred, upast=up, thpast=tp, thnow=y_hat)
                print("  ↳ wrote", out_pred)

            # (4) Hidden SIMULATION export (skip=50)
            if hidden_sim is not None and "u" in hidden_sim and "th" in hidden_sim:
                u_test = hidden_sim["u"].astype(np.float32)
                th_test_init = hidden_sim["th"].astype(np.float32)
                th_sim = simulate_lstm_free_run(model, list(u_test), list(th_test_init), seq_len=lstm_seq_len, skip=WARMUP)
                out_sim = f"disc-submission-files/grid/ann-lstm_hidden-simulation_{tag}.npz"
                np.savez(out_sim, th=th_sim, u=u_test)
                print("  ↳ wrote", out_sim)

print("\nDone.")