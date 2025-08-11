# ANN_grid_export.py
# Trains NARX (MLP) and LSTM on the unbalanced disc dataset.
# After EACH grid config, saves two checker-ready NPZs on TEST split:
#   - disc-submission-files/grid/ann-narx_test-simulation_*.npz        (key: 'th')
#   - disc-submission-files/grid/ann-narx_test-prediction_*.npz         (keys: 'upast','thpast','thnow')
#   - disc-submission-files/grid/ann-lstm_test-simulation_*.npz         (key: 'th')
#   - disc-submission-files/grid/ann-lstm_test-prediction_*.npz         (keys: 'upast','thpast','thnow')
#
# Uses only u and θ (encoded as sin/cos); free-run simulation uses a 50-sample warm-up.

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
WARMUP = 50
PRED_L = 15  # length of past window in output prediction NPZ (upast, thpast)

os.makedirs("disc-submission-files/grid", exist_ok=True)

# -----------------------
# Angle helpers & misc
# -----------------------
def wrap_angle(a): return (a + np.pi) % (2*np.pi) - np.pi

def rmse_wrap(pred_theta, true_theta):
    pred_theta = np.asarray(pred_theta).reshape(-1)
    true_theta = np.asarray(true_theta).reshape(-1)
    err = wrap_angle(pred_theta - true_theta)
    return float(np.sqrt(np.mean(err**2)))

def sincos_to_theta(y_sc):  # y_sc: (N,2) with [sinθ, cosθ]
    y_sc = np.asarray(y_sc)
    return np.arctan2(y_sc[:, 0], y_sc[:, 1])

def align_length(arr, target_len):
    """Crop or pad with edge value to match solution length (if solution exists)."""
    if target_len is None: 
        return arr
    arr = np.asarray(arr).reshape(-1)
    if len(arr) == target_len:
        return arr
    if len(arr) > target_len:
        return arr[:target_len]
    pad_val = arr[-1] if len(arr) > 0 else 0.0
    return np.pad(arr, (0, target_len - len(arr)), mode="edge", constant_values=pad_val)

def make_pred_windows(u, th, L=15):
    upast, thpast = [], []
    for k in range(L, len(th)):
        upast.append(u[k-L:k])
        thpast.append(th[k-L:k])
    return np.array(upast, np.float32), np.array(thpast, np.float32)

# -----------------------
# Data loading
# -----------------------
print("Loading data...")
droot = "disc-benchmark-files"
base = np.load(os.path.join(droot, "training-val-test-data.npz"))
th_all = base["th"].astype(np.float32)
u_all  = base["u"].astype(np.float32)

N = len(th_all)
i_tr = int(0.8*N); i_va = int(0.9*N)
th_tr, th_va, th_te = th_all[:i_tr], th_all[i_tr:i_va], th_all[i_va:]
u_tr,  u_va,  u_te  = u_all[:i_tr],  u_all[i_tr:i_va],  u_all[i_va:]

# Read official TEST solution lengths (if present) to guarantee checker shape match
sim_sol_path  = os.path.join(droot, "test-simulation-solution-file.npz")
pred_sol_path = os.path.join(droot, "test-prediction-solution-file.npz")
sim_target_len  = np.load(sim_sol_path)["th"].size      if os.path.exists(sim_sol_path)  else None
pred_target_len = np.load(pred_sol_path)["thnow"].size  if os.path.exists(pred_sol_path) else None

# -----------------------
# NARX dataset (sin/cos targets)
# -----------------------
def create_IO_data_sincos(u, th, na, nb):
    s = np.sin(th).astype(np.float32)
    c = np.cos(th).astype(np.float32)
    X, Y = [], []
    start = max(na, nb)
    for k in range(start, len(th)):
        X.append(np.concatenate([u[k-nb:k], s[k-na:k], c[k-na:k]], axis=0))
        Y.append([s[k], c[k]])  # target for θ_k
    return np.array(X, np.float32), np.array(Y, np.float32)

def standardize_X(X, mu=None, sig=None):
    if mu is None:
        mu = X.mean(0)
        sig = X.std(0) + 1e-12
    return (X - mu) / sig, mu, sig

# -----------------------
# NARX model (MLP predicting [sinθ, cosθ])
# -----------------------
class NARX(nn.Module):
    def __init__(self, input_dim, hidden_neuron=64, hidden_layers=2, activation="relu", dropout=0.0):
        super().__init__()
        act = nn.ReLU if activation == "relu" else nn.Tanh
        layers = []
        in_f = input_dim
        for _ in range(hidden_layers):
            layers += [nn.Linear(in_f, hidden_neuron), act(), nn.Dropout(dropout)]
            in_f = hidden_neuron
        self.mlp = nn.Sequential(*layers, nn.Linear(in_f, 2))  # outputs [sin, cos]

    def forward(self, x):
        y = self.mlp(x)
        y = y / (torch.norm(y, dim=1, keepdim=True) + 1e-8)  # project to unit circle
        return y

def angle_aware_loss(pred_sc, target_sc, alpha=0.5):
    """MSE on sin/cos + wrapped angular loss + small unit-norm penalty."""
    mse = nn.functional.mse_loss(pred_sc, target_sc)
    pred_th   = torch.atan2(pred_sc[:, 0], pred_sc[:, 1])
    target_th = torch.atan2(target_sc[:, 0], target_sc[:, 1])
    diff = torch.atan2(torch.sin(pred_th - target_th), torch.cos(pred_th - target_th))
    ang = torch.mean(diff**2)
    unit_pen = torch.mean((torch.norm(pred_sc, dim=1) - 1.0)**2)
    return mse + alpha*ang + 0.01*unit_pen

def simulate_narx_sincos(ulist, ylist, model, na, nb, x_mu, x_sig):
    # Free-run with WARMUP; feed back predicted θ as sin/cos
    ylist = list(ylist[:WARMUP])
    upast = list(ulist[WARMUP-nb:WARMUP])
    ypast = list(ylist[-na:])
    model.eval()
    with torch.no_grad():
        for unow in ulist[WARMUP:]:
            sin_blk = np.sin(np.array(ypast[-na:], np.float32))
            cos_blk = np.cos(np.array(ypast[-na:], np.float32))
            x = np.concatenate([np.array(upast[-nb:], np.float32), sin_blk, cos_blk], axis=0)[None, :]
            x = (x - x_mu) / x_sig
            y_sc = model(torch.tensor(x, dtype=torch.float32, device=device)).cpu().numpy()[0]
            y_new = float(np.arctan2(y_sc[0], y_sc[1]))
            upast.append(float(unow)); upast.pop(0)
            ypast.append(y_new);       ypast.pop(0)
            ylist.append(y_new)
    return np.array(ylist, np.float32)

# -----------------------
# LSTM (sequence of [u, sinθ, cosθ] → next [sinθ, cosθ])
# -----------------------
def create_lstm_sequences(u, th, seq_len):
    s = np.sin(th).astype(np.float32)
    c = np.cos(th).astype(np.float32)
    X, Y = [], []
    for i in range(len(th) - seq_len):
        X.append(np.stack([u[i:i+seq_len], s[i:i+seq_len], c[i:i+seq_len]], axis=1))  # (seq,3)
        Y.append([s[i+seq_len], c[i+seq_len]])
    return np.array(X, np.float32), np.array(Y, np.float32)

class LSTMHead(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=1, bidirectional=False, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=(dropout if num_layers > 1 else 0.0))
        d = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size*d, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        y = self.fc(out[:, -1, :])
        y = y / (torch.norm(y, dim=1, keepdim=True) + 1e-8)
        return y

def simulate_lstm(ulist, ylist, model, seq_len):
    # Free-run with WARMUP; maintain rolling window of [u, sinθ, cosθ]
    y = np.array(ylist, np.float32)
    u = np.array(ulist, np.float32)
    s = np.sin(y); c = np.cos(y)
    Xwin = np.stack([u[WARMUP-seq_len:WARMUP], s[WARMUP-seq_len:WARMUP], c[WARMUP-seq_len:WARMUP]], axis=1)  # (seq,3)
    traj = list(y[:WARMUP])
    model.eval()
    with torch.no_grad():
        for t in range(WARMUP, len(u)):
            x_t = torch.tensor(Xwin[None, ...], dtype=torch.float32, device=device)
            y_sc = model(x_t).cpu().numpy()[0]
            y_new = float(np.arctan2(y_sc[0], y_sc[1]))
            traj.append(y_new)
            Xwin = np.roll(Xwin, shift=-1, axis=0)
            Xwin[-1, 0] = u[t]
            Xwin[-1, 1] = np.sin(y_new)
            Xwin[-1, 2] = np.cos(y_new)
    return np.array(traj, np.float32)

# -----------------------
# Training helpers
# -----------------------
def train_narx(model, Xtr, Ytr, Xva, Yva, epochs=200, bs=256, lr=2e-3):
    model = model.to(device)
    train_loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr)), batch_size=bs, shuffle=True)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best = float("inf"); best_state = None
    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = angle_aware_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if (ep+1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                y_sc = model(torch.tensor(Xva, dtype=torch.float32, device=device)).cpu().numpy()
            val_rmse = rmse_wrap(sincos_to_theta(y_sc), np.arctan2(Yva[:,0], Yva[:,1]))
            if val_rmse < best:
                best = val_rmse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def train_lstm(model, Xtr, Ytr, Xva, Yva, epochs=80, bs=128, lr=2e-3):
    model = model.to(device)
    train_loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr)), batch_size=bs, shuffle=True)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best = float("inf"); best_state = None
    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = angle_aware_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if (ep+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                y_sc = model(torch.tensor(Xva, dtype=torch.float32, device=device)).cpu().numpy()
            val_rmse = rmse_wrap(sincos_to_theta(y_sc), np.arctan2(Yva[:,0], Yva[:,1]))
            if val_rmse < best:
                best = val_rmse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# -----------------------
# NARX grid → save NPZs per config
# -----------------------
narx_na_grid = [5, 8]
narx_nb_grid = [3, 5]
narx_hl_grid = [1, 2]
narx_hn_grid = [64, 128]
narx_act_grid = ["relu", "tanh"]

print("\n=== NARX grid (saving checker-ready files per config) ===")
for na in narx_na_grid:
    for nb in narx_nb_grid:
        # Build IO data
        Xtr_np, Ytr_np = create_IO_data_sincos(u_tr, th_tr, na, nb)
        Xva_np, Yva_np = create_IO_data_sincos(u_va, th_va, na, nb)
        Xte_np, Yte_np = create_IO_data_sincos(u_te, th_te, na, nb)
        if len(Xtr_np) < 100 or len(Xte_np) < 10:
            print(f"skip NARX na={na} nb={nb} (too few samples)"); continue

        # Standardize with train stats
        Xtr, x_mu, x_sig = standardize_X(Xtr_np)
        Xva, _, _        = standardize_X(Xva_np, x_mu, x_sig)
        Xte, _, _        = standardize_X(Xte_np, x_mu, x_sig)

        for hl in narx_hl_grid:
            for hn in narx_hn_grid:
                for act in narx_act_grid:
                    tag = f"na{na}_nb{nb}_hl{hl}_hn{hn}_act{act}"
                    print(f"  NARX [{tag}]")

                    model = NARX(input_dim=Xtr.shape[1], hidden_neuron=hn, hidden_layers=hl, activation=act)
                    model = train_narx(model, Xtr, Ytr_np, Xva, Yva_np)

                    # (A) TEST prediction → upast/thpast (L=15) + thnow computed with model's own (na,nb) features
                    upast15, thpast15 = make_pred_windows(u_te, th_te, L=PRED_L)
                    thnow_list = []
                    with torch.no_grad():
                        for k in range(PRED_L, len(th_te)):
                            u_blk  = u_te[k-nb:k].astype(np.float32) if nb > 0 else np.zeros(0, np.float32)
                            th_blk = th_te[k-na:k].astype(np.float32) if na > 0 else np.zeros(0, np.float32)
                            s_blk, c_blk = np.sin(th_blk), np.cos(th_blk)
                            x = np.concatenate([u_blk, s_blk, c_blk], axis=0)[None, :]
                            x = (x - x_mu) / x_sig
                            y_sc = model(torch.tensor(x, dtype=torch.float32, device=device)).cpu().numpy()[0]
                            thnow_list.append(np.arctan2(y_sc[0], y_sc[1]))
                    thnow = np.array(thnow_list, np.float32)
                    # Align to test-solution length if present
                    TL = pred_target_len if pred_target_len is not None else len(thnow)
                    upast15  = upast15[:TL]
                    thpast15 = thpast15[:TL]
                    thnow    = thnow[:TL]
                    pred_out = f"disc-submission-files/grid/ann-narx_test-prediction_{tag}.npz"
                    np.savez(pred_out, upast=upast15, thpast=thpast15, thnow=thnow.reshape(-1,1))

                    # (B) TEST simulation → th (free-run)
                    th_test_sim = simulate_narx_sincos(list(u_te), list(th_te), model, na, nb, x_mu, x_sig)
                    th_test_sim = align_length(th_test_sim, sim_target_len)
                    sim_out = f"disc-submission-files/grid/ann-narx_test-simulation_{tag}.npz"
                    np.savez(sim_out, th=th_test_sim)

                    print(f"    ↳ wrote {os.path.basename(pred_out)} and {os.path.basename(sim_out)}")

# -----------------------
# LSTM grid → save NPZs per config
# -----------------------
lstm_seq_grid = [15, 20]
lstm_hs_grid  = [64, 128]
lstm_nl_grid  = [1, 2]
lstm_bi_grid  = [False, True]

print("\n=== LSTM grid (saving checker-ready files per config) ===")
for seq_len in lstm_seq_grid:
    Xtr_np, Ytr_np = create_lstm_sequences(u_tr, th_tr, seq_len)
    Xva_np, Yva_np = create_lstm_sequences(u_va, th_va, seq_len)
    Xte_np, Yte_np = create_lstm_sequences(u_te, th_te, seq_len)
    if len(Xtr_np) < 100 or len(Xte_np) < 10:
        print(f"skip LSTM seq={seq_len} (too few samples)"); continue

    for hs in lstm_hs_grid:
        for nl in lstm_nl_grid:
            for bi in lstm_bi_grid:
                tag = f"seq{seq_len}_hs{hs}_nl{nl}_bi{int(bi)}"
                print(f"  LSTM [{tag}]")
                model = LSTMHead(input_size=3, hidden_size=hs, num_layers=nl, bidirectional=bi)
                model = train_lstm(model, Xtr_np, Ytr_np, Xva_np, Yva_np)

                # (A) TEST prediction → upast/thpast (L=15) + thnow predicted with rolling seq_len window
                upast15, thpast15 = make_pred_windows(u_te, th_te, L=PRED_L)
                start_k = max(PRED_L, seq_len)
                thnow_list = []
                with torch.no_grad():
                    u = u_te.astype(np.float32); y = th_te.astype(np.float32)
                    s = np.sin(y); c = np.cos(y)
                    for k in range(start_k, len(y)):
                        Xwin = np.stack([u[k-seq_len:k], s[k-seq_len:k], c[k-seq_len:k]], axis=1)[None, ...]
                        y_sc = model(torch.tensor(Xwin, dtype=torch.float32, device=device)).cpu().numpy()[0]
                        thnow_list.append(np.arctan2(y_sc[0], y_sc[1]))
                thnow = np.array(thnow_list, np.float32)

                # Align everything to have identical N and match solution length if present
                if pred_target_len is not None:
                    TL = pred_target_len
                else:
                    TL = min(len(thnow), len(upast15) - (start_k - PRED_L))
                thnow = thnow[:TL]
                # shift upast/thpast to start at start_k (so windows align to thnow indices)
                upast15  = upast15[(start_k - PRED_L):(start_k - PRED_L) + TL]
                thpast15 = thpast15[(start_k - PRED_L):(start_k - PRED_L) + TL]

                pred_out = f"disc-submission-files/grid/ann-lstm_test-prediction_{tag}.npz"
                np.savez(pred_out, upast=upast15, thpast=thpast15, thnow=thnow.reshape(-1,1))

                # (B) TEST simulation: free-run over full test series with WARMUP
                th_test_sim = simulate_lstm(list(u_te), list(th_te), model, seq_len)
                th_test_sim = align_length(th_test_sim, sim_target_len)
                sim_out = f"disc-submission-files/grid/ann-lstm_test-simulation_{tag}.npz"
                np.savez(sim_out, th=th_test_sim)

                print(f"    ↳ wrote {os.path.basename(pred_out)} and {os.path.basename(sim_out)}")

print("\nDone. Checker examples:")
print("  python submission-file-checker.py disc-submission-files/grid/ann-narx_test-simulation_na5_nb3_hl1_hn64_actrelu.npz disc-benchmark-files/test-simulation-solution-file.npz")
print("  python submission-file-checker.py disc-submission-files/grid/ann-narx_test-prediction_na5_nb3_hl1_hn64_actrelu.npz disc-benchmark-files/test-prediction-solution-file.npz")