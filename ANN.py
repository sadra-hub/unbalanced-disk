# ann_optuna_narx_lstm.py
# Sadra's Optuna tuner for the unbalanced disc assignment
# - Tunes NARX (MLP) with Optuna (default ON)
# - (Optional) Tunes LSTM with Optuna (default OFF)
# - Prints *exact* training prediction/simulation errors (RMS rad/deg + NRMS)
# - Exports checker-ready NPZs for HIDDEN prediction/simulation:
#     disc-submission-files/ann-narx_hidden-prediction_best_optuna.npz  (upast, thpast, thnow)
#     disc-submission-files/ann-narx_hidden-simulation_best_optuna.npz  (u, th)
#     (If LSTM enabled, similar files with ann-lstm_* names)
#
# Data files expected in ./disc-benchmark-files/:
#   training-val-test-data.npz
#   hidden-test-prediction-submission-file.npz
#   hidden-test-simulation-submission-file.npz

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna

# -----------------------
# Switches
# -----------------------
RUN_NARX_OPTUNA = True
RUN_LSTM_OPTUNA = True   # flip to True if you want to tune LSTM too

# -----------------------
# Setup
# -----------------------
SEED = 88
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WARMUP_HIDDEN = 50     # hidden simulation warm-up (as in assignment)
HIDDEN_PRED_L = 15     # hidden prediction past window length (upast/thpast)
SAVE_DIR = "disc-submission-files"
os.makedirs(SAVE_DIR, exist_ok=True)

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
    uarr = np.asarray(ulist, dtype=np.float32).reshape(-1)
    yarr = np.asarray(ylist, dtype=np.float32).reshape(-1)

    upast = uarr[skip - nb:skip].tolist()
    ypast = yarr[skip - na:skip].tolist()
    Y     = yarr[:skip].tolist()

    for u_now in uarr[skip:]:
        x = np.asarray(upast + ypast, dtype=np.float32)[None, :]
        ypred = float(predict_fn(x))
        Y.append(ypred)
        upast.append(float(u_now)); upast.pop(0)
        ypast.append(ypred);       ypast.pop(0)
    return np.asarray(Y, dtype=np.float32)

def print_train_pred_metrics(y_pred, y_true, skip):
    err = y_pred.reshape(-1) - y_true.reshape(-1)
    RMS = float(np.sqrt(np.mean(err[max(skip,0):]**2)))
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

# -----------------------
# Models
# -----------------------
class NARX_MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers=1, hidden_units=64, activation="relu"):
        super().__init__()
        act = nn.ReLU if activation == "relu" else nn.Tanh
        layers = []
        in_f = input_dim
        for _ in range(hidden_layers):
            layers += [nn.Linear(in_f, hidden_units), act()]
            in_f = hidden_units
        layers += [nn.Linear(in_f, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class LSTM_Seq2One(nn.Module):
    # Input sequence: (batch, seq_len, 2) with features [u, y]
    def __init__(self, hidden_units=64, num_layers=1, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(2, hidden_units, num_layers, batch_first=True,
                            bidirectional=bidirectional)
        d = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_units*d, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        y = self.fc(out[:, -1, :])
        return y

def create_lstm_sequences(u, y, seq_len=15):
    X, Y = [], []
    for i in range(len(y) - seq_len):
        X.append(np.stack([u[i:i+seq_len], y[i:i+seq_len]], axis=1))  # (seq_len, 2)
        Y.append(y[i+seq_len])
    return np.array(X, np.float32), np.array(Y, np.float32)

def simulate_lstm_free_run(model, ulist, ylist, seq_len, skip):
    u = np.array(ulist, np.float32); y = np.array(ylist, np.float32)
    Xwin = np.stack([u[skip-seq_len:skip], y[skip-seq_len:skip]], axis=1)  # (seq_len, 2)
    traj = list(y[:skip])
    model.eval()
    with torch.no_grad():
        for t in range(skip, len(u)):
            x_t = torch.tensor(Xwin[None, ...], dtype=torch.float32, device=device)
            y_pred = float(model(x_t).detach().cpu().numpy().reshape(-1)[0])
            traj.append(y_pred)
            Xwin = np.roll(Xwin, shift=-1, axis=0)
            Xwin[-1, 0] = u[t]
            Xwin[-1, 1] = y_pred
    return np.array(traj, np.float32)

# -----------------------
# Load data
# -----------------------
droot = "disc-benchmark-files"
base = np.load(os.path.join(droot, "training-val-test-data.npz"))
th_all = base["th"].astype(np.float32)
u_all  = base["u"].astype(np.float32)

# simple split for tuning
N = len(th_all)
i_tr = int(0.7*N); i_va = int(0.85*N)
u_tr,  th_tr  = u_all[:i_tr],     th_all[:i_tr]
u_va,  th_va  = u_all[i_tr:i_va], th_all[i_tr:i_va]
u_te,  th_te  = u_all[i_va:],     th_all[i_va:]

# Hidden files (templates)
hidden_pred_path = os.path.join(droot, "hidden-test-prediction-submission-file.npz")
hidden_sim_path  = os.path.join(droot, "hidden-test-simulation-submission-file.npz")
hidden_pred = np.load(hidden_pred_path) if os.path.exists(hidden_pred_path) else None
hidden_sim  = np.load(hidden_sim_path)  if os.path.exists(hidden_sim_path)  else None
if hidden_pred is None: print("⚠ hidden-test-prediction-submission-file.npz not found; will skip hidden prediction export.")
if hidden_sim  is None: print("⚠ hidden-test-simulation-submission-file.npz not found; will skip hidden simulation export.")

# -----------------------
# NARX + Optuna
# -----------------------
def train_narx(model, X, Y, epochs=120, bs=256, lr=1e-3):
    model = model.to(device)
    ds = TensorDataset(torch.tensor(X), torch.tensor(Y).unsqueeze(1))
    loader = DataLoader(ds, batch_size=bs, shuffle=True)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model

def narx_objective(trial: optuna.trial.Trial):
    # IMPORTANT: keep na, nb <= HIDDEN_PRED_L so hidden prediction slicing works
    na = trial.suggest_int("na", 3, 10)
    nb = trial.suggest_int("nb", 2, 10)
    na = min(na, HIDDEN_PRED_L)
    nb = min(nb, HIDDEN_PRED_L)

    hl = trial.suggest_int("hidden_layers", 1, 3)
    hn = trial.suggest_categorical("hidden_units", [32, 64, 128, 192])
    act = trial.suggest_categorical("activation", ["relu", "tanh"])
    lr  = trial.suggest_float("lr", 5e-5, 3e-3, log=True)
    bs  = trial.suggest_categorical("batch_size", [128, 256, 512])
    ep  = trial.suggest_int("epochs", 80, 180, step=20)

    # Build datasets (NARX IO)
    Xtr, Ytr = create_IO_data(u_tr, th_tr, na, nb)
    Xva, Yva = create_IO_data(u_va, th_va, na, nb)
    if len(Xtr) < 200 or len(Xva) < 100:
        raise optuna.TrialPruned()

    model = NARX_MLP(input_dim=Xtr.shape[1], hidden_layers=hl, hidden_units=hn, activation=act)
    model = train_narx(model, Xtr, Ytr, epochs=ep, bs=bs, lr=lr)

    # Validation metrics: use simulation RMS on validation split as objective
    model.eval()

    def _mlp_fn(x_np):
        x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            y = model(x_t)
        return y.squeeze().detach().cpu().item()

    skip_val = max(na, nb)
    th_val_sim = simulation_IO_model(_mlp_fn, list(u_va), list(th_va), skip=skip_val, na=na, nb=nb)
    err = th_val_sim[skip_val:] - th_va[skip_val:]
    val_sim_rms = float(np.sqrt(np.mean(err**2)))

    # Report intermediate value to enable pruning
    trial.report(val_sim_rms, step=1)
    if trial.should_prune():
        raise optuna.TrialPruned()

    # Also record pred RMS for logging
    with torch.no_grad():
        y_pred_val = model(torch.tensor(Xva, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)
    pred_rms = float(np.sqrt(np.mean((y_pred_val - Yva)**2)))

    trial.set_user_attr("na", na)
    trial.set_user_attr("nb", nb)
    trial.set_user_attr("hidden_layers", hl)
    trial.set_user_attr("hidden_units", hn)
    trial.set_user_attr("activation", act)
    trial.set_user_attr("lr", lr)
    trial.set_user_attr("batch_size", bs)
    trial.set_user_attr("epochs", ep)
    trial.set_user_attr("val_pred_rms", pred_rms)
    return val_sim_rms

def run_narx_optuna(n_trials=25):
    print("\n=== Optuna: NARX (MLP) tuning ===")
    study = optuna.create_study(direction="minimize",
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
    study.optimize(narx_objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    na = best.user_attrs["na"]; nb = best.user_attrs["nb"]
    hl = best.user_attrs["hidden_layers"]; hn = best.user_attrs["hidden_units"]
    act = best.user_attrs["activation"]; lr = best.user_attrs["lr"]
    bs  = best.user_attrs["batch_size"]; ep = best.user_attrs["epochs"]

    print("\nBest NARX trial:")
    print(f"  na={na}, nb={nb}, hl={hl}, hn={hn}, act={act}, lr={lr:.5g}, bs={bs}, epochs={ep}")
    print(f"  val_sim_rms={best.value:.6f}, val_pred_rms={best.user_attrs.get('val_pred_rms'):.6f}")

    # Retrain on (train + val)
    Xtr, Ytr = create_IO_data(np.concatenate([u_tr, u_va]), np.concatenate([th_tr, th_va]), na, nb)
    model = NARX_MLP(input_dim=Xtr.shape[1], hidden_layers=hl, hidden_units=hn, activation=act)
    model = train_narx(model, Xtr, Ytr, epochs=ep, bs=bs, lr=lr)
    model.eval()

    # TRAIN prediction metrics (on the combined training set)
    with torch.no_grad():
        y_pred_train = model(torch.tensor(Xtr, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)
    print_train_pred_metrics(y_pred_train, Ytr, skip=max(na, nb))

    # TRAIN simulation metrics (on full series th_all/u_all)
    def _mlp_fn(x_np):
        x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            y = model(x_t)
        return y.squeeze().detach().cpu().item()

    skip_train = max(na, nb)
    th_train_sim = simulation_IO_model(_mlp_fn, list(u_all), list(th_all), skip=skip_train, na=na, nb=nb)
    print_train_sim_metrics(th_train_sim, th_all, skip_train)

    # HIDDEN prediction export
    if hidden_pred is not None and "upast" in hidden_pred and "thpast" in hidden_pred:
        up = hidden_pred["upast"].astype(np.float32)   # (N, 15)
        tp = hidden_pred["thpast"].astype(np.float32)  # (N, 15)
        if na > HIDDEN_PRED_L or nb > HIDDEN_PRED_L:
            print("⚠ Best (na,nb) exceeds 15; clipping to 15 for hidden prediction slicing.")
        Xh = np.concatenate([up[:, HIDDEN_PRED_L-nb:], tp[:, HIDDEN_PRED_L-na:]], axis=1).astype(np.float32)
        with torch.no_grad():
            thnow = model(torch.tensor(Xh, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1,1)
        out_pred = os.path.join(SAVE_DIR, "ann-narx_hidden-prediction_best_optuna.npz")
        np.savez(out_pred, upast=up, thpast=tp, thnow=thnow)
        print("  ↳ wrote", out_pred)

    # HIDDEN simulation export
    if hidden_sim is not None and "u" in hidden_sim and "th" in hidden_sim:
        u_test = hidden_sim["u"].astype(np.float32)
        th_init = hidden_sim["th"].astype(np.float32)
        th_sim = simulation_IO_model(_mlp_fn, list(u_test), list(th_init), skip=WARMUP_HIDDEN, na=na, nb=nb)
        out_sim = os.path.join(SAVE_DIR, "ann-narx_hidden-simulation_best_optuna.npz")
        np.savez(out_sim, u=u_test, th=th_sim)
        print("  ↳ wrote", out_sim)

# -----------------------
# LSTM + Optuna (optional)
# -----------------------
def train_lstm(model, X, Y, epochs=80, bs=128, lr=1e-3):
    model = model.to(device)
    ds = TensorDataset(torch.tensor(X), torch.tensor(Y).unsqueeze(1))
    loader = DataLoader(ds, batch_size=bs, shuffle=True)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model

def lstm_objective(trial: optuna.trial.Trial):
    # Fix seq_len=15 to match hidden prediction past window
    seq_len = 15
    hs  = trial.suggest_categorical("hidden_units", [32, 64, 96, 128])
    nl  = trial.suggest_int("num_layers", 1, 3)
    bi  = trial.suggest_categorical("bidirectional", [False, True])
    lr  = trial.suggest_float("lr", 5e-5, 3e-3, log=True)
    bs  = trial.suggest_categorical("batch_size", [64, 128, 256])
    ep  = trial.suggest_int("epochs", 60, 140, step=20)

    Xtr, Ytr = create_lstm_sequences(u_tr, th_tr, seq_len)
    Xva, Yva = create_lstm_sequences(u_va, th_va, seq_len)
    if len(Xtr) < 200 or len(Xva) < 100:
        raise optuna.TrialPruned()

    model = LSTM_Seq2One(hidden_units=hs, num_layers=nl, bidirectional=bi)
    model = train_lstm(model, Xtr, Ytr, epochs=ep, bs=bs, lr=lr)
    model.eval()

    # Validation simulation RMS (teacher-forced rolling): simulate free-run on val with skip=seq_len
    th_val_sim = simulate_lstm_free_run(model, list(u_va), list(th_va), seq_len=seq_len, skip=seq_len)
    err = th_val_sim[seq_len:] - th_va[seq_len:]
    val_sim_rms = float(np.sqrt(np.mean(err**2)))

    # Also log pred RMS on val
    with torch.no_grad():
        y_pred_val = model(torch.tensor(Xva, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)
    pred_rms = float(np.sqrt(np.mean((y_pred_val - Yva)**2)))

    trial.set_user_attr("seq_len", seq_len)
    trial.set_user_attr("hidden_units", hs)
    trial.set_user_attr("num_layers", nl)
    trial.set_user_attr("bidirectional", bi)
    trial.set_user_attr("lr", lr)
    trial.set_user_attr("batch_size", bs)
    trial.set_user_attr("epochs", ep)
    trial.set_user_attr("val_pred_rms", pred_rms)

    trial.report(val_sim_rms, step=1)
    if trial.should_prune():
        raise optuna.TrialPruned()
    return val_sim_rms

def run_lstm_optuna(n_trials=20):
    print("\n=== Optuna: LSTM tuning (seq_len=15) ===")
    study = optuna.create_study(direction="minimize",
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
    study.optimize(lstm_objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    hs = best.user_attrs["hidden_units"]
    nl = best.user_attrs["num_layers"]
    bi = best.user_attrs["bidirectional"]
    lr = best.user_attrs["lr"]
    bs = best.user_attrs["batch_size"]
    ep = best.user_attrs["epochs"]
    seq_len = best.user_attrs["seq_len"]

    print("\nBest LSTM trial:")
    print(f"  seq_len={seq_len}, hs={hs}, nl={nl}, bi={bi}, lr={lr:.5g}, bs={bs}, epochs={ep}")
    print(f"  val_sim_rms={best.value:.6f}, val_pred_rms={best.user_attrs.get('val_pred_rms'):.6f}")

    # Retrain on (train + val)
    Xtr_all, Ytr_all = create_lstm_sequences(np.concatenate([u_tr, u_va]),
                                             np.concatenate([th_tr, th_va]), seq_len)
    model = LSTM_Seq2One(hidden_units=hs, num_layers=nl, bidirectional=bi)
    model = train_lstm(model, Xtr_all, Ytr_all, epochs=ep, bs=bs, lr=lr)
    model.eval()

    # TRAIN prediction metrics (on sequences from whole data)
    Xtrain_seq, Ytrain_seq = create_lstm_sequences(u_all, th_all, seq_len)
    with torch.no_grad():
        y_pred_train = model(torch.tensor(Xtrain_seq, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)
    print_train_pred_metrics(y_pred_train, Ytrain_seq, skip=seq_len)

    # TRAIN simulation metrics on full series
    th_train_sim = simulate_lstm_free_run(model, list(u_all), list(th_all), seq_len=seq_len, skip=seq_len)
    print_train_sim_metrics(th_train_sim, th_all, seq_len)

    # HIDDEN prediction export
    if hidden_pred is not None and "upast" in hidden_pred and "thpast" in hidden_pred:
        up = hidden_pred["upast"].astype(np.float32)   # (N, 15)
        tp = hidden_pred["thpast"].astype(np.float32)  # (N, 15)
        Xh = np.stack([up, tp], axis=2)                # (N, 15, 2)
        with torch.no_grad():
            thnow = model(torch.tensor(Xh, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1,1)
        out_pred = os.path.join(SAVE_DIR, "ann-lstm_hidden-prediction_best_optuna.npz")
        np.savez(out_pred, upast=up, thpast=tp, thnow=thnow)
        print("  ↳ wrote", out_pred)

    # HIDDEN simulation export
    if hidden_sim is not None and "u" in hidden_sim and "th" in hidden_sim:
        u_test = hidden_sim["u"].astype(np.float32)
        th_init = hidden_sim["th"].astype(np.float32)
        th_sim = simulate_lstm_free_run(model, list(u_test), list(th_init), seq_len=seq_len, skip=WARMUP_HIDDEN)
        out_sim = os.path.join(SAVE_DIR, "ann-lstm_hidden-simulation_best_optuna.npz")
        np.savez(out_sim, u=u_test, th=th_sim)
        print("  ↳ wrote", out_sim)

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    if RUN_NARX_OPTUNA:
        run_narx_optuna(n_trials=25)     # adjust for speed/quality
    if RUN_LSTM_OPTUNA:
        run_lstm_optuna(n_trials=20)
    print("\nDone.")