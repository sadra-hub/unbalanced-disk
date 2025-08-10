# bc_policy.py
# Behavior Cloning: learn u = pi(theta, omega) from logs (theta, omega, u).
# Uses [sin(theta), cos(theta), omega] as inputs for stability.

import os, glob, csv, math
import numpy as np
import torch
import torch.nn as nn
from UnbalancedDisk import UnbalancedDisk

# ======================
# Config
# ======================
LOGS_DIR = "logs"
OUT_DIR  = "disc-submission-files"
MODEL_PATH  = os.path.join(OUT_DIR, "bc_policy.pt")
SCALER_PATH = os.path.join(OUT_DIR, "bc_policy_scaler.npz")

DT = 0.005
UMAX = 3.0

# Training
EPOCHS = 60
BATCH  = 1024
LR     = 2e-3
WD     = 1e-6
VAL_SPLIT = 0.15
PATIENCE  = 8
HIDDEN    = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

# ======================
# IO helpers
# ======================
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_csv(path):
    with open(path, "r") as f:
        r = csv.DictReader(f)
        rows = [(float(x["theta"]), float(x["omega"]), float(x["u"])) for x in r]
    return np.array(rows, dtype=np.float64) if rows else None

def load_npz(path):
    d = np.load(path)
    th, om, u = d["theta"], d["omega"], d["u"]
    T = min(len(th), len(om), len(u))
    return np.stack([th[:T], om[:T], u[:T]], axis=1).astype(np.float64)

def load_all_trajs(logs_dir=LOGS_DIR):
    trajs=[]
    for p in glob.glob(os.path.join(logs_dir, "*.csv")):
        try:
            a=load_csv(p)
            if a is not None and len(a)>5: trajs.append(a)
        except: pass
    for p in glob.glob(os.path.join(logs_dir, "*.npz")):
        try:
            a=load_npz(p)
            if a is not None and len(a)>5: trajs.append(a)
        except: pass
    if not trajs: raise FileNotFoundError(f"No logs in {logs_dir}")
    return trajs

# ======================
# Dataset (stateless BC)
# Inputs: [sinθ, cosθ, ω], Target: u
# ======================
def build_supervised(trajs):
    Xs, Ys = [], []
    for tr in trajs:
        th, om, u = tr[:,0], tr[:,1], tr[:,2]
        s, c = np.sin(th), np.cos(th)
        Xs.append(np.stack([s, c, om], axis=1))
        Ys.append(u[:, None])
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    return X, Y  # X:(N,3) Y:(N,1)

def split_train_val(X, Y, val_split=VAL_SPLIT):
    N = len(X); idx = np.arange(N)
    np.random.shuffle(idx)
    n_val = int(round(val_split * N))
    val_idx = idx[:n_val]; tr_idx = idx[n_val:]
    return X[tr_idx], Y[tr_idx], X[val_idx], Y[val_idx]

def fit_scalers(Xtr, Ytr):
    x_mean = Xtr.mean(axis=0); x_std = Xtr.std(axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    y_mean = Ytr.mean(axis=0); y_std = Ytr.std(axis=0)
    y_std = np.where(y_std < 1e-8, 1.0, y_std)
    return x_mean, x_std, y_mean, y_std

def standardize(X, m, s): return (X - m) / s

# ======================
# Model
# ======================
class BCPolicy(nn.Module):
    def __init__(self, in_dim=3, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)  # predict u (standardized)
        )
    def forward(self, x): return self.net(x)

# ======================
# Training utils
# ======================
def to_tensor(x): return torch.tensor(x, dtype=torch.float32, device=DEVICE)

def iterate_minibatches(X, Y, batch, shuffle=True):
    idx = np.arange(len(X))
    if shuffle: np.random.shuffle(idx)
    for i in range(0, len(X), batch):
        j = idx[i:i+batch]
        yield X[j], Y[j]

def train(model, Xtr, Ytr, Xval, Yval, epochs=EPOCHS, lr=LR, wd=WD):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.SmoothL1Loss()  # robust vs outliers

    best = {"val": float("inf"), "state": None, "epoch": 0}
    patience = PATIENCE

    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for xb, yb in iterate_minibatches(Xtr, Ytr, BATCH, shuffle=True):
            xb = to_tensor(xb); yb = to_tensor(yb)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(Xtr)

        model.eval()
        with torch.no_grad():
            pred = model(to_tensor(Xval))
            val_loss = loss_fn(pred, to_tensor(Yval)).item()

        print(f"[{ep:03d}/{epochs}] train={tr_loss:.6f}  val={val_loss:.6f}")

        if val_loss < best["val"] - 1e-6:
            best.update(val=val_loss, state=model.state_dict(), epoch=ep)
            patience = PATIENCE
        else:
            patience -= 1
            if patience <= 0:
                print(f"Early stopping at epoch {ep}. Best val={best['val']:.6f} @ {best['epoch']}")
                break

    model.load_state_dict(best["state"])
    return model, best

# ======================
# Rollout test in env
# ======================
@torch.no_grad()
def act(model, obs, x_mean, x_std, y_mean, y_std, umax=UMAX):
    th, om = float(obs[0]), float(obs[1])
    x = np.array([math.sin(th), math.cos(th), om], dtype=np.float32)[None, :]
    xz = (x - x_mean) / x_std
    uz = model(to_tensor(xz)).cpu().numpy()[0,0]
    u  = float(uz * y_std[0] + y_mean[0])
    return float(np.clip(u, -umax, umax))

def rollout_env(model, steps=1500, dt=DT, umax=UMAX, render=False):
    env = UnbalancedDisk(dt=dt, umax=umax, render_mode=("human" if render else None))
    obs, _ = env.reset()
    returns = 0.0
    for k in range(steps):
        u = act(model, obs, rollout_env.x_mean, rollout_env.x_std, rollout_env.y_mean, rollout_env.y_std, umax)
        obs, r, term, trunc, _ = env.step(u)
        returns += float(r)
        if render: env.render()
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    return returns

# ======================
# Main
# ======================
if __name__ == "__main__":
    ensure_dir(OUT_DIR)
    trajs = load_all_trajs(LOGS_DIR)
    print(f"Loaded {len(trajs)} trajectories")

    X, Y = build_supervised(trajs)
    print(f"Dataset: X={X.shape}, Y={Y.shape}")

    # Train/val split
    Xtr, Ytr, Xval, Yval = split_train_val(X, Y)

    # Fit scalers on TRAIN only; standardize targets too
    x_mean, x_std, y_mean, y_std = fit_scalers(Xtr, Ytr)
    Xtr_z = standardize(Xtr, x_mean, x_std)
    Xval_z = standardize(Xval, x_mean, x_std)
    Ytr_z  = standardize(Ytr, y_mean, y_std)
    Yval_z = standardize(Yval, y_mean, y_std)

    # Train
    model = BCPolicy(in_dim=3, hidden=HIDDEN)
    model, best = train(model, Xtr_z, Ytr_z, Xval_z, Yval_z)

    # Save
    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": 3,
        "hidden": HIDDEN,
        "x_mean": x_mean.astype(np.float32),
        "x_std":  x_std.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "y_std":  y_std.astype(np.float32),
        "umax": UMAX,
    }, MODEL_PATH)
    np.savez(SCALER_PATH, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    print(f"Saved policy -> {MODEL_PATH}")
    print(f"Saved scalers -> {SCALER_PATH}")

    # Quick rollout (no render by default)
    # stash scalers for rollout_env()
    rollout_env.x_mean, rollout_env.x_std = x_mean, x_std
    rollout_env.y_mean, rollout_env.y_std = y_mean, y_std
    ret = rollout_env(model, steps=1500, render=False)
    print(f"Rollout return (no render): {ret:.3f}")

    print("\nTip: if it jitters near top, try HIDDEN=256 or add a small history "
          "[sinθ, cosθ, ω, u_{t-1}] as inputs for damping.")