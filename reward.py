import os
import glob
import csv
import math
import numpy as np
import torch
import torch.nn as nn

LOGS_DIR = "logs"
MODEL_SAVE_PATH = "disc-submission-files/reward_model.npz"

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def load_npz_logs(pattern):
    X = []
    for p in glob.glob(pattern):
        try:
            d = np.load(p)
            th = d["theta"]
            om = d["omega"]
            u = d["u"]
            T = min(len(th), len(om), len(u))
            X.append(np.stack([th[:T], om[:T], u[:T]], axis=1))
        except:
            pass
    return X

def load_csv_logs(pattern):
    X = []
    for p in glob.glob(pattern):
        try:
            with open(p, "r") as f:
                r = csv.DictReader(f)
                rows = [(float(row["theta"]), float(row["omega"]), float(row["u"])) for row in r]
            if rows:
                X.append(np.array(rows, dtype=np.float64))
        except:
            pass
    return X

def collect_expert_pairs(logs_dir=LOGS_DIR):
    data = []
    data += load_npz_logs(os.path.join(logs_dir, "*.npz"))
    data += load_csv_logs(os.path.join(logs_dir, "*.csv"))
    if not data:
        raise FileNotFoundError(f"No logs found in '{logs_dir}'.")
    return np.concatenate(data, axis=0)  # (N, 3): [theta, omega, u]

def features(theta, omega, u):
    return np.stack([
        theta,
        theta**2,
        np.sin(theta),
        np.cos(theta),
        omega,
        omega**2,
        u,
        u**2,
        np.abs(theta*omega),
        np.abs(u)*np.abs(theta),
        np.abs(u)*np.abs(omega),
        np.sign(u)*np.abs(theta),
    ], axis=-1)

def standardize(X, mean=None, std=None, eps=1e-8):
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return (X - mean) / std, mean, std

class LogisticLinear(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.w = nn.Linear(in_dim, 1, bias=True)

    def forward(self, x):
        return self.w(x).squeeze(-1)

def make_noise_pairs(states_actions, umax=None, scale=1.0):
    th = states_actions[:, 0]
    om = states_actions[:, 1]
    if umax is None:
        umax = max(1.0, np.percentile(np.abs(states_actions[:, 2]), 95))
    u_noise = np.random.uniform(-umax*scale, umax*scale, size=th.shape[0])
    return np.stack([th, om, u_noise], axis=1)

def train_irl_linear(expert_sa, steps=4000, lr=1e-2, batch=1024):
    th_e, om_e, u_e = expert_sa[:, 0], expert_sa[:, 1], expert_sa[:, 2]
    Phi_e = features(th_e, om_e, u_e)
    Phi_e_std, phi_mean, phi_std = standardize(Phi_e)

    N, D = Phi_e_std.shape
    model = LogisticLinear(D)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for t in range(steps):
        idx = np.random.randint(0, N, size=min(batch, N))
        xb_e = torch.tensor(Phi_e_std[idx], dtype=torch.float32)
        yb_e = torch.ones(xb_e.shape[0])

        noise_sa = make_noise_pairs(expert_sa[idx], umax=np.max(np.abs(expert_sa[:, 2])))
        Phi_n = features(noise_sa[:, 0], noise_sa[:, 1], noise_sa[:, 2])
        Phi_n_std = (Phi_n - phi_mean) / phi_std
        xb_n = torch.tensor(Phi_n_std, dtype=torch.float32)
        yb_n = torch.zeros(xb_n.shape[0])

        xb = torch.cat([xb_e, xb_n], dim=0)
        yb = torch.cat([yb_e, yb_n], dim=0)

        logits = model(xb)
        loss = loss_fn(logits, yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

    w = model.w.weight.detach().numpy().reshape(-1)
    b = model.w.bias.detach().numpy().item()
    return w, b, phi_mean, phi_std

if __name__ == "__main__":
    ensure_dir("disc-submission-files")

    expert = collect_expert_pairs(LOGS_DIR)
    w, b, phi_mean, phi_std = train_irl_linear(expert)

    # Save model parameters
    np.savez(MODEL_SAVE_PATH, weights=w, bias=b, phi_mean=phi_mean, phi_std=phi_std)

    # Print learned reward function
    feature_names = [
        "theta", "theta^2", "sin(theta)", "cos(theta)",
        "omega", "omega^2", "u", "u^2",
        "|theta*omega|", "|u|*|theta|", "|u|*|omega|", "sign(u)*|theta|"
    ]
    print("\nLearned Reward Function:")
    print("R(theta, omega, u) = sum_i w[i] * phi_i(theta, omega, u) + b")
    for name, wi in zip(feature_names, w):
        print(f"{name:20s} : {wi:+.6f}")
    print(f"Bias{' '*16}: {b:+.6f}")
    print(f"\nModel saved to {MODEL_SAVE_PATH}")