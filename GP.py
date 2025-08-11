import os
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import gpytorch
import optuna

# -------------------------
# Constants / I/O
# -------------------------
WARMUP = 50  # warmup samples used for free-run simulation
SAVE_DIR = "disc-submission-files"
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------
# Utilities
# -------------------------
def to_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)

def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2*np.pi) - np.pi

def rmse(a, b):
    a = np.asarray(a).reshape(-1); b = np.asarray(b).reshape(-1)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def rmse_wrap(a, b):
    e = wrap_angle(a) - wrap_angle(b)
    return float(np.sqrt(np.mean(wrap_angle(e) ** 2)))

def construct_io_dataset(u, th, na, nb):
    s, c = np.sin(th).astype(np.float32), np.cos(th).astype(np.float32)
    X, y = [], []
    start = max(na, nb)
    for k in range(start, len(th)):
        X.append(np.concatenate([u[k-nb:k], s[k-na:k], c[k-na:k]], axis=0))
        y.append(th[k])
    return np.array(X, np.float32), np.array(y, np.float32)

def standardize_X(X, mu=None, sig=None):
    if mu is None:
        mu = X.mean(0)
        sig = X.std(0) + 1e-12
    return (X - mu) / sig, mu, sig

def align_length(arr, target_len):
    """Crop or pad (edge) to match target_len, if provided."""
    arr = np.asarray(arr).reshape(-1)
    if target_len is None:
        return arr
    L = len(arr)
    if L == target_len:
        return arr
    if L > target_len:
        return arr[:target_len]
    pad_val = arr[-1] if L > 0 else 0.0
    return np.pad(arr, (0, target_len - L), mode="edge", constant_values=pad_val)

def maybe_kmeans_Z(X, M, device, seed=0):
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=M, n_init=10, random_state=seed).fit(X)
        Z = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device)
    except Exception:
        perm = torch.randperm(X.shape[0], device=device)[:M]
        Z = torch.tensor(X[perm.cpu().numpy()], dtype=torch.float32, device=device)
    return Z

# -------------------------
# GPyTorch SVGP
# -------------------------
class NARXSVGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, input_dim):
        q = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        strat = gpytorch.variational.VariationalStrategy(
            self, inducing_points, q, learn_inducing_locations=True
        )
        super().__init__(strat)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )

def train_svgp(model, likelihood, train_loader, epochs=60, lr=0.01, device="cpu"):
    model.train(); likelihood.train()
    opt = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": likelihood.parameters()}],
        lr=lr
    )
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(train_loader.dataset))
    for _ in range(epochs):
        for xb, yb in train_loader:
            opt.zero_grad()
            out = model(xb)
            loss = -mll(out, yb)
            loss.backward()
            opt.step()
    return model, likelihood

@torch.no_grad()
def predict_svgp(model, likelihood, X, device="cpu", batch_size=4096):
    model.eval(); likelihood.eval()
    means = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i+batch_size]
        pred = likelihood(model(xb))
        means.append(pred.mean.detach().cpu())
    return torch.cat(means, dim=0).numpy()

def simulate_narx_model(u_sequence, y_initial, gp_model, likelihood, na, nb, device, x_mu, x_sig, skip=WARMUP):
    """Free-run sim with sin/cos; 'skip' controls warm-up length."""
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        y_initial = np.asarray(y_initial, dtype=np.float32)
        u_sequence = np.asarray(u_sequence, dtype=np.float32)

        y_sim = list(y_initial[:skip].copy())
        upast = list(u_sequence[skip - nb:skip].copy())
        ypast = y_sim[-na:].copy()

        for u_now in u_sequence[skip:]:
            sin_blk = np.sin(np.array(ypast[-na:], dtype=np.float32))
            cos_blk = np.cos(np.array(ypast[-na:], dtype=np.float32))
            x = np.concatenate([np.array(upast[-nb:], dtype=np.float32), sin_blk, cos_blk])[None, :]
            x = (x - x_mu) / x_sig
            pred = likelihood(gp_model(to_tensor(x, device)))
            mean = float(pred.mean.item())
            upast.append(float(u_now)); upast.pop(0)
            ypast.append(mean);         ypast.pop(0)
            y_sim.append(mean)
    return np.array(y_sim, dtype=np.float32)

# -------------------------
# training prints (according to example files)
# -------------------------
def print_train_prediction_metrics(y_pred_train, Ytrain):
    RMS = rmse(y_pred_train, Ytrain)
    print('train prediction errors:')
    print('RMS:', RMS, 'radians')
    print('RMS:', RMS/(2*np.pi)*360, 'degrees')
    print('NRMS:', RMS/np.std(Ytrain)*100, '%')

def print_train_simulation_metrics(th_sim, th_true, skip):
    err = th_sim[skip:] - th_true[skip:]
    RMS = float(np.sqrt(np.mean(err**2)))
    print('train simulation errors:')
    print('RMS:', RMS, 'radians')
    print('RMS:', RMS/(2*np.pi)*360, 'degrees')
    print('NRMS:', RMS/np.std(th_true)*100, '%')

# -------------------------
# Main + Optuna objective
# -------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    data = np.load('disc-benchmark-files/training-val-test-data.npz')
    th_all = data['th'].astype(np.float32)
    u_all  = data['u'].astype(np.float32)

    N = len(th_all)
    i_tr = int(0.8*N); i_va = int(0.9*N)
    th_tr, th_va, th_te = th_all[:i_tr], th_all[i_tr:i_va], th_all[i_va:]
    u_tr,  u_va,  u_te  = u_all[:i_tr],  u_all[i_tr:i_va],  u_all[i_va:]

    # Solution lengths (optional; align NPZ sizes if present)
    sim_sol_path  = 'disc-benchmark-files/test-simulation-solution-file.npz'
    pred_sol_path = 'disc-benchmark-files/test-prediction-solution-file.npz'
    sim_target_len  = np.load(sim_sol_path)['th'].size      if os.path.exists(sim_sol_path)  else None
    pred_target_len = np.load(pred_sol_path)['thnow'].size  if os.path.exists(pred_sol_path) else None

    # ----------------- Optuna objective -----------------
    best_snapshot = {"val_sim_rmse": float("inf")}  # we’ll also keep last trained model for preview

    def objective(trial: optuna.Trial) -> float:
        # Hyperparams
        na = trial.suggest_categorical("na", [2, 4, 5, 6, 8, 10, 12])
        nb = trial.suggest_categorical("nb", [2, 4, 5, 6, 8, 10, 12])
        M  = trial.suggest_categorical("M",  [50, 100, 200, 300])
        lr = trial.suggest_float("lr", 1e-3, 3e-2, log=True)
        epochs = trial.suggest_int("epochs", 60, 140, step=20)
        batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048])

        # Build datasets
        Xtr, Ytr = construct_io_dataset(u_tr, th_tr, na, nb)
        Xva, Yva = construct_io_dataset(u_va, th_va, na, nb)
        if len(Xtr) < 100 or len(Xva) < 50:
            return 1e9  # invalid config given data size

        # Normalize features on TRAIN
        Xtr_n, x_mu, x_sig = standardize_X(Xtr)
        Xva_n, _, _        = standardize_X(Xva, x_mu, x_sig)

        # DataLoader
        train_loader = DataLoader(TensorDataset(to_tensor(Xtr_n, device), to_tensor(Ytr, device)),
                                  batch_size=batch_size, shuffle=True, drop_last=False)

        # Model
        Z = maybe_kmeans_Z(Xtr_n, M, device, seed=0)
        model = NARXSVGP(Z, input_dim=Xtr_n.shape[1]).to(device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

        # Train
        train_svgp(model, likelihood, train_loader, epochs=epochs, lr=lr, device=device)

        # ---------- TRAIN prints (exact format) ----------
        with torch.no_grad():
            y_pred_train = predict_svgp(model, likelihood, to_tensor(Xtr_n, device), device=device).reshape(-1)
        print_train_prediction_metrics(y_pred_train, Ytr)

        skip_train = max(na, nb)
        th_train_sim = simulate_narx_model(u_tr, th_tr, model, likelihood, na, nb, device, x_mu, x_sig, skip=skip_train)
        print_train_simulation_metrics(th_train_sim, th_tr, skip_train)

        # ---------- VALIDATION objective (wrapped sim RMSE) ----------
        # (more robust for angles than plain RMSE)
        sim_va = simulate_narx_model(u_va, th_va, model, likelihood, na, nb, device, x_mu, x_sig, skip=max(na, nb))
        val_sim_rmse = rmse_wrap(sim_va, th_va)

        # Track best-by-val-sim
        if val_sim_rmse < best_snapshot["val_sim_rmse"]:
            best_snapshot.update({
                "val_sim_rmse": val_sim_rmse,
                "na": na, "nb": nb, "M": M, "lr": lr, "epochs": epochs, "batch_size": batch_size,
                "x_mu": x_mu.copy(), "x_sig": x_sig.copy(),
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "lik_state":   {k: v.detach().cpu().clone() for k, v in likelihood.state_dict().items()},
            })

        return val_sim_rmse

    # ----------------- Run study -----------------
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    best = study.best_trial.params
    print("\n=== Optuna best (by VAL wrapped Sim RMSE) ===")
    print(best, "| value:", study.best_value)

    # ----------------- Retrain on TRAIN+VAL with best config -----------------
    na = best["na"]; nb = best["nb"]; M = best["M"]
    lr = best["lr"]; epochs = best["epochs"]; batch_size = best["batch_size"]

    # Merge train+val
    th_trv = np.concatenate([th_tr, th_va], axis=0)
    u_trv  = np.concatenate([u_tr,  u_va],  axis=0)

    Xtrv, Ytrv = construct_io_dataset(u_trv, th_trv, na, nb)
    Xte,  Yte  = construct_io_dataset(u_te,  th_te,  na, nb)
    Xtrv_n, x_mu, x_sig = standardize_X(Xtrv)
    Xte_n,  _, _        = standardize_X(Xte,  x_mu, x_sig)

    train_loader = DataLoader(TensorDataset(to_tensor(Xtrv_n, device), to_tensor(Ytrv, device)),
                              batch_size=batch_size, shuffle=True, drop_last=False)
    Z = maybe_kmeans_Z(Xtrv_n, M, device, seed=0)
    model = NARXSVGP(Z, input_dim=Xtrv_n.shape[1]).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    train_svgp(model, likelihood, train_loader, epochs=epochs, lr=lr, device=device)

    # ----------------- Final TRAIN prints (on TRAIN+VAL, matching your format) -----------------
    with torch.no_grad():
        y_pred_trv = predict_svgp(model, likelihood, to_tensor(Xtrv_n, device), device=device).reshape(-1)
    print_train_prediction_metrics(y_pred_trv, Ytrv)

    skip_trv = max(na, nb)
    th_trv_sim = simulate_narx_model(u_trv, th_trv, model, likelihood, na, nb, device, x_mu, x_sig, skip=skip_trv)
    print_train_simulation_metrics(th_trv_sim, th_trv, skip_trv)

    # ----------------- TEST submissions (checker-ready) -----------------
    # (A) Simulation on test split (free-run from first WARMUP truth samples)
    th_test_sim = simulate_narx_model(u_te, th_te, model, likelihood, na, nb, device, x_mu, x_sig, skip=WARMUP)
    th_test_sim = align_length(th_test_sim, sim_target_len)
    sim_out = os.path.join(SAVE_DIR, f"gp_optuna_test-simulation_M{M}_na{na}_nb{nb}.npz")
    np.savez(sim_out, th=th_test_sim)

    # (B) Prediction on test split (one-step using IO features)
    yte_hat = predict_svgp(model, likelihood, to_tensor(Xte_n, device), device=device).reshape(-1)
    yte_hat = align_length(yte_hat, pred_target_len)
    pred_out = os.path.join(SAVE_DIR, f"gp_optuna_test-prediction_M{M}_na{na}_nb{nb}.npz")
    np.savez(pred_out, thnow=yte_hat.reshape(-1, 1))

    print(f"\nWrote best NPZs:")
    print("  ", sim_out)
    print("  ", pred_out)

if __name__ == "__main__":
    main()