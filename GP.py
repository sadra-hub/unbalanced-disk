# GP_torch.py
# PyTorch + GPyTorch SVGP (Sparse Variational GP) for NARX modeling.
# - Grid search over (na, nb, inducing M) with clear RMSE logs per iteration
# - Final training with best (by simulation RMSE) config
# - Evaluates on hidden prediction & hidden simulation files (if GT present)
# - Saves model as .pth and writes hidden-test submission .npz

import os, math, numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
import gpytorch

# =========================
# Utilities
# =========================
def to_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)

def rmse(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    return math.sqrt(np.mean((a - b) ** 2))

def construct_io_dataset(u_data, y_data, na, nb):
    inputs, targets = [], []
    start = max(na, nb)
    for k in range(start, len(y_data)):
        io_vec = np.concatenate([u_data[k - nb:k], y_data[k - na:k]])
        inputs.append(io_vec)
        targets.append(y_data[k])
    X = np.array(inputs, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    return X, y

def simulate_narx_model(u_sequence, y_initial, gp_model, likelihood, na, nb, device):
    """
    Closed-loop (free-run) simulation:
    - uses first 50 y samples as warm-up (like original)
    - evolves with predicted y
    """
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        y_initial = np.asarray(y_initial, dtype=np.float32)
        u_sequence = np.asarray(u_sequence, dtype=np.float32)

        y_simulated = list(y_initial[:50].copy())
        past_u = list(u_sequence[50 - nb:50].copy())
        past_y = y_simulated[-na:].copy()

        for u_current in u_sequence[50:]:
            x = np.concatenate([past_u, past_y])[None, :]
            x_t = to_tensor(x, device)
            pred = likelihood(gp_model(x_t))
            mean = pred.mean.item()

            past_u.append(float(u_current)); past_u.pop(0)
            past_y.append(mean); past_y.pop(0)
            y_simulated.append(mean)

    return np.array(y_simulated, dtype=np.float32)

# =========================
# SVGP Model
# =========================
class NARXSVGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_dist = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_dist, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_svgp(model, likelihood, train_loader, epochs=60, lr=0.01, device="cpu"):
    model.train(); likelihood.train()
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": likelihood.parameters()}],
        lr=lr
    )
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(train_loader.dataset))
    for ep in range(epochs):
        total = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = -mll(out, yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
        # Uncomment for per-epoch loss:
        # print(f"  Epoch {ep+1:03d}/{epochs} | ELBO {-total/len(train_loader.dataset):.4f}")
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

# Helpers to robustly fetch keys from hidden files
def get_first_key(dct, candidates):
    for k in candidates:
        if k in dct:
            return k
    return None

# =========================
# Main
# =========================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("models", exist_ok=True)
    os.makedirs("disc-submission-files", exist_ok=True)

    # Load data
    npz_data = np.load('disc-benchmark-files/training-val-test-data.npz')
    th_all = npz_data['th'].astype(np.float32)
    u_all  = npz_data['u'].astype(np.float32)

    N = len(th_all)
    train_split = int(0.8 * N)
    valid_split = int(0.9 * N)

    th_train, th_valid, th_test = th_all[:train_split], th_all[train_split:valid_split], th_all[valid_split:]
    u_train,  u_valid,  u_test  = u_all[:train_split],  u_all[train_split:valid_split],  u_all[valid_split:]

    # Hyperparameter grids
    na_nb_candidates = [2, 5, 8]
    inducing_point_counts = [300, 100, 50]

    best_pred = {"rmse": float("inf")}
    best_sim  = {"rmse": float("inf")}

    print("\n=== Starting Grid Search over (na, nb, M) ===")
    for num_inducing in inducing_point_counts:
        for na in na_nb_candidates:
            for nb in na_nb_candidates:
                # Build datasets
                Xtr, ytr = construct_io_dataset(u_train, th_train, na, nb)
                Xva, yva = construct_io_dataset(u_valid, th_valid, na, nb)

                if len(Xtr) < 100 or len(Xva) < 100:
                    print(f"⚠ Skipping na={na}, nb={nb}, M={num_inducing} (insufficient samples)")
                    continue

                # Tensors + loader
                Xtr_t = to_tensor(Xtr, device)
                ytr_t = to_tensor(ytr, device)
                train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=2048, shuffle=True)

                # Inducing points from training inputs (stable init)
                perm = torch.randperm(Xtr_t.size(0), device=device)[:num_inducing]
                Z = Xtr_t[perm].clone().detach()

                model = NARXSVGP(Z).to(device)
                likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

                # Train
                train_svgp(model, likelihood, train_loader, epochs=60, lr=0.01, device=device)

                # Prediction RMSE (validation)
                Xva_t = to_tensor(Xva, device)
                yva_hat = predict_svgp(model, likelihood, Xva_t, device=device)
                pred_rmse = rmse(yva_hat, yva)

                # Simulation RMSE (validation)
                sim_valid = simulate_narx_model(list(u_valid), list(th_valid), model, likelihood, na, nb, device)
                sim_rmse = rmse(sim_valid, th_valid)

                # Log this config
                print(
                    f"[Grid] M={num_inducing:4d}, na={na:2d}, nb={nb:2d} "
                    f"| Pred RMSE={pred_rmse:.4f} | Sim RMSE={sim_rmse:.4f} "
                    f"| BestPred={best_pred['rmse']:.4f} | BestSim={best_sim['rmse']:.4f}"
                )

                # Update bests
                if pred_rmse < best_pred["rmse"]:
                    best_pred = {"rmse": pred_rmse, "na": na, "nb": nb, "M": num_inducing}
                if sim_rmse < best_sim["rmse"]:
                    best_sim  = {"rmse": sim_rmse,  "na": na, "nb": nb, "M": num_inducing}

    print("\n=== Best by Prediction ===", best_pred)
    print("=== Best by Simulation ===", best_sim)

    # Choose final config (by simulation best; change to best_pred if you prefer)
    na = best_sim.get("na", 8)
    nb = best_sim.get("nb", 2)
    inducing = best_sim.get("M", 100)

    # Final train on train split (you can merge train+valid if you want)
    Xtr, ytr = construct_io_dataset(u_train, th_train, na, nb)
    Xva, yva = construct_io_dataset(u_valid, th_valid, na, nb)
    Xte, yte = construct_io_dataset(u_test,  th_test,  na, nb)

    Xtr_t = to_tensor(Xtr, device)
    ytr_t = to_tensor(ytr, device)
    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=4096, shuffle=True)

    perm = torch.randperm(Xtr_t.size(0), device=device)[:inducing]
    Z = Xtr_t[perm].clone().detach()

    model = NARXSVGP(Z).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    train_svgp(model, likelihood, train_loader, epochs=120, lr=0.01, device=device)

    # Validation metrics
    yva_hat = predict_svgp(model, likelihood, to_tensor(Xva, device), device=device)
    rmse_valid_pred = rmse(yva_hat, yva)
    sim_valid = simulate_narx_model(list(u_valid), list(th_valid), model, likelihood, na, nb, device)
    rmse_valid_sim = rmse(sim_valid, th_valid)
    print(f"\nValidation Prediction RMSE: {rmse_valid_pred:.4f}")
    print(f"Validation Simulation RMSE: {rmse_valid_sim:.4f}")

    # Test metrics
    yte_hat = predict_svgp(model, likelihood, to_tensor(Xte, device), device=device)
    rmse_test_pred = rmse(yte_hat, yte)
    sim_test = simulate_narx_model(list(u_test), list(th_test), model, likelihood, na, nb, device)
    rmse_test_sim = rmse(sim_test, th_test)
    print(f"Test Prediction RMSE: {rmse_test_pred:.4f}")
    print(f"Test Simulation RMSE: {rmse_test_sim:.4f}")

    # =========================
    # Hidden-set EVALUATION
    # =========================

    # ---- Hidden Prediction (point-wise) ----
    pred_hidden_path = 'disc-benchmark-files/hidden-test-prediction-submission-file.npz'
    if os.path.exists(pred_hidden_path):
        hidden_pred = np.load(pred_hidden_path)
        # Required inputs
        upast = hidden_pred['upast']
        thpast = hidden_pred['thpast']
        X_hidden = np.concatenate([upast[:, 15 - nb:], thpast[:, 15 - na:]], axis=1).astype(np.float32)
        y_hidden_pred = predict_svgp(model, likelihood, to_tensor(X_hidden, device), device=device).reshape(-1, 1)

        # Try to find ground-truth key if provided
        gt_key = get_first_key(hidden_pred, ['thnow_true', 'thnow', 'y_true', 'y', 'target'])
        if gt_key is not None:
            gt = hidden_pred[gt_key].astype(np.float32).reshape(-1, 1)
            if gt.shape[0] == y_hidden_pred.shape[0]:
                rmse_hidden_pred = rmse(y_hidden_pred, gt)
                print(f"Hidden Prediction RMSE ({gt_key}): {rmse_hidden_pred:.4f}")
            else:
                print(f"Hidden Prediction: GT length {gt.shape[0]} != pred length {y_hidden_pred.shape[0]} (skip RMSE)")
        else:
            print("Hidden Prediction: no ground-truth key found (thnow_true/thnow/y_true/y/target). Skipping RMSE.")

        # Also write submission file
        np.savez(
            'disc-submission-files/sparse-gp-hidden-test-prediction-submission-file.npz',
            upast=upast, thpast=thpast, thnow=y_hidden_pred
        )
        print("Wrote hidden prediction submission npz.")
    else:
        print(f"Hidden prediction file not found at {pred_hidden_path} — skipping.")

    # ---- Hidden Simulation (free-run sequence) ----
    sim_hidden_path = 'disc-benchmark-files/hidden-test-simulation-submission-file.npz'
    if os.path.exists(sim_hidden_path):
        hidden_sim = np.load(sim_hidden_path)

        # Try to extract keys robustly
        u_key  = get_first_key(hidden_sim, ['u', 'u_sequence', 'u_valid', 'u_test'])
        th_key = get_first_key(hidden_sim, ['th_true', 'th', 'y_true', 'y'])
        th0_key = get_first_key(hidden_sim, ['th_initial', 'y_initial', 'th0'])

        if u_key is None:
            print("Hidden Simulation: no input sequence key found (u/u_sequence/...). Skipping.")
        else:
            u_seq = hidden_sim[u_key].astype(np.float32)

            # Warmup initial y: prefer explicit th_initial; else use first 50 of provided th_true if available
            if th0_key is not None:
                th_init = hidden_sim[th0_key].astype(np.float32)
            elif th_key is not None:
                th_init = hidden_sim[th_key].astype(np.float32)[:50]
            else:
                # last resort: zeros warmup
                th_init = np.zeros(50, dtype=np.float32)
                print("Hidden Simulation: no th_initial/th_true provided; using zeros for warmup 50 samples.")

            sim_hidden = simulate_narx_model(list(u_seq), list(th_init), model, likelihood, na, nb, device)

            # If GT full sequence exists and lengths match, compute RMSE
            if th_key is not None:
                th_true = hidden_sim[th_key].astype(np.float32)
                if len(sim_hidden) == len(th_true):
                    rmse_hidden_sim = rmse(sim_hidden, th_true)
                    print(f"Hidden Simulation RMSE ({th_key}): {rmse_hidden_sim:.4f}")
                else:
                    print(f"Hidden Simulation: GT length {len(th_true)} != sim length {len(sim_hidden)} (skip RMSE)")
            else:
                print("Hidden Simulation: no ground-truth key (th_true/th) found. Skipping RMSE.")

            # Save simulation submission
            np.savez('disc-submission-files/sparse-gp-hidden-test-simulation-submission-file.npz',
                     th=sim_hidden)
            print("Wrote hidden simulation submission npz.")
    else:
        print(f"Hidden simulation file not found at {sim_hidden_path} — skipping.")

    # Save PyTorch model (.pth)
    save_path = 'disc-submission-files/gp_narx_svgp.pth'
    torch.save({
        "state_dict": model.state_dict(),
        "likelihood_state_dict": likelihood.state_dict(),
        "na": na,
        "nb": nb,
        "inducing_points": model.variational_strategy.inducing_points.detach().cpu(),
        "input_dim": Xtr.shape[1],
    }, save_path)
    print(f"Saved model to {save_path}")

if __name__ == "__main__":
    main()