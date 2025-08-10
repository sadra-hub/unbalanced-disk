# GP_torch.py  — SVGP NARX with sin/cos features, wrapped RMSE, ARD RBF, X-normalization
import os, math, numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
import gpytorch

# =========================
# Utils
# =========================
def to_tensor(x, device): return torch.as_tensor(x, dtype=torch.float32, device=device)

def rmse(a, b):
    a = np.asarray(a).reshape(-1); b = np.asarray(b).reshape(-1)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def wrap_angle(a): return (np.asarray(a) + np.pi) % (2*np.pi) - np.pi
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

def simulate_narx_model(u_sequence, y_initial, gp_model, likelihood, na, nb, device, x_mu, x_sig):
    """Free-run sim with sin/cos features, 50-sample warmup."""
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        y_initial = np.asarray(y_initial, dtype=np.float32)
        u_sequence = np.asarray(u_sequence, dtype=np.float32)

        y_sim = list(y_initial[:50].copy())
        upast = list(u_sequence[50 - nb:50].copy())
        ypast = y_sim[-na:].copy()

        for u_now in u_sequence[50:]:
            sin_blk = np.sin(np.array(ypast[-na:], dtype=np.float32))
            cos_blk = np.cos(np.array(ypast[-na:], dtype=np.float32))
            x = np.concatenate([np.array(upast[-nb:], dtype=np.float32), sin_blk, cos_blk])[None, :]
            x = (x - x_mu) / x_sig
            x_t = to_tensor(x, device)
            pred = likelihood(gp_model(x_t))
            mean = pred.mean.item()

            upast.append(float(u_now)); upast.pop(0)
            ypast.append(mean);         ypast.pop(0)
            y_sim.append(mean)
    return np.array(y_sim, dtype=np.float32)

# -------- NPZ save helpers --------
def _encode_key(k: str) -> str:
    # np.savez keys can include dots, but encoding avoids surprises
    return k.replace(".", "__")

def _state_to_numpy_dict(state):
    out = {}
    for k, v in state.items():
        if torch.is_tensor(v):
            out[_encode_key(k)] = v.detach().cpu().numpy()
        else:
            # buffers like None shouldn't appear, guard anyway
            if v is not None:
                out[_encode_key(k)] = np.array(v)
    return out

def save_svgp_npz(path, model_state, likelihood_state, meta: dict):
    blob = {}
    blob.update(_state_to_numpy_dict(model_state))
    for k, v in _state_to_numpy_dict(likelihood_state).items():
        blob[f"likelihood__{k}"] = v
    # meta fields
    for k, v in meta.items():
        if isinstance(v, (int, float, np.ndarray)):
            blob[f"meta__{k}"] = np.array(v)
        else:
            # lists/tuples -> arrays; strings -> small object array
            try:
                blob[f"meta__{k}"] = np.array(v)
            except Exception:
                blob[f"meta__{k}"] = np.array(str(v))
    np.savez_compressed(path, **blob)

# =========================
# SVGP model
# =========================
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

def get_first_key(dct, candidates):
    for k in candidates:
        if k in dct: return k
    return None

def maybe_kmeans_Z(X, M, device, seed=0):
    try:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=M, n_init=10, random_state=seed).fit(X)
        Z = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
    except Exception:
        perm = torch.randperm(X.shape[0], device=device)[:M]
        Z = torch.tensor(X[perm.cpu().numpy()], dtype=torch.float32, device=device)
    return Z

# =========================
# Main
# =========================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("models", exist_ok=True)
    os.makedirs("disc-submission-files", exist_ok=True)

    # Load
    data = np.load('disc-benchmark-files/training-val-test-data.npz')
    th_all = data['th'].astype(np.float32)
    u_all  = data['u'].astype(np.float32)

    N = len(th_all)
    i_tr = int(0.8*N); i_va = int(0.9*N)
    th_tr, th_va, th_te = th_all[:i_tr], th_all[i_tr:i_va], th_all[i_va:]
    u_tr,  u_va,  u_te  = u_all[:i_tr],  u_all[i_tr:i_va],  u_all[i_va:]

    # Grids
    na_grid = [2, 5, 8]
    nb_grid = [2, 5, 8]
    M_grid  = [300, 100, 50]

    best_pred = {"rmse": float("inf")}
    best_sim  = {"rmse": float("inf")}
    best_pred_snap = None
    best_sim_snap  = None

    print("\n=== Starting Grid Search over (na, nb, M) ===")
    for M in M_grid:
        for na in na_grid:
            for nb in nb_grid:
                # Datasets (sin/cos)
                Xtr, ytr = construct_io_dataset(u_tr, th_tr, na, nb)
                Xva, yva = construct_io_dataset(u_va, th_va, na, nb)
                if len(Xtr) < 100 or len(Xva) < 100:
                    print(f"⚠ Skip na={na}, nb={nb}, M={M} — not enough samples"); continue

                # Standardize X
                Xtr_n, x_mu, x_sig = standardize_X(Xtr)
                Xva_n, _, _        = standardize_X(Xva, x_mu, x_sig)

                Xtr_t = to_tensor(Xtr_n, device)
                ytr_t = to_tensor(ytr, device)
                train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=2048, shuffle=True)

                # Inducing points
                Z = maybe_kmeans_Z(Xtr_n, M, device, seed=0)

                model = NARXSVGP(Z, input_dim=Xtr_n.shape[1]).to(device)
                likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

                # Train
                train_svgp(model, likelihood, train_loader, epochs=60, lr=0.01, device=device)

                # One-step val (pred)
                yva_hat = predict_svgp(model, likelihood, to_tensor(Xva_n, device), device=device)
                pred_rmse = rmse_wrap(yva_hat, yva)

                # Free-run val (sim)
                sim_va = simulate_narx_model(u_va, th_va, model, likelihood, na, nb, device, x_mu, x_sig)
                sim_rmse = rmse_wrap(sim_va, th_va)

                print(f"[Grid] M={M:4d}, na={na:2d}, nb={nb:2d} | Pred RMSE={pred_rmse:.4f} | Sim RMSE={sim_rmse:.4f} "
                      f"| BestPred={best_pred['rmse']:.4f} | BestSim={best_sim['rmse']:.4f}")

                # Update best-by-pred snapshot
                if pred_rmse < best_pred["rmse"]:
                    best_pred = {"rmse": pred_rmse, "na": na, "nb": nb, "M": M}
                    # snapshot states (CPU copies)
                    best_pred_snap = {
                        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "lik_state":   {k: v.detach().cpu().clone() for k, v in likelihood.state_dict().items()},
                        "x_mu": x_mu.copy(), "x_sig": x_sig.copy(),
                        "input_dim": Xtr_n.shape[1],
                        "rmse_pred": pred_rmse, "rmse_sim": sim_rmse,
                    }

                # Update best-by-sim snapshot
                if sim_rmse < best_sim["rmse"]:
                    best_sim  = {"rmse": sim_rmse,  "na": na, "nb": nb, "M": M}
                    best_sim_snap = {
                        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "lik_state":   {k: v.detach().cpu().clone() for k, v in likelihood.state_dict().items()},
                        "x_mu": x_mu.copy(), "x_sig": x_sig.copy(),
                        "input_dim": Xtr_n.shape[1],
                        "rmse_pred": pred_rmse, "rmse_sim": sim_rmse,
                    }

    # ---- Save NPZ snapshots from grid ----
    os.makedirs("models", exist_ok=True)
    if best_pred_snap is not None:
        save_svgp_npz(
            "models/gp_svgp_best_pred.npz",
            best_pred_snap["model_state"],
            best_pred_snap["lik_state"],
            {
                "na": best_pred["na"], "nb": best_pred["nb"], "M": best_pred["M"],
                "input_dim": best_pred_snap["input_dim"],
                "x_mu": best_pred_snap["x_mu"], "x_sig": best_pred_snap["x_sig"],
                "rmse_pred": best_pred_snap["rmse_pred"], "rmse_sim": best_pred_snap["rmse_sim"],
                "criterion": "best_pred"
            }
        )
        print("Saved grid best-by-pred model → models/gp_svgp_best_pred.npz")
    if best_sim_snap is not None:
        save_svgp_npz(
            "models/gp_svgp_best_sim.npz",
            best_sim_snap["model_state"],
            best_sim_snap["lik_state"],
            {
                "na": best_sim["na"], "nb": best_sim["nb"], "M": best_sim["M"],
                "input_dim": best_sim_snap["input_dim"],
                "x_mu": best_sim_snap["x_mu"], "x_sig": best_sim_snap["x_sig"],
                "rmse_pred": best_sim_snap["rmse_pred"], "rmse_sim": best_sim_snap["rmse_sim"],
                "criterion": "best_sim"
            }
        )
        print("Saved grid best-by-sim model → models/gp_svgp_best_sim.npz")

    print("\n=== Best by Prediction ===", best_pred)
    print("=== Best by Simulation ===", best_sim)

    # -------- Final train on best-sim config (kept as in your script) --------
    na = best_sim.get("na", 8); nb = best_sim.get("nb", 2); M = best_sim.get("M", 100)

    Xtr, ytr = construct_io_dataset(u_tr, th_tr, na, nb)
    Xva, yva = construct_io_dataset(u_va, th_va, na, nb)
    Xte, yte = construct_io_dataset(u_te, th_te, na, nb)
    Xtr_n, x_mu, x_sig = standardize_X(Xtr)
    Xva_n, _, _ = standardize_X(Xva, x_mu, x_sig)
    Xte_n, _, _ = standardize_X(Xte, x_mu, x_sig)

    Xtr_t = to_tensor(Xtr_n, device); ytr_t = to_tensor(ytr, device)
    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=4096, shuffle=True)

    Z = maybe_kmeans_Z(Xtr_n, M, device, seed=0)
    model = NARXSVGP(Z, input_dim=Xtr_n.shape[1]).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    train_svgp(model, likelihood, train_loader, epochs=120, lr=0.01, device=device)

    # Validation
    yva_hat = predict_svgp(model, likelihood, to_tensor(Xva_n, device), device=device)
    rmse_valid_pred = rmse_wrap(yva_hat, yva)
    sim_valid = simulate_narx_model(u_va, th_va, model, likelihood, na, nb, device, x_mu, x_sig)
    rmse_valid_sim = rmse_wrap(sim_valid, th_va)
    print(f"\nValidation Prediction RMSE (wrap): {rmse_valid_pred:.4f}")
    print(f"Validation Simulation RMSE (wrap): {rmse_valid_sim:.4f}")

    # Test
    yte_hat = predict_svgp(model, likelihood, to_tensor(Xte_n, device), device=device)
    rmse_test_pred = rmse_wrap(yte_hat, yte)
    sim_test = simulate_narx_model(u_te, th_te, model, likelihood, na, nb, device, x_mu, x_sig)
    rmse_test_sim = rmse_wrap(sim_test, th_te)
    print(f"Test Prediction RMSE (wrap): {rmse_test_pred:.4f}")
    print(f"Test Simulation RMSE (wrap): {rmse_test_sim:.4f}")

    # ---- Hidden-set evaluation & submissions (unchanged) ----
    pred_hidden_path = 'disc-benchmark-files/hidden-test-prediction-submission-file.npz'
    if os.path.exists(pred_hidden_path):
        hid = np.load(pred_hidden_path)
        upast = hid['upast'].astype(np.float32)
        thpast = hid['thpast'].astype(np.float32)
        u_blk = upast[:, -nb:] if nb > 0 else np.zeros((upast.shape[0], 0), np.float32)
        s_blk = np.sin(thpast[:, -na:]) if na > 0 else np.zeros((upast.shape[0], 0), np.float32)
        c_blk = np.cos(thpast[:, -na:]) if na > 0 else np.zeros((upast.shape[0], 0), np.float32)
        X_hidden = np.concatenate([u_blk, s_blk, c_blk], axis=1).astype(np.float32)
        X_hidden_n, _, _ = standardize_X(X_hidden, x_mu, x_sig)
        y_hidden_pred = predict_svgp(model, likelihood, to_tensor(X_hidden_n, device), device=device).reshape(-1, 1)

        gt_key = get_first_key(hid, ['thnow_true', 'thnow', 'y_true', 'y', 'target'])
        if gt_key is not None:
            gt = hid[gt_key].astype(np.float32).reshape(-1, 1)
            if gt.shape[0] == y_hidden_pred.shape[0]:
                rmse_hidden_pred = rmse_wrap(y_hidden_pred, gt)
                print(f"Hidden Prediction RMSE (wrap, {gt_key}): {rmse_hidden_pred:.4f}")
            else:
                print(f"Hidden Prediction: GT len {gt.shape[0]} != pred len {y_hidden_pred.shape[0]} (skip RMSE)")
        else:
            print("Hidden Prediction: no GT key found; skipping RMSE.")

        np.savez('disc-submission-files/sparse-gp-hidden-test-prediction-submission-file.npz',
                 upast=upast, thpast=thpast, thnow=y_hidden_pred)
        print("Wrote hidden prediction submission npz.")
    else:
        print(f"Hidden prediction file not found at {pred_hidden_path} — skipping.")

    sim_hidden_path = 'disc-benchmark-files/hidden-test-simulation-submission-file.npz'
    if os.path.exists(sim_hidden_path):
        hid = np.load(sim_hidden_path)
        u_key  = get_first_key(hid, ['u','u_sequence','u_valid','u_test'])
        th_key = get_first_key(hid, ['th_true','th','y_true','y'])
        th0_key = get_first_key(hid, ['th_initial','y_initial','th0'])
        if u_key is None:
            print("Hidden Simulation: no input key (u*). Skipping.")
        else:
            u_seq = hid[u_key].astype(np.float32)
            if th0_key is not None:
                th_init = hid[th0_key].astype(np.float32)
            elif th_key is not None:
                th_init = hid[th_key].astype(np.float32)[:50]
            else:
                th_init = np.zeros(50, np.float32)
                print("Hidden Simulation: no th_initial/th; using zeros warmup 50.")

            sim_hidden = simulate_narx_model(u_seq, th_init, model, likelihood, na, nb, device, x_mu, x_sig)

            if th_key is not None:
                th_true = hid[th_key].astype(np.float32)
                if len(sim_hidden) == len(th_true):
                    rmse_hidden_sim = rmse_wrap(sim_hidden, th_true)
                    print(f"Hidden Simulation RMSE (wrap, {th_key}): {rmse_hidden_sim:.4f}")
                else:
                    print(f"Hidden Simulation: GT len {len(th_true)} != sim len {len(sim_hidden)} (skip RMSE)")
            else:
                print("Hidden Simulation: no GT key; skipping RMSE.")

            np.savez('disc-submission-files/sparse-gp-hidden-test-simulation-submission-file.npz',
                     th=sim_hidden)
            print("Wrote hidden simulation submission npz.")
    else:
        print(f"Hidden simulation file not found at {sim_hidden_path} — skipping.")

    # Save Torch .pth (kept)
    save_path = 'disc-submission-files/gp_narx_svgp.pth'
    torch.save({
        "state_dict": model.state_dict(),
        "likelihood_state_dict": likelihood.state_dict(),
        "na": na, "nb": nb, "M": M,
        "x_mu": x_mu, "x_sig": x_sig,
        "inducing_points": model.variational_strategy.inducing_points.detach().cpu(),
        "input_dim": Xtr_n.shape[1],
    }, save_path)
    print(f"Saved model to {save_path}")

if __name__ == "__main__":
    main()