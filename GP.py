import os, math, numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
import gpytorch

# ========== Utils ==========
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
    Closed-loop (free-run) simulation like your original:
    - y_initial: list/array of initial outputs; we take first 50 for warm-up
    - u_sequence: full u for valid/test split (list/array)
    """
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
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

# ========== GP Model (Sparse Variational) ==========
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
        )  # RBF with learned lengthscale

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_svgp(model, likelihood, train_loader, epochs=50, lr=0.01, device="cpu"):
    model.train(); likelihood.train()
    optimizer = torch.optim.Adam([
        {"params": model.parameters()},
        {"params": likelihood.parameters()},
    ], lr=lr)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(train_loader.dataset))
    for ep in range(epochs):
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = -mll(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        # Optional: print(f"Epoch {ep+1}/{epochs} - Loss {total_loss/len(train_loader.dataset):.4f}")
    return model, likelihood

@torch.no_grad()
def predict_svgp(model, likelihood, X, device="cpu", batch_size=4096):
    model.eval(); likelihood.eval()
    means = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i+batch_size]
        pred = likelihood(model(xb))
        means.append(pred.mean.detach().cpu())
    mean = torch.cat(means, dim=0).numpy()
    return mean

# ========== Main ==========
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("models", exist_ok=True)
    npz_data = np.load('disc-benchmark-files/training-val-test-data.npz')
    th_all = npz_data['th'].astype(np.float32)
    u_all  = npz_data['u'].astype(np.float32)

    N = len(th_all)
    train_split = int(0.8 * N)
    valid_split = int(0.9 * N)

    th_train, th_valid, th_test = th_all[:train_split], th_all[train_split:valid_split], th_all[valid_split:]
    u_train,  u_valid,  u_test  = u_all[:train_split],  u_all[train_split:valid_split],  u_all[valid_split:]

    # Hyperparam grids (keep modest—tune if you like)
    na_nb_candidates = [2, 5, 8]
    inducing_point_counts = [300, 100, 50]

    best_pred = {"rmse": float("inf")}
    best_sim  = {"rmse": float("inf")}

    for num_inducing in inducing_point_counts:
        for na in na_nb_candidates:
            for nb in na_nb_candidates:
                # Build datasets
                Xtr, ytr = construct_io_dataset(u_train, th_train, na, nb)
                Xva, yva = construct_io_dataset(u_valid, th_valid, na, nb)

                # Skip tiny/degenerate cases
                if len(Xtr) < 100 or len(Xva) < 100:
                    continue

                # Tensors + loader
                Xtr_t = to_tensor(Xtr, device)
                ytr_t = to_tensor(ytr, device)
                train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=2048, shuffle=True)

                # Init inducing
                # Sample inducing from training inputs for stability
                perm = torch.randperm(Xtr_t.size(0), device=device)[:num_inducing]
                Z = Xtr_t[perm].clone().detach()

                model = NARXSVGP(Z).to(device)
                likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

                # Train
                train_svgp(model, likelihood, train_loader, epochs=60, lr=0.01, device=device)

                # Validation prediction RMSE
                Xva_t = to_tensor(Xva, device)
                yva_hat = predict_svgp(model, likelihood, Xva_t, device=device)
                pred_rmse = rmse(yva_hat, yva)

                # Simulation RMSE (free-run on valid split)
                sim_valid = simulate_narx_model(list(u_valid), list(th_valid), model, likelihood, na, nb, device)
                sim_rmse = rmse(sim_valid, th_valid)

                if pred_rmse < best_pred["rmse"]:
                    best_pred = {"rmse": pred_rmse, "na": na, "nb": nb, "M": num_inducing}
                if sim_rmse < best_sim["rmse"]:
                    best_sim  = {"rmse": sim_rmse,  "na": na, "nb": nb, "M": num_inducing}

                print(f"[Inducing={num_inducing:4d}] na={na:2d}, nb={nb:2d} | "
                      f"Pred RMSE={pred_rmse:.4f} | Sim RMSE={sim_rmse:.4f}")

    print("\n=== Best by Prediction ===", best_pred)
    print("=== Best by Simulation ===", best_sim)

    # ----- Final training with the sim-optimal (or pick pred-optimal) -----
    na = best_sim.get("na", 8)
    nb = best_sim.get("nb", 2)
    inducing = best_sim.get("M", 100)

    Xtr, ytr = construct_io_dataset(u_train, th_train, na, nb)
    Xva, yva = construct_io_dataset(u_valid, th_valid, na, nb)
    Xte, yte = construct_io_dataset(u_test,  th_test,  na, nb)

    Xtr_t = to_tensor(Xtr, device)
    ytr_t = to_tensor(ytr, device)
    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=4096, shuffle=True)

    # Inducing from train
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
    print(f"Validation Prediction RMSE: {rmse_valid_pred:.4f}")
    print(f"Validation Simulation RMSE: {rmse_valid_sim:.4f}")

    # Test metrics
    yte_hat = predict_svgp(model, likelihood, to_tensor(Xte, device), device=device)
    rmse_test_pred = rmse(yte_hat, yte)
    sim_test = simulate_narx_model(list(u_test), list(th_test), model, likelihood, na, nb, device)
    rmse_test_sim = rmse(sim_test, th_test)
    print(f"Test Prediction RMSE: {rmse_test_pred:.4f}")
    print(f"Test Simulation RMSE: {rmse_test_sim:.4f}")

    # ----- Hidden test prediction export -----
    hidden = np.load('disc-benchmark-files/hidden-test-prediction-submission-file.npz')
    upast_hidden = hidden['upast'].astype(np.float32)
    thpast_hidden = hidden['thpast'].astype(np.float32)

    X_hidden = np.concatenate([upast_hidden[:, 15 - nb:], thpast_hidden[:, 15 - na:]], axis=1).astype(np.float32)
    y_hidden_pred = predict_svgp(model, likelihood, to_tensor(X_hidden, device), device=device).reshape(-1, 1)
    assert y_hidden_pred.shape[0] == upast_hidden.shape[0], "Sample size mismatch in hidden test!"

    os.makedirs('disc-submission-files', exist_ok=True)
    np.savez('disc-submission-files/sparse-gp-hidden-test-prediction-submission-file.npz',
             upast=upast_hidden, thpast=thpast_hidden, thnow=y_hidden_pred)

    # ----- Save PyTorch model (.pth) -----
    save_path = 'disc-submission-files/gp-narx-model.pth'
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