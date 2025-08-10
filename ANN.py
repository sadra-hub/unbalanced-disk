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
# Improved NARX Model with Dropout and BatchNorm
# ========================
class ImprovedNARX(nn.Module):
    def __init__(self, input_dim, hidden_neuron, hidden_layers, activation, dropout_rate=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_neuron))
        self.batch_norms.append(nn.BatchNorm1d(hidden_neuron))
        self.dropouts.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_neuron, hidden_neuron))
            self.batch_norms.append(nn.BatchNorm1d(hidden_neuron))
            self.dropouts.append(nn.Dropout(dropout_rate))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_neuron, 2)
        self.act = activation()

    def forward(self, x):
        for i, (layer, bn, dropout) in enumerate(zip(self.layers, self.batch_norms, self.dropouts)):
            x = layer(x)
            if x.size(0) > 1:  # Only apply batch norm if batch size > 1
                x = bn(x)
            x = self.act(x)
            x = dropout(x)
        
        x = self.output_layer(x)
        # Normalize to unit circle
        x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
        return x

# ========================
# Advanced Loss Function
# ========================
def angle_aware_loss(pred_sc, target_sc, alpha=0.5):
    """
    Combines MSE loss on sin/cos with angular error
    """
    # MSE loss on sin/cos
    mse_loss = nn.functional.mse_loss(pred_sc, target_sc)
    
    # Angular error loss
    pred_theta = torch.atan2(pred_sc[:, 0], pred_sc[:, 1])
    target_theta = torch.atan2(target_sc[:, 0], target_sc[:, 1])
    
    # Wrap angle differences
    angle_diff = pred_theta - target_theta
    angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
    angular_loss = torch.mean(angle_diff ** 2)
    
    # Unit circle constraint
    unit_penalty = torch.mean((torch.norm(pred_sc, dim=1) - 1.0) ** 2)
    
    return mse_loss + alpha * angular_loss + 0.01 * unit_penalty

# ========================
# Load data & normalize
# ========================
print("Loading and preprocessing data...")
data = np.load('disc-benchmark-files/training-val-test-data.npz')
th = data['th'].astype(np.float32)
u = data['u'].astype(np.float32)

# Hidden files
hidden_pred_npz = np.load('disc-benchmark-files/hidden-test-prediction-submission-file.npz') if os.path.exists('disc-benchmark-files/hidden-test-prediction-submission-file.npz') else None
hidden_sim_npz  = np.load('disc-benchmark-files/hidden-test-simulation-submission-file.npz')  if os.path.exists('disc-benchmark-files/hidden-test-simulation-submission-file.npz')  else None
if hidden_pred_npz is None: print("⚠ Hidden prediction file not found; will skip hidden prediction RMSE.")
if hidden_sim_npz is None:  print("⚠ Hidden simulation file not found; will skip hidden simulation RMSE.")

# ========================
# Enhanced Training with Early Stopping and Learning Rate Scheduling
# ========================
print("\n=== Training Improved NARX Models ===")

def train_model_with_early_stopping(model, train_loader, val_data, epochs=500, patience=50):
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=20, verbose=False)
    
    X_val_t, y_val_theta, x_mu, x_sig = val_data
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred_sc = model(xb)
            loss = angle_aware_loss(pred_sc, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            train_loss += loss.item()
        
        # Validation every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                pred_sc_val = model(X_val_t.to(device)).cpu().numpy()
            th_val_hat = sincos_to_theta(pred_sc_val)
            val_rmse = rmse_wrap(th_val_hat, y_val_theta)
            
            scheduler.step(val_rmse)
            
            if val_rmse < best_val_loss:
                best_val_loss = val_rmse
                patience_counter = 0
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, best_val_loss

best_score = float("inf")
best_model_narx = None
best_config_narx = None
best_norm_stats = None

# Expanded hyperparameter search with better configurations
configs_to_try = [
    (5, 3, 2, 64, "relu", 0.15),    # More layers, higher dropout
    (5, 3, 2, 96, "relu", 0.1),    # Wider network
    (8, 3, 2, 64, "relu", 0.1),    # More history
    (5, 5, 2, 64, "relu", 0.1),    # More input history
    (8, 5, 3, 64, "relu", 0.15),   # Deep and wide
    (5, 3, 2, 128, "relu", 0.1),   # Very wide
    (10, 3, 2, 64, "relu", 0.1),   # More output history
    (5, 3, 1, 128, "tanh", 0.05),  # Shallow but wide
]

for na, nb, hl, hn, act_name, dropout in configs_to_try:
    print(f"\nTrying: na={na}, nb={nb}, hl={hl}, hn={hn}, act={act_name}, dropout={dropout}")
    
    # Build IO data
    X_train_np, y_train_np = create_IO_data_sincos(u[:int(0.75*len(th))], th[:int(0.75*len(th))], na, nb)
    X_val_np, _ = create_IO_data_sincos(u[int(0.75*len(th)):int(0.9*len(th))], 
                                       th[int(0.75*len(th)):int(0.9*len(th))], na, nb)
    y_val_theta = th[int(0.75*len(th)) + max(na, nb): int(0.9*len(th))]

    # Standardize
    X_train, x_mu, x_sig = standardize_X(X_train_np)
    X_val, _, _ = standardize_X(X_val_np, x_mu, x_sig)

    # Create data loaders
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_np, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)

    # Create model
    activation = nn.ReLU if act_name == "relu" else nn.Tanh
    model_narx = ImprovedNARX(
        input_dim=nb + 2*na, 
        hidden_neuron=hn, 
        hidden_layers=hl, 
        activation=activation, 
        dropout_rate=dropout
    ).to(device)

    # Train with early stopping
    val_data = (X_val_t, y_val_theta, x_mu, x_sig)
    model_narx, val_rmse = train_model_with_early_stopping(model_narx, train_loader, val_data)

    # Evaluate on hidden sets
    rmse_hidden_pred = narx_hidden_prediction_rmse(model_narx, na, nb, hidden_pred_npz, x_mu, x_sig)
    rmse_hidden_sim = narx_hidden_simulation_rmse(model_narx, na, nb, hidden_sim_npz, x_mu, x_sig)

    print(f"Results → ValRMSE: {val_rmse:.5f} | "
          f"HiddenPred: {('NA' if rmse_hidden_pred is None else f'{rmse_hidden_pred:.5f}')} | "
          f"HiddenSim: {('NA' if rmse_hidden_sim is None else f'{rmse_hidden_sim:.5f}')}")

    # Score selection (prioritize hidden simulation)
    if rmse_hidden_sim is not None:
        score = rmse_hidden_sim
    elif rmse_hidden_pred is not None:
        score = rmse_hidden_pred
    else:
        score = val_rmse

    if score < best_score:
        best_score = score
        best_model_narx = model_narx
        best_config_narx = (na, nb, hl, hn, act_name, dropout)
        best_norm_stats = (x_mu.copy(), x_sig.copy())
        print(f"*** NEW BEST MODEL! Score: {score:.5f} ***")

# Save best model
if best_model_narx is not None:
    x_mu, x_sig = best_norm_stats
    torch.save(best_model_narx.state_dict(), "disc-submission-files/ann-narx-model.pth")
    print(f"\nBest NARX: na:{best_config_narx[0]} nb:{best_config_narx[1]} hl:{best_config_narx[2]} "
          f"hn:{best_config_narx[3]} act:{best_config_narx[4]} dropout:{best_config_narx[5]} | score:{best_score:.5f}")

    # Export hidden simulation if available
    if hidden_sim_npz is not None:
        u_key = get_first_key(hidden_sim_npz, ['u','u_sequence','u_valid','u_test'])
        th_key = get_first_key(hidden_sim_npz, ['th','th_true','y','y_true'])
        if (u_key is not None) and (th_key is not None):
            u_test = hidden_sim_npz[u_key].astype(np.float32)
            th_test = hidden_sim_npz[th_key].astype(np.float32)
            th_sim = use_NARX_model_in_simulation_sincos(list(u_test), list(th_test), 
                                                       best_model_narx, best_config_narx[0], best_config_narx[1], x_mu, x_sig)
            np.savez('disc-submission-files/ann-narx-hidden-test-simulation-submission-file.npz',
                     th=th_sim, u=u_test)
            print("Saved improved NARX hidden simulation submission.")