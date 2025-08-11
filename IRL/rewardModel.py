# rewardModel.py
# MaxEnt IRL for UnbalancedDisk with standardized linear features.
# Optional: tiny soft value iteration (SVI) to give the policy one-step lookahead.
#
# Output: disc-submission-files/maxent_reward_linear_std.npz
#         (contains weights + phi_mean/std + metadata)

import os, glob, csv, math, time
import numpy as np

from env.UnbalancedDisk import UnbalancedDisk

# =========================
# Config
# =========================
LOGS_DIR   = "logs"
OUT_DIR    = "disc-submission-files"
MODEL_PATH = os.path.join(OUT_DIR, "maxent_reward_linear_std.npz")

DT           = 0.005
UMAX         = 3.0
ACTION_POINTS= 21                         # discrete action grid for continuous torque
DISCOUNT     = 0.995
TEMPERATURE  = 0.05                       # softer -> more exploration; smaller = sharper
EPOCHS       = 50                         # IRL outer iters (theta updates)
PI_ROLLOUTS  = 100                        # episodes to estimate mu_pi (higher = lower noise)
ROLLOUT_STEPS= 1200                       # steps/episode (ensure swing+hold possible)
LR_THETA     = 0.02                       # gradient ascent step
L2_DECAY     = 1e-3                       # weight decay per iter to avoid runaway θ
SEED         = 42

# Optional planning (recommended): tiny soft value iteration
USE_SOFT_VI  = True
STATE_SAMPLES= 1024                       # sampled states for soft value iteration
VI_ITERS     = 15                         # soft value iteration sweeps
NN_K         = 1                          # nearest-neighbor mapping for V(s') (k=1)

np.random.seed(SEED)

# =========================
# Features
# =========================
def phi_raw(th, om, u):
    """Unstandardized features (same spirit as your logistic IRL)."""
    return np.array([
        th,
        th**2,
        math.sin(th),
        math.cos(th),
        om,
        om**2,
        u,
        u**2,
        abs(th*om),
        abs(u)*abs(th),
        abs(u)*abs(om),
        (1.0 if u>=0 else -1.0)*abs(th),
    ], dtype=np.float64)

FEATURE_NAMES = [
    "theta","theta^2","sin","cos","omega","omega^2","u","u^2","|th*om|","|u||th|","|u||om|","sign(u)|th|"
]
FEATURE_DIM = len(FEATURE_NAMES)

def collect_demo_features(trajs):
    """Stack all demo φ(s,a) to compute mean/std for standardization."""
    feats = []
    for tr in trajs:
        for th, om, u in tr:
            feats.append(phi_raw(th, om, u))
    feats = np.vstack(feats)
    mean = feats.mean(axis=0)
    std  = feats.std(axis=0)
    std  = np.where(std < 1e-8, 1.0, std)
    return mean, std

def phi_z(th, om, u, m, s):
    """Standardized features."""
    return (phi_raw(th, om, u) - m) / s

# =========================
# Load expert trajectories
# =========================
def load_csv(path):
    with open(path, "r") as f:
        r = csv.DictReader(f)
        rows = [(float(x["theta"]), float(x["omega"]), float(x["u"])) for x in r]
    return np.array(rows, dtype=np.float64) if rows else None

def load_npz(path):
    d = np.load(path)
    th, om, u = d["theta"], d["omega"], d["u"]
    T = min(len(th), len(om), len(u))
    return np.stack([th[:T], om[:T], u[:T]], axis=1)

def load_trajectories(logs_dir=LOGS_DIR):
    trajs = []
    for p in glob.glob(os.path.join(logs_dir, "*.csv")):
        try:
            arr = load_csv(p)
            if arr is not None and len(arr) > 10:
                trajs.append(arr)
        except Exception:
            pass
    for p in glob.glob(os.path.join(logs_dir, "*.npz")):
        try:
            arr = load_npz(p)
            if arr is not None and len(arr) > 10:
                trajs.append(arr)
        except Exception:
            pass
    if not trajs:
        raise FileNotFoundError(f"No logs found in {logs_dir}")
    return trajs

# =========================
# Expert / Policy feature expectations
# =========================
def empirical_feature_expectations(trajs, discount, m, s):
    """
    μ_E = discounted average of standardized features over all demos.
    """
    num = np.zeros(FEATURE_DIM, dtype=np.float64)
    den = 0.0
    for tr in trajs:
        g = 1.0
        for (th, om, u) in tr:
            num += g * phi_z(th, om, u, m, s)
            den += g
            g *= discount
    return num / max(den, 1e-12)

# =========================
# Soft policy + rollouts
# =========================
A_GRID = np.linspace(-UMAX, UMAX, ACTION_POINTS)

def softmax_row(q_row, tau=TEMPERATURE):
    z = q_row / max(tau, 1e-8)
    z = z - np.max(z)               # stabilize
    p = np.exp(z); p /= np.sum(p)
    return p

def reward_linear(theta_w, th, om, a, m, s):
    return float(np.dot(theta_w, phi_z(th, om, a, m, s)))

# ---- Myopic soft policy (immediate reward only) ----
def action_dist_myopic(theta_w, th, om, m, s):
    q = np.array([reward_linear(theta_w, th, om, a, m, s) for a in A_GRID])
    return softmax_row(q)

# ---- Tiny Soft Value Iteration (OPTIONAL, recommended) ----
def sample_states(n=STATE_SAMPLES):
    S = np.zeros((n,2), dtype=np.float64)
    # random coverage in typical ranges
    S[:,0] = np.random.uniform(-np.pi, np.pi, size=n)   # theta
    S[:,1] = np.random.uniform(-8.0, 8.0, size=n)       # omega
    return S

def nearest_idx(S, th, om):
    # L1 distance is fine here
    d = np.abs(S[:,0]-th) + np.abs(S[:,1]-om)
    return int(np.argmin(d))

def step_env(env, th, om, a):
    # set internal state & step once deterministically
    env.th = float(th)
    env.omega = float(om)
    obs, r, term, trunc, info = env.step(float(a))
    return float(obs[0]), float(obs[1])

def soft_value_iteration(theta_w, S, env, m, s, tau=TEMPERATURE, gamma=DISCOUNT, iters=VI_ITERS):
    """
    V(s) ≈ τ log Σ_a exp( (R(s,a) + γ V(s')) / τ ), with s' from a one-step env transition.
    Uses nearest-neighbor bootstrapping on the sampled state set S.
    """
    V = np.zeros(len(S), dtype=np.float64)
    for _ in range(iters):
        V_new = np.zeros_like(V)
        for i, (th, om) in enumerate(S):
            q = []
            for a in A_GRID:
                thn, omn = step_env(env, th, om, a)
                j = nearest_idx(S, thn, omn)
                q.append(reward_linear(theta_w, th, om, a, m, s) + gamma * V[j])
            q = np.asarray(q)
            # soft value
            vmax = np.max(q)
            V_new[i] = tau * (np.log(np.sum(np.exp((q - vmax)/tau))) + vmax/tau)
        V = V_new
    return V

def action_dist_with_V(theta_w, th, om, S, V, env, m, s, tau=TEMPERATURE, gamma=DISCOUNT):
    q = []
    for a in A_GRID:
        thn, omn = step_env(env, th, om, a)
        j = nearest_idx(S, thn, omn)
        q.append(reward_linear(theta_w, th, om, a, m, s) + gamma * V[j])
    q = np.asarray(q)
    return softmax_row(q, tau=tau)

def rollout_mu_pi(env, theta_w, m, s,
                  episodes=PI_ROLLOUTS, steps=ROLLOUT_STEPS,
                  use_soft_vi=USE_SOFT_VI, S=None, V=None):
    """
    Estimate μ_π under current θ. If use_soft_vi=True, use π from Q=R+γV(s').
    """
    feats_sum = np.zeros(FEATURE_DIM, dtype=np.float64)
    weight_sum = 0.0

    for _ in range(episodes):
        obs, _ = env.reset()
        th, om = float(obs[0]), float(obs[1])
        g = 1.0
        for t in range(steps):
            if use_soft_vi and (S is not None and V is not None):
                pi_a = action_dist_with_V(theta_w, th, om, S, V, env, m, s)
            else:
                pi_a = action_dist_myopic(theta_w, th, om, m, s)
            a = np.random.choice(A_GRID, p=pi_a)

            # accumulate standardized features *before* stepping (on s,a)
            feats_sum += g * phi_z(th, om, a, m, s)
            weight_sum += g
            g *= DISCOUNT

            # step env
            obs, r, term, trunc, _ = env.step(float(a))
            th, om = float(obs[0]), float(obs[1])
            if term or trunc:
                obs, _ = env.reset()
                th, om = float(obs[0]), float(obs[1])

    return feats_sum / max(weight_sum, 1e-12)

# =========================
# IRL training loop
# =========================
def maxent_irl(trajs):
    print(f"Loaded {len(trajs)} trajectories")
    # print demo μ_E (unstandardized) just once for reference (as you saw before)
    # (not used in learning; purely for visibility)
    ref = np.zeros(FEATURE_DIM, dtype=np.float64)
    wsum = 0.0
    for tr in trajs:
        g=1.0
        for th,om,u in tr:
            ref += g*phi_raw(th,om,u); wsum += g; g*=DISCOUNT
    print("Expert feature expectations mu_E (raw):")
    print(np.round(ref/max(wsum,1e-12), 5))

    # Standardization stats from demos
    phi_mean, phi_std = collect_demo_features(trajs)

    # Standardized expert expectations
    mu_E = empirical_feature_expectations(trajs, DISCOUNT, phi_mean, phi_std)
    print("Expert feature expectations mu_E (standardized):")
    print(np.round(mu_E, 4))

    # Init theta
    theta = np.zeros(FEATURE_DIM, dtype=np.float64)

    # Env for evaluation/policy
    env = UnbalancedDisk(dt=DT, umax=UMAX, render_mode=None)

    # Pre-sample states for soft VI
    if USE_SOFT_VI:
        S = sample_states(STATE_SAMPLES)
    else:
        S = None

    for it in range(1, EPOCHS+1):
        # Soft VI to get V(s) under current theta (optional)
        V = None
        if USE_SOFT_VI:
            V = soft_value_iteration(theta, S, env, phi_mean, phi_std)

        # Estimate mu_pi under current policy
        mu_pi = rollout_mu_pi(env, theta, phi_mean, phi_std,
                              episodes=PI_ROLLOUTS, steps=ROLLOUT_STEPS,
                              use_soft_vi=USE_SOFT_VI, S=S, V=V)

        # Gradient & update (with mild L2 decay)
        grad = mu_E - mu_pi
        theta = (1.0 - L2_DECAY) * theta + LR_THETA * grad

        gap = float(np.linalg.norm(grad))
        print(f"[{it:03d}/{EPOCHS}] ||mu_E - mu_pi|| = {gap:.6f}")
        if it % 5 == 0:
            top_idx = np.argsort(-np.abs(theta))[:5]
            tops = ", ".join([f"{FEATURE_NAMES[i]}:{theta[i]:+.3f}" for i in top_idx])
            print("  mu_pi:", np.round(mu_pi, 3))
            print("  grad :", np.round(grad, 3))
            print("  top θ:", tops)

    env.close()
    return theta, phi_mean, phi_std

# =========================
# Save / load helpers
# =========================
def save_reward(path, w, mean, std, meta=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, weights=w, phi_mean=mean, phi_std=std, meta=meta if meta is not None else {})

# =========================
# Main
# =========================
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    trajs = load_trajectories(LOGS_DIR)

    theta, mean, std = maxent_irl(trajs)

    meta = {
        "discount": DISCOUNT,
        "temperature": TEMPERATURE,
        "action_points": ACTION_POINTS,
        "use_soft_vi": USE_SOFT_VI,
        "state_samples": STATE_SAMPLES if USE_SOFT_VI else 0,
        "vi_iters": VI_ITERS if USE_SOFT_VI else 0,
        "lr_theta": LR_THETA,
        "l2_decay": L2_DECAY,
        "rollouts": PI_ROLLOUTS,
        "steps": ROLLOUT_STEPS,
        "dt": DT,
        "umax": UMAX,
        "seed": SEED,
        "feature_names": FEATURE_NAMES,
    }
    save_reward(MODEL_PATH, theta, mean.astype(np.float64), std.astype(np.float64), meta)
    print("\nSaved MaxEnt (standardized) linear reward ->", MODEL_PATH)

    # Tiny usage example
    th, om, u = 0.05, -0.10, 0.30
    r = float(np.dot(theta, (phi_raw(th, om, u) - mean) / std))
    print(f"Example R(th={th:.3f}, om={om:.3f}, u={u:.3f}) = {r:+.4f}")