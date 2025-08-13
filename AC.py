import os
import math
import json
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import deque

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

from scipy.integrate import solve_ivp

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnMaxEpisodes, CallbackList

# =========================
# Config
# =========================
SEED = 88
np.random.seed(SEED); torch.manual_seed(SEED)

LOG_UPRIGHT = "disc-benchmark-files/expert-log-without-reference.csv"     # t,theta,omega,u
LOG_REF     = "disc-benchmark-files/expert-log-reference-tracking.csv"    # t,theta,omega,u,theta_ref,track_err

# History length for sequence features (number of past actions including current)
HIST_LEN = 10

MAX_EP_STEPS   = 600
N_TRIALS        = 10
TOTAL_TIMESTEPS = 100_000
EVAL_FREQ       = 5_000
N_EVAL_EPISODES = 5

SAVE_DIR = "disc-submission-files"
TRIAL_DIR = "optuna_sac_trials"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(TRIAL_DIR, exist_ok=True)

# =========================
# Base environment
# =========================
def wrap_to_pi(x):
    return (x + np.pi) % (2*np.pi) - np.pi

class UnbalancedDisk(gym.Env):
    '''
    UnbalancedDisk. Dynamics block marked "do not edit" left intact.
    '''
    metadata = {"render_modes": ["human"]}

    def __init__(self, umax=3., dt=0.005, render_mode=None, hist_len=HIST_LEN):
        ############# start do not edit  ################
        self.omega0 = 11.339846957335382
        self.delta_th = 0
        self.gamma = 1.3328339309394384
        self.Ku = 28.136158407237073
        self.Fc = 6.062729509386865
        self.coulomb_omega = 0.001
        ############# end do not edit ###################

        self.umax = umax
        self.dt = dt
        self.hist_len = int(hist_len)

        # continuous action (scalar for native env; SAC wrapper passes (1,) and we convert)
        self.action_space = spaces.Box(low=-umax, high=umax, shape=tuple())
        low = [-float('inf'), -40]
        high = [float('inf'), 40]
        self.observation_space = spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            shape=(2,)
        )

        # default (will be replaced by IRL in SAC wrapper)
        self.reward_fun = self._reward_swing_balance

        self.render_mode = render_mode
        self.viewer = None
        self.u = 0.0
        self.u_hist = deque([0.0]*self.hist_len, maxlen=self.hist_len)  # <-- action history buffer
        self.reset()

    def _reward_swing_balance(self):
        th_err = wrap_to_pi(self.th - np.pi)
        base = math.cos(th_err)
        k_omega = 0.10
        k_u = 0.01
        pen_spin = k_omega * (self.omega ** 2)
        pen_u = k_u * ((float(self.u) / self.umax) ** 2)
        bonus = 0.5 if (abs(th_err) < math.radians(12.0) and abs(self.omega) < 1.0) else 0.0
        return base - pen_spin - pen_u + bonus

    def step(self, action):
        # Accept scalar or (1,)
        if isinstance(action, (list, tuple, np.ndarray)):
            self.u = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        else:
            self.u = float(action)

        ##### Start Do not edit ######
        self.u = np.clip(self.u, -self.umax, self.umax)

        def f(t, y):
            th, omega = y
            dthdt = omega
            friction = self.gamma * omega + self.Fc * np.tanh(omega / self.coulomb_omega)
            domegadt = -self.omega0**2 * np.sin(th + self.delta_th) - friction + self.Ku * self.u
            return np.array([dthdt, domegadt])

        sol = solve_ivp(f, [0, self.dt], [self.th, self.omega])  # integration
        self.th, self.omega = sol.y[:, -1]
        ##### End do not edit   #####

        # Update history AFTER applying this action (so reward sees current u included)
        self.u_hist.append(float(self.u))

        reward = self.reward_fun()
        return self.get_obs(), reward, False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.th = np.random.normal(loc=0, scale=0.001)
        self.omega = np.random.normal(loc=0, scale=0.001)
        self.u = 0.0
        self.u_hist = deque([0.0]*self.hist_len, maxlen=self.hist_len)
        return self.get_obs(), {}

    def get_obs(self):
        self.th_noise = self.th + np.random.normal(loc=0, scale=0.001)
        self.omega_noise = self.omega + np.random.normal(loc=0, scale=0.001)
        return np.array([self.th_noise, self.omega_noise], dtype=np.float32)

    def render(self):
        return True

    def close(self):
        self.viewer = None

# =========================
# IRL: sequence feature map & training
# =========================
def phi_features_seq(theta, omega, u_hist_vec):
    """
    Sequence-aware features: state terms + raw past actions + summary stats.
    u_hist_vec shape: (H,)
    """
    u_hist = np.asarray(u_hist_vec, dtype=np.float32).reshape(-1)
    H = u_hist.shape[0]
    # summary stats on the action history
    du = np.diff(u_hist, prepend=u_hist[0])
    sign_changes = np.sum(np.sign(u_hist[1:]) * np.sign(u_hist[:-1]) < 0)
    feats = [
        theta,
        theta**2,
        np.sin(theta),
        np.cos(theta),
        omega,
        omega**2,
    ]
    # raw actions over the window
    feats.extend(list(u_hist))
    # summaries (scale-invariant-ish)
    feats.extend([
        float(u_hist[-1]),                         # last u
        float(np.mean(u_hist)),
        float(np.std(u_hist) + 1e-8),
        float(np.mean(u_hist**2)),                 # energy
        float(np.mean(np.abs(du))),                # smoothness / TV
        float(sign_changes) / max(H-1, 1),         # normalized sign-change rate
    ])
    return np.array(feats, dtype=np.float32)

def load_expert(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expert log not found: {csv_path}")
    data = np.genfromtxt(csv_path, delimiter=',', names=True)
    for col in ("theta", "omega", "u"):
        if col not in data.dtype.names:
            raise ValueError(f"Column '{col}' not found in {csv_path}. Found: {data.dtype.names}")
    th = np.asarray(data["theta"], dtype=np.float32)
    om = np.asarray(data["omega"], dtype=np.float32)
    u  = np.asarray(data["u"],     dtype=np.float32)
    return th, om, u

def make_u_hist_matrix(u, hist_len):
    """
    Build a (N, H) matrix of action histories aligned at each time index.
    History is left-padded with the first value (or zeros) to keep length H.
    """
    u = np.asarray(u, dtype=np.float32).reshape(-1)
    N = u.shape[0]
    H = int(hist_len)
    U = np.zeros((N, H), dtype=np.float32)
    for k in range(N):
        start = max(0, k - H + 1)
        hist = u[start:k+1]
        if hist.shape[0] < H:
            pad = np.full((H - hist.shape[0],), hist[0] if hist.shape[0] > 0 else 0.0, dtype=np.float32)
            hist = np.concatenate([pad, hist], axis=0)
        U[k, :] = hist
    return U  # shape (N, H)

def sample_random_negatives(N, umax=3.0, dt=0.005, max_ep_steps=600, seed=SEED+1, hist_len=HIST_LEN):
    rng = np.random.default_rng(seed)
    env = UnbalancedDisk(umax=umax, dt=dt, render_mode=None, hist_len=hist_len)
    ths, oms, us = [], [], []
    while len(ths) < N:
        obs, _ = env.reset()
        for _ in range(max_ep_steps):
            a = rng.uniform(-umax, umax)
            obs, _, _, _, _ = env.step(a)
            ths.append(float(obs[0]))
            oms.append(float(obs[1]))
            us.append(float(a))
            if len(ths) >= N:
                break
    env.close()
    return np.array(ths, dtype=np.float32), np.array(oms, dtype=np.float32), np.array(us, dtype=np.float32)

def standardize(Z, mean=None, std=None, eps=1e-8):
    if mean is None:
        mean = Z.mean(axis=0)
        std = Z.std(axis=0) + eps
    Zs = (Z - mean) / std
    return Zs, mean, std

class LinearLogit(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.linear = nn.Linear(D, 1, bias=True)
    def forward(self, x):
        return self.linear(x).squeeze(1)

def build_feature_matrix_seq(th, om, u, hist_len):
    """
    Assemble sequence-aware features for every time index.
    """
    N = th.shape[0]
    Uhist = make_u_hist_matrix(u, hist_len)  # (N, H)
    feats = np.zeros((N, 6 + hist_len + 6), dtype=np.float32)  # 6 state + H raw + 6 summary
    for k in range(N):
        feats[k, :] = phi_features_seq(th[k], om[k], Uhist[k])
    return feats

def train_irl_from_expert(csv_path, umax=3.0, dt=0.005, max_ep_steps=600, device="cpu", hist_len=HIST_LEN):
    # Expert positives
    th_e, om_e, u_e = load_expert(csv_path)
    N = int(min(th_e.shape[0], 40_000))
    th_e, om_e, u_e = th_e[:N], om_e[:N], u_e[:N]
    Phi_e = build_feature_matrix_seq(th_e, om_e, u_e, hist_len)

    # Random negatives
    th_n, om_n, u_n = sample_random_negatives(N, umax=umax, dt=dt, max_ep_steps=max_ep_steps,
                                              seed=SEED+7, hist_len=hist_len)
    Phi_n = build_feature_matrix_seq(th_n, om_n, u_n, hist_len)

    # Normalize with EXPERT stats
    Z_e, mu, sig = standardize(Phi_e)
    Z_n, _, _ = standardize(Phi_n, mu, sig)

    X = np.concatenate([Z_e, Z_n], axis=0).astype(np.float32)
    y = np.concatenate([np.ones(N, dtype=np.float32), np.zeros(N, dtype=np.float32)], axis=0)

    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    loader = DataLoader(ds, batch_size=512, shuffle=True)

    model = LinearLogit(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(200):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

    with torch.no_grad():
        w = model.linear.weight.detach().cpu().numpy().reshape(-1).astype(np.float32)
        b = float(model.linear.bias.detach().cpu().item())
    return w, b, mu.astype(np.float32), sig.astype(np.float32), int(hist_len)

# =========================
# SAC wrapper with IRL (sequence reward)
# =========================
class SAC_UnbalancedDisk(UnbalancedDisk):
    def __init__(self, w, b, phi_mean, phi_std, hist_len=HIST_LEN, umax=3.0, dt=0.005, render_mode=None):
        super().__init__(umax=umax, dt=dt, render_mode=render_mode, hist_len=hist_len)

        # SAC expects a 1D Box action
        self.action_space = spaces.Box(low=np.array([-umax], dtype=np.float32),
                                       high=np.array([umax], dtype=np.float32),
                                       shape=(1,), dtype=np.float32)

        # Bound obs box for SB3
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -40], dtype=np.float32),
            high=np.array([ np.pi,  40], dtype=np.float32),
            shape=(2,),
        )

        self.w = np.asarray(w, dtype=np.float32)
        self.b = float(b)
        self.phi_mean = np.asarray(phi_mean, dtype=np.float32)
        self.phi_std  = np.asarray(phi_std,  dtype=np.float32)

        # Use IRL reward
        self.reward_fun = self._irl_reward

    def _irl_reward(self):
        th = float(self.th)
        om = float(self.omega)
        u_hist_vec = np.array(self.u_hist, dtype=np.float32)
        z = (phi_features_seq(th, om, u_hist_vec) - self.phi_mean) / (self.phi_std + 1e-8)
        return float(self.w @ z + self.b)

    def step(self, action):
        # Clip to env limits, append to history before reward (super() adds again, but we reset history in super.reset)
        u = float(np.clip(np.asarray(action).reshape(-1)[0], -self.umax, self.umax))
        # We let base class append in its step (after integration); no need to append here.
        return super().step(u)

    def reset(self, seed=None, options=None):
        try:
            obs, info = super().reset(seed=seed, options=options)
        except TypeError:
            obs, info = super().reset(seed=seed)
        return obs, info

# =========================
# Optuna objective for SAC
# =========================
def make_policy_kwargs(style):
    return {
        "medium": dict(net_arch=[128, 128], activation_fn=nn.ReLU),
        "large":  dict(net_arch=[256, 256], activation_fn=nn.ReLU),
    }[style]

def sac_objective(trial, w, b, mu, sig, hist_len, trial_dir_prefix):
    lr = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    tau = trial.suggest_float("tau", 0.005, 0.05)
    gamma = trial.suggest_float("gamma", 0.96, 0.995)
    buffer_size = trial.suggest_categorical("buffer_size", [50_000, 100_000])
    batch_size = trial.suggest_categorical("batch_size", [256, 512])
    net_arch = trial.suggest_categorical("net_arch_style", ["medium", "large"])
    policy_kwargs = make_policy_kwargs(net_arch)

    env = Monitor(TimeLimit(SAC_UnbalancedDisk(w, b, mu, sig, hist_len=hist_len),  max_episode_steps=MAX_EP_STEPS))
    eval_env = Monitor(TimeLimit(SAC_UnbalancedDisk(w, b, mu, sig, hist_len=hist_len), max_episode_steps=MAX_EP_STEPS))

    stop_cb = StopTrainingOnMaxEpisodes(max_episodes=150, verbose=0)
    save_path = os.path.join(TRIAL_DIR, trial_dir_prefix, f"trial_{trial.number}")
    os.makedirs(save_path, exist_ok=True)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=0,
    )
    callback = CallbackList([stop_cb, eval_cb])

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=lr,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=SEED,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    best = eval_cb.best_mean_reward
    return float(best if best is not None else -np.inf)

# =========================
# Train one policy (helper)
# =========================
def train_policy_from_log(log_path, tag):
    print(f"\n[IRL:{tag}] Training sequence-aware reward from '{log_path}' (H={HIST_LEN}) ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    w, b, mu, sig, hist_len = train_irl_from_expert(log_path, device=device, hist_len=HIST_LEN)
    print(f"[IRL:{tag}] Learned reward: dim={w.shape[0]}  ||w||={np.linalg.norm(w):.3f}  b={b:.3f}")
    np.savez(os.path.join(SAVE_DIR, f"reward_model_{tag}.npz"),
             weights=w, bias=b, phi_mean=mu, phi_std=sig, hist_len=hist_len)

    print(f"[Optuna:{tag}] Starting SAC search (trials={N_TRIALS}, steps/trial={TOTAL_TIMESTEPS}) ...")
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED), pruner=MedianPruner())
    study.optimize(lambda t: sac_objective(t, w, b, mu, sig, hist_len, trial_dir_prefix=tag),
                   n_trials=N_TRIALS, gc_after_trial=True)

    print(f"[Optuna:{tag}] Best reward: {study.best_trial.value:.3f}")
    print(f"[Optuna:{tag}] Best params: {json.dumps(study.best_trial.params, indent=2)}")

    # copy best model
    best_trial_number = study.best_trial.number
    optuna_model_path = os.path.join(TRIAL_DIR, tag, f"trial_{best_trial_number}", "best_model.zip")
    final_model_path = os.path.join(SAVE_DIR, f"sac-model-{tag}.zip")
    if os.path.exists(optuna_model_path):
        shutil.copy(optuna_model_path, final_model_path)
        print(f"[SAC:{tag}] Saved best policy to {final_model_path}")
    else:
        print(f"[WARN:{tag}] best_model.zip not found for trial {best_trial_number}")

    # quick demo rollout
    try:
        env = SAC_UnbalancedDisk(w, b, mu, sig, hist_len=hist_len)
        env = TimeLimit(env, max_episode_steps=MAX_EP_STEPS)
        model = SAC.load(final_model_path)
        obs, _ = env.reset(seed=SEED)
        ret = 0.0
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ret += reward
            if terminated or truncated:
                break
        print(f"[Demo:{tag}] return over 1000 steps (or early stop): {ret:.3f}")
        env.close()
    except Exception as e:
        print(f"[Demo:{tag}] Skipped (no model?): {e}")

# =========================
# Main
# =========================
def main():
    train_policy_from_log(LOG_UPRIGHT, tag="upright")
    train_policy_from_log(LOG_REF,     tag="ref")
    print("\n[Done] Two policies saved to:")
    print(f"  - {os.path.join(SAVE_DIR, 'sac-model-upright.zip')}")
    print(f"  - {os.path.join(SAVE_DIR, 'sac-model-ref.zip')}")

if __name__ == "__main__":
    main()