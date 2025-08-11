# sac_irl_dual_policies.py
# Self-contained:
# - Defines UnbalancedDisk env (keeps "do not edit" dynamics intact)
# - Trains TWO linear IRL reward models from two expert logs:
#     1) Upright-only    → disc-benchmark-files/expert-log-without-reference.csv
#        (headers: t,theta,omega,u)
#     2) Ref-tracking    → disc-benchmark-files/expert-log-reference-tracking.csv
#        (headers: t,theta,omega,u,theta_ref,track_err)  ← extra columns are ignored
# - Uses each learned reward to train a SAC policy via Optuna (few trials, higher eval_freq)
# - Saves the two best policies to:
#     disc-submission-files/sac-model-upright.zip
#     disc-submission-files/sac-model-ref.zip

import os
import math
import json
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
# Config (tuned for <~2h wall time on modest CPU)
# =========================
SEED = 88
np.random.seed(SEED); torch.manual_seed(SEED)

LOG_UPRIGHT = "disc-benchmark-files/expert-log-without-reference.csv"     # t,theta,omega,u
LOG_REF     = "disc-benchmark-files/expert-log-reference-tracking.csv"    # t,theta,omega,u,theta_ref,track_err

MAX_EP_STEPS   = 600
N_TRIALS        = 4                 # small search per objective
TOTAL_TIMESTEPS = 30_000            # per trial
EVAL_FREQ       = 20_000            # evaluate infrequently to save time
N_EVAL_EPISODES = 3

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
    UnbalancedDisk
    th =
                  +-pi   (top / upright)
                    |
           pi/2   -----  -pi/2
                    |
                    0    (bottom / start)
    '''
    metadata = {"render_modes": ["human"]}

    def __init__(self, umax=3., dt=0.025, render_mode=None):
        ############# start do not edit  ################
        self.omega0 = 11.339846957335382
        self.delta_th = 0
        self.gamma = 1.3328339309394384
        self.Ku = 28.136158407237073
        self.Fc = 6.062729509386865
        self.coulomb_omega = 0.001
        ############# end do not edit ###################

        self.umax = umax
        self.dt = dt  # time step

        # continuous action (scalar)
        self.action_space = spaces.Box(low=-umax, high=umax, shape=tuple())
        low = [-float('inf'), -40]
        high = [float('inf'), 40]
        self.observation_space = spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            shape=(2,)
        )

        # default reward (will be replaced in IRL subclass)
        self.reward_fun = self._reward_swing_balance

        self.render_mode = render_mode
        self.viewer = None
        self.u = 0.0
        self.reset()

    def _reward_swing_balance(self):
        th_err = wrap_to_pi(self.th - np.pi)   # 0 at top
        base = math.cos(th_err)
        k_omega = 0.10
        k_u = 0.01
        pen_spin = k_omega * (self.omega ** 2)
        pen_u = k_u * ((float(self.u) / self.umax) ** 2)
        bonus = 0.5 if (abs(th_err) < math.radians(12.0) and abs(self.omega) < 1.0) else 0.0
        return base - pen_spin - pen_u + bonus

    def step(self, action):
        self.u = action  # scalar or array

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

        reward = self.reward_fun()
        return self.get_obs(), reward, False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.th = np.random.normal(loc=0, scale=0.001)     # near bottom
        self.omega = np.random.normal(loc=0, scale=0.001)
        self.u = 0.0
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
# IRL: feature map & training
# =========================
def phi_features(theta, omega, u):
    # 12D features that don't depend on reference signals (so we can deploy w/o ref)
    return np.array([
        theta,
        theta**2,
        np.sin(theta),
        np.cos(theta),
        omega,
        omega**2,
        u,
        u**2,
        np.abs(theta * omega),
        np.abs(u) * np.abs(theta),
        np.abs(u) * np.abs(omega),
        np.sign(u) * np.abs(theta),
    ], dtype=np.float32)

def load_expert(csv_path):
    """
    Reads expert logs robustly.
    - Upright log headers: t,theta,omega,u
    - Ref-tracking log headers: t,theta,omega,u,theta_ref,track_err
      (we simply ignore the extra columns).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expert log not found: {csv_path}")
    data = np.genfromtxt(csv_path, delimiter=',', names=True)
    # Ensure required columns exist
    for col in ("theta", "omega", "u"):
        if col not in data.dtype.names:
            raise ValueError(f"Column '{col}' not found in {csv_path}. Found: {data.dtype.names}")
    th = np.asarray(data["theta"], dtype=np.float32)
    om = np.asarray(data["omega"], dtype=np.float32)
    u  = np.asarray(data["u"],     dtype=np.float32)
    return th, om, u

def sample_random_negatives(N, umax=3.0, dt=0.025, max_ep_steps=600, seed=SEED+1):
    rng = np.random.default_rng(seed)
    env = UnbalancedDisk(umax=umax, dt=dt, render_mode=None)
    states, acts = [], []
    while len(states) < N:
        obs, _ = env.reset()
        for _ in range(max_ep_steps):
            a = rng.uniform(-umax, umax)
            obs, _, _, _, _ = env.step(a)
            states.append((float(obs[0]), float(obs[1])))
            acts.append(float(a))
            if len(states) >= N:
                break
    env.close()
    th = np.array([s[0] for s in states], dtype=np.float32)
    om = np.array([s[1] for s in states], dtype=np.float32)
    u  = np.array(acts, dtype=np.float32)
    return th, om, u

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

def train_irl_from_expert(csv_path, umax=3.0, dt=0.025, max_ep_steps=600, device="cpu"):
    # Expert positives
    th_e, om_e, u_e = load_expert(csv_path)
    N = th_e.shape[0]
    N = int(min(N, 40_000))  # cap for speed if very long logs
    Phi_e = np.stack([phi_features(th_e[i], om_e[i], u_e[i]) for i in range(N)], axis=0)

    # Random negatives
    th_n, om_n, u_n = sample_random_negatives(N, umax=umax, dt=dt, max_ep_steps=max_ep_steps, seed=SEED+7)
    Phi_n = np.stack([phi_features(th_n[i], om_n[i], u_n[i]) for i in range(N)], axis=0)

    # Normalize using EXPERT stats
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
    return w, b, mu.astype(np.float32), sig.astype(np.float32)

# =========================
# SAC wrapper with IRL reward
# =========================
class SAC_UnbalancedDisk(UnbalancedDisk):
    def __init__(self, w, b, phi_mean, phi_std, umax=3.0, dt=0.025, render_mode=None):
        super().__init__(umax=umax, dt=dt, render_mode=render_mode)

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

        # Use IRL reward instead of base
        self.reward_fun = self._irl_reward

    def _irl_reward(self):
        th = float(self.th)
        om = float(self.omega)
        u  = float(self.u if np.isscalar(self.u) else np.asarray(self.u).reshape(-1)[0])
        z = (phi_features(th, om, u) - self.phi_mean) / (self.phi_std + 1e-8)
        return float(self.w @ z + self.b)

    def step(self, action):
        u = float(np.clip(action[0], -self.umax, self.umax))
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

def sac_objective(trial, w, b, mu, sig, trial_dir_prefix):
    lr = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    tau = trial.suggest_float("tau", 0.005, 0.05)
    gamma = trial.suggest_float("gamma", 0.96, 0.995)
    buffer_size = trial.suggest_categorical("buffer_size", [50_000, 100_000])
    batch_size = trial.suggest_categorical("batch_size", [256, 512])
    net_arch = trial.suggest_categorical("net_arch_style", ["medium", "large"])
    policy_kwargs = make_policy_kwargs(net_arch)

    env = Monitor(TimeLimit(SAC_UnbalancedDisk(w, b, mu, sig),  max_episode_steps=MAX_EP_STEPS))
    eval_env = Monitor(TimeLimit(SAC_UnbalancedDisk(w, b, mu, sig), max_episode_steps=MAX_EP_STEPS))

    # callbacks
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
    print(f"\n[IRL:{tag}] Training reward from '{log_path}' ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    w, b, mu, sig = train_irl_from_expert(log_path, device=device)
    print(f"[IRL:{tag}] Learned reward: dim={w.shape[0]}  ||w||={np.linalg.norm(w):.3f}  b={b:.3f}")
    np.savez(os.path.join(SAVE_DIR, f"reward_model_{tag}.npz"), weights=w, bias=b, phi_mean=mu, phi_std=sig)

    print(f"[Optuna:{tag}] Starting SAC search (trials={N_TRIALS}, steps/trial={TOTAL_TIMESTEPS}) ...")
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED), pruner=MedianPruner())
    study.optimize(lambda t: sac_objective(t, w, b, mu, sig, trial_dir_prefix=tag),
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
        env = SAC_UnbalancedDisk(w, b, mu, sig)
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
    # Train from upright-only expert log
    train_policy_from_log(LOG_UPRIGHT, tag="upright")

    # Train from reference-tracking expert log (extra columns are ignored safely)
    #train_policy_from_log(LOG_REF, tag="ref")

    print("\n[Done] Two policies saved to:")
    print(f"  - {os.path.join(SAVE_DIR, 'sac-model-upright.zip')}")
    #print(f"  - {os.path.join(SAVE_DIR, 'sac-model-ref.zip')}")

if __name__ == "__main__":
    main()