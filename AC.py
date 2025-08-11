import shutil
import numpy as np
import torch.nn as nn
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

from env.UnbalancedDisk import UnbalancedDisk   # ⬅ match DQN’s import (local file)
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnMaxEpisodes, CallbackList


class SAC_UnbalancedDisk(UnbalancedDisk):
    def __init__(self, umax=3.0, dt=0.025, render_mode=None):
        super().__init__(umax=umax, dt=dt, render_mode=render_mode)

        # SAC needs a 1D continuous action
        self.action_space = spaces.Box(low=np.array([-umax], dtype=np.float32),
                                       high=np.array([umax], dtype=np.float32),
                                       shape=(1,), dtype=np.float32)

        # Tighten obs bounds like DQN wrapper
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -40], dtype=np.float32),
            high=np.array([ np.pi,  40], dtype=np.float32),
            shape=(2,),
        )

        # === IRL reward model (same as DQN) ===
        m = np.load("disc-submission-files/reward_model.npz")
        self.w = m["weights"]         # (D,)
        self.b = float(m["bias"])     # scalar
        self.phi_mean = m["phi_mean"] # (D,)
        self.phi_std  = m["phi_std"]  # (D,)

    def _irl_reward(self, theta, omega, u):
        phi = np.array([
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
        ], dtype=np.float64)
        z = (phi - self.phi_mean) / self.phi_std
        return float(self.w @ z + self.b)

    def step(self, action):
        # action is shape (1,), turn into scalar in [-umax, umax]
        u = float(np.clip(action[0], -self.umax, self.umax))

        # Base env integration
        obs, _reward_env, terminated, truncated, info = super().step(u)

        # Replace reward with IRL reward (same as DQN)
        th, omega = float(obs[0]), float(obs[1])
        reward = self._irl_reward(th, omega, u)

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Gym/Gymnasium compatibility shim (same pattern as we used for DQN)
        try:
            obs, info = super().reset(seed=seed, options=options)
        except TypeError:
            obs, info = super().reset(seed=seed)
        return obs, info


# === Optuna Objective ===
def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    tau = trial.suggest_float("tau", 0.001, 0.05)
    gamma = trial.suggest_float("gamma", 0.95, 0.99)
    buffer_size = trial.suggest_categorical("buffer_size", [50_000, 100_000])
    batch_size = trial.suggest_categorical("batch_size", [256, 512])
    net_arch = trial.suggest_categorical("net_arch_style", ["medium", "large", "huge"])

    net_arch_cfg = {
        "medium": [128, 128],
        "large":  [128, 128, 128],
        "huge":   [256, 256, 256],
    }[net_arch]

    policy_kwargs = dict(net_arch=net_arch_cfg, activation_fn=nn.ReLU)

    # Train / Eval envs 
    env = Monitor(TimeLimit(SAC_UnbalancedDisk(),  max_episode_steps=600))
    eval_env = Monitor(TimeLimit(SAC_UnbalancedDisk(), max_episode_steps=600))

    stop_cb = StopTrainingOnMaxEpisodes(max_episodes=150, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"optuna_sac_trials/trial_{trial.number}",
        eval_freq=5000,
        deterministic=True,
        render=False,
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
        verbose=1,
        seed=42,
    )
    model.learn(total_timesteps=100_000, callback=callback)

    return eval_cb.best_mean_reward


# === Run Optuna Study ===
study = optuna.create_study(direction="maximize", sampler=TPESampler(), pruner=MedianPruner())
study.optimize(objective, n_trials=20)

print("Best trial reward:", study.best_trial.value)
print("Best hyperparameters:", study.best_trial.params)

# Copy best model to submission folder
best_trial_number = study.best_trial.number
optuna_model_path = f"optuna_sac_trials/trial_{best_trial_number}/best_model.zip"
final_model_path = "disc-submission-files/sac-model.zip"
shutil.copy(optuna_model_path, final_model_path)

# Clean up trial dirs
shutil.rmtree("optuna_sac_trials", ignore_errors=True)

# === Demo ===
env = SAC_UnbalancedDisk()
model = SAC.load(final_model_path)

obs, _ = env.reset()
for _ in range(5000):
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    theta = (obs[0] + np.pi) % (2 * np.pi) - np.pi
    print(f"theta = {theta: .4f}, omega: {obs[1]: .4f}, action: {action[0]: .4f}, reward: {reward: .4f}")
    if terminated or truncated:
        obs, _ = env.reset()
env.close()