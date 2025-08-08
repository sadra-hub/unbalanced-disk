import shutil
import numpy as np
import torch.nn as nn
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

from gym_unbalanced_disk import UnbalancedDisk
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnMaxEpisodes, CallbackList


class DQN_UnbalancedDisk(UnbalancedDisk):
    def __init__(self, umax=3.0, dt=0.025, n_actions=10, render_mode='human'):
        super().__init__(umax=umax, dt=dt, render_mode=render_mode)
        self.actions = np.linspace(-umax, umax, n_actions)
        self.action_space = spaces.Discrete(n_actions)
        self.target = np.pi
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -40], dtype=np.float32),
            high=np.array([np.pi, 40], dtype=np.float32),
            shape=(2,),
        )

        # Load IRL reward model
        m = np.load("disc-submission-files/reward_model.npz")
        self.w = m["weights"]           # (D,)
        self.b = float(m["bias"])       # scalar
        self.phi_mean = m["phi_mean"]   # (D,)
        self.phi_std = m["phi_std"]     # (D,)

    def _reward(self, theta, omega, u):
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
        u = float(self.actions[action])
        obs, terminated, truncated, info = super().step(u)

        th, omega = float(obs[0]), float(obs[1])
        reward = self._reward(th, omega, u)

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return obs, info


# === Optuna Optimization ===
def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.5, log=True)
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 1e-3, 0.1, log=True)
    gamma = trial.suggest_float("discount_factor", 0.95, 0.99)
    target_update = trial.suggest_categorical("target_update_interval", [500, 1000, 5000, 10000])
    batch_size = trial.suggest_categorical("batch_size", [256, 512])
    n_actions = trial.suggest_int("n_actions", 8, 24)
    net_arch = trial.suggest_categorical("net_arch_style", ["medium", "large", "huge"])

    net_arch_cfg = {
        "medium": [128, 128],
        "large": [128, 128, 128],
        "huge": [256, 256, 256],
    }[net_arch]

    policy_kwargs = dict(net_arch=net_arch_cfg, activation_fn=nn.ReLU)

    env = Monitor(TimeLimit(DQN_UnbalancedDisk(n_actions=n_actions), max_episode_steps=600))
    eval_env = Monitor(TimeLimit(DQN_UnbalancedDisk(n_actions=n_actions), max_episode_steps=600))

    stop_cb = StopTrainingOnMaxEpisodes(max_episodes=150, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"optuna_dqn_trials/trial_{trial.number}",
        best_model_name="best_model",
        eval_freq=5000,
        deterministic=True,
        render=False,
    )
    callback = CallbackList([stop_cb, eval_cb])

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=lr,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        target_update_interval=target_update,
        discount_factor=gamma,
        batch_size=batch_size,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=88,
    )

    model.learn(total_timesteps=100_000, callback=callback)

    return eval_cb.best_mean_reward


# Run Optuna Study
study = optuna.create_study(direction="maximize", sampler=TPESampler(), pruner=MedianPruner())
study.optimize(objective, n_trials=20)

print("Best trial value:", study.best_trial.value)
print("Best hyperparameters:", study.best_trial.params)

# Copy best model to submission folder
best_trial_number = study.best_trial.number
optuna_model_path = f"optuna_dqn_trials/trial_{best_trial_number}/best_model.zip"
final_model_path = "disc-submission-files/dqn-model.zip"
shutil.copy(optuna_model_path, final_model_path)

# Delete Optuna model folders
shutil.rmtree("optuna_dqn_trials", ignore_errors=True)

# === Demo the Best Model ===
env = DQN_UnbalancedDisk()
model = DQN.load(final_model_path)
obs, _ = env.reset()
for _ in range(5000):
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    # env.render()
    theta = (obs[0] + np.pi) % (2 * np.pi) - np.pi
    print(f"theta = {theta: .4f}, omega: {obs[1]: .4f}, action: {action: .4f}, reward: {reward: .4f}")
    if terminated or truncated:
        obs, _ = env.reset()
env.close()