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
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnMaxEpisodes, CallbackList

class SAC_UnbalancedDisk(UnbalancedDisk):
    def __init__(self, umax=3.0, dt=0.025, randomize_friction=True, render_mode='human'):
        super().__init__(umax=umax, dt=dt, render_mode=render_mode)
        self.randomize_friction = randomize_friction

        # SAC uses continuous action space
        self.action_space = spaces.Box(low=-umax, high=umax, shape=(1,), dtype=np.float32)
        self.target = np.pi

        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -40], dtype=np.float32),
            high=np.array([np.pi, 40], dtype=np.float32),
            shape=(2,),
        )

    def step(self, action):
        friction_scale = np.random.uniform(0.6, 1.5) if self.randomize_friction else 1.0
        original_gamma = self.discount_factor
        original_Fc = self.Fc
        self.discount_factor *= friction_scale
        self.Fc *= friction_scale

        u = np.clip(action[0], -self.umax, self.umax)
        obs, reward, terminated, truncated, info = super().step(u)

        self.discount_factor = original_gamma
        self.Fc = original_Fc

        th, omega = obs
        theta = ((th - np.pi) % (2 * np.pi)) - np.pi
        theta_abs = abs(theta)
        omega_abs = abs(omega)

        if theta_abs <= 0.5 * np.pi:
            reward = min(-1, -np.pi - 1 + abs(omega))
        elif theta_abs <= 0.75 * np.pi:
            reward = (-np.cos(theta_abs)) / (0.1 + omega_abs)
        else:
            reward = 5 * (-np.cos(theta_abs))**0.5 / (0.1 + omega_abs)

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return obs, info


# === Optuna Optimization ===
def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    tau = trial.suggest_float("tau", 0.001, 0.05)
    gamma = trial.suggest_float("gamma", 0.95, 0.99)
    buffer_size = trial.suggest_categorical("buffer_size", [50_000, 100_000])
    batch_size = trial.suggest_categorical("batch_size", [256, 512])
    net_arch = trial.suggest_categorical("net_arch_style", ["medium", "large", "huge"])

    net_arch_cfg = {
        "medium": [128, 128],
        "large": [128, 128, 128],
        "huge": [256, 256, 256],
    }[net_arch]

    policy_kwargs = dict(net_arch=net_arch_cfg, activation_fn=nn.ReLU)

    # Train and eval environments
    env = Monitor(TimeLimit(SAC_UnbalancedDisk(randomize_friction=True), max_episode_steps=600))
    eval_env = Monitor(TimeLimit(SAC_UnbalancedDisk(randomize_friction=False), max_episode_steps=600))

    # Callbacks
    stop_cb = StopTrainingOnMaxEpisodes(max_episodes=150, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"optuna_sac_trials/trial_{trial.number}",
        best_model_name="best_model",
        eval_freq=5000,
        deterministic=True,
        render=False,
    )
    callback = CallbackList([stop_cb, eval_cb])

    # Train model
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


# Run Optuna Study
study = optuna.create_study(direction="maximize", sampler=TPESampler(), pruner=MedianPruner())
study.optimize(objective, n_trials=20)

print("Best trial reward:", study.best_trial.value)
print("Best hyperparameters:", study.best_trial.params)

# Copy best model to submission folder
best_trial_number = study.best_trial.number
optuna_model_path = f"optuna_sac_trials/trial_{best_trial_number}/best_model.zip"
final_model_path = "disc-submission-files/sac-model.zip"
shutil.copy(optuna_model_path, final_model_path)

# Delete Optuna model folders
shutil.rmtree("optuna_sac_trials", ignore_errors=True)

# === Demo the Best Model ===
env = SAC_UnbalancedDisk(randomize_friction=False)
model = SAC.load(final_model_path)

obs, _ = env.reset()
for _ in range(5000):
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    theta = (obs[0] + np.pi) % (2 * np.pi) - np.pi
    print(f"theta = {theta: .4f}, omega: {obs[1]: .4f}, action: {action[0]: .4f}")
    if terminated or truncated:
        obs, _ = env.reset()
env.close()