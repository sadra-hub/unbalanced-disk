# dqn_unbalanced_disk.py
import os
import math
import shutil
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

import torch.nn as nn
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnMaxEpisodes, CallbackList
from scipy.integrate import solve_ivp


def wrap_to_pi(x):
    # wrap angle to (-pi, pi]
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
    metadata = {"render_modes": ["human"], "render_fps": 1 / 0.025}

    def __init__(self, umax=3., dt=0.025, render_mode='human'):
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

        # --- Reward parameters (tune here) ---
        self.k_omega = 0.10      # penalize spin
        self.k_u = 0.01          # tiny control penalty (prevents chattering)
        self.upright_deg = 12.0  # bonus window (deg)
        self.upright_bonus = 0.5
        # -------------------------------------

        # Reward function: swing-up + balance near top (theta ≈ ±pi)
        self.reward_fun = self._reward_swing_balance

        self.render_mode = render_mode
        self.viewer = None
        self.u = 0  # for visual
        self.reset()

    # ---------------- Reward ----------------
    def _reward_swing_balance(self):
        # Error to TOP (±pi). Using pi target; squared/abs treat ±pi the same.
        th_err = wrap_to_pi(self.th - np.pi)   # 0 at top, ±pi at bottom

        # Base term: prefer being near top (cos=1 at th_err=0; -1 at bottom)
        base = math.cos(th_err)

        # Penalties
        pen_spin = self.k_omega * (self.omega ** 2)
        pen_u = self.k_u * ((float(self.u) / self.umax) ** 2)

        # Small bonus when you're really upright & calm
        bonus = self.upright_bonus if (
            abs(th_err) < math.radians(self.upright_deg) and abs(self.omega) < 1.0
        ) else 0.0

        return base - pen_spin - pen_u + bonus
    # ---------------------------------------

    def step(self, action):
        # convert action to u (continuous)
        self.u = action

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
        # no terminal/truncation logic here; keep False/False
        return self.get_obs(), reward, False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.th = np.random.normal(loc=0, scale=0.001)     # near bottom
        self.omega = np.random.normal(loc=0, scale=0.001)
        self.u = 0.0
        return self.get_obs(), {}

    def get_obs(self):
        # keep tiny sensor noise
        self.th_noise = self.th + np.random.normal(loc=0, scale=0.001)
        self.omega_noise = self.omega + np.random.normal(loc=0, scale=0.001)
        return np.array([self.th_noise, self.omega_noise], dtype=np.float32)

    def render(self):
        # Minimal pygame rendering (no PNGs), just a disk and a simple control bar.
        import pygame
        from pygame import gfxdraw

        screen_width = 500
        screen_height = 500

        th = self.th
        omega = self.omega

        if self.viewer is None:
            pygame.init()
            pygame.display.init()
            self.viewer = pygame.display.set_mode((screen_width, screen_height))

        surf = pygame.Surface((screen_width, screen_height))
        surf.fill((255, 255, 255))

        # background circles
        gfxdraw.filled_circle(
            surf,
            screen_width // 2,
            screen_height // 2,
            int(screen_width / 2 * 0.65 * 1.3),
            (32, 60, 92),
        )
        gfxdraw.filled_circle(
            surf,
            screen_width // 2,
            screen_height // 2,
            int(screen_width / 2 * 0.06 * 1.3),
            (132, 132, 126),
        )

        # moving mass
        r = int(screen_width * 0.40 * 1.3 * 0.5)
        cx = screen_width // 2
        cy = screen_height // 2
        mx = int(cx - math.sin(th) * r)
        my = int(cy - math.cos(th) * r)
        gfxdraw.filled_circle(surf, mx, my, int(screen_width / 2 * 0.22 * 1.3), (155, 140, 108))
        gfxdraw.filled_circle(surf, mx, my, int(screen_width / 2 * 0.22 / 8 * 1.3), (71, 63, 48))

        # simple torque bar (no image): draw a horizontal bar scaled by |u|
        if self.u:
            u_norm = float(np.clip(self.u, -self.umax, self.umax)) / self.umax
            bar_len = int(abs(u_norm) * screen_width * 0.25)
            color = (220, 50, 50) if u_norm > 0 else (50, 50, 220)
            rect = pygame.Rect(cx - bar_len // 2, cy - 10, bar_len, 20)
            pygame.draw.rect(surf, color, rect)

        # flip vertical to match original
        surf = pygame.transform.flip(surf, False, True)
        self.viewer.blit(surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()

        return True

    def close(self):
        if self.viewer is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.viewer = None


class DQN_UnbalancedDisk(UnbalancedDisk):
    """
    Wrap UnbalancedDisk for DQN by discretizing the action space.
    Uses the reward defined inside UnbalancedDisk (no IRL here).
    """
    def __init__(self, umax=3.0, dt=0.025, n_actions=10, render_mode=None):
        super().__init__(umax=umax, dt=dt, render_mode=render_mode)
        # Discretize the continuous torque into n_actions bins in [-umax, umax]
        self.actions = np.linspace(-umax, umax, n_actions).astype(np.float32)
        self.action_space = spaces.Discrete(n_actions)

        # Keep a bounded observation box for SB3 (same shape as base env)
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -40.0], dtype=np.float32),
            high=np.array([+np.pi, +40.0], dtype=np.float32),
            shape=(2,),
        )

    def step(self, action):
        # Map discrete action index -> continuous torque
        u = float(self.actions[int(action)])
        obs, reward, terminated, truncated, info = super().step(u)
        # reward is computed by UnbalancedDisk._reward_swing_balance()
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        try:
            obs, info = super().reset(seed=seed, options=options)
        except TypeError:
            obs, info = super().reset(seed=seed)
        return obs, info


# === Optuna objective ===
def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.5)
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

    # no rendering during training/eval
    env = Monitor(TimeLimit(DQN_UnbalancedDisk(n_actions=n_actions, render_mode=None), max_episode_steps=600))
    eval_env = Monitor(TimeLimit(DQN_UnbalancedDisk(n_actions=n_actions, render_mode=None), max_episode_steps=600))

    stop_cb = StopTrainingOnMaxEpisodes(max_episodes=150, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"optuna_dqn_trials/trial_{trial.number}",
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
        gamma=gamma,
        batch_size=batch_size,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=88,
    )

    model.learn(total_timesteps=100_000, callback=callback)

    return eval_cb.best_mean_reward


if __name__ == "__main__":
    # === Run Optuna Study ===
    study = optuna.create_study(direction="maximize", sampler=TPESampler(), pruner=MedianPruner())
    study.optimize(objective, n_trials=20)

    print("Best trial value:", study.best_trial.value)
    print("Best hyperparameters:", study.best_trial.params)

    # Copy best model to submission folder
    best_trial_number = study.best_trial.number
    optuna_model_path = f"optuna_dqn_trials/trial_{best_trial_number}/best_model.zip"
    os.makedirs("disc-submission-files", exist_ok=True)
    final_model_path = "disc-submission-files/dqn-model.zip"
    shutil.copy(optuna_model_path, final_model_path)

    # Clean up
    shutil.rmtree("optuna_dqn_trials", ignore_errors=True)

    # === Demo the Best Model ===
    env = DQN_UnbalancedDisk(render_mode=None)
    model = DQN.load(final_model_path)
    obs, _ = env.reset()
    for _ in range(5000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        theta_wrapped = (obs[0] + np.pi) % (2 * np.pi) - np.pi
        print(f"theta={theta_wrapped: .4f} rad, omega={obs[1]: .4f} rad/s, action_idx={int(action)}, reward={reward: .4f}")
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()