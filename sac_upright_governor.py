# sac_upright_governor.py
import os
import math
import time
import argparse
import shutil
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from scipy.integrate import solve_ivp

import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

# -------------------------
# Config (your best hyperparams)
# -------------------------
SEED = 88
BEST_HP = dict(
    learning_rate=0.0027898175145701226,
    tau=0.01980706820315988,
    gamma=0.9808435175032927,
    buffer_size=50_000,
    batch_size=512,
    net_arch=[128, 128],  # "medium"
)
TOTAL_TIMESTEPS = 100_000
EVAL_FREQ = 5_000
N_EVAL_EPISODES = 5
MAX_EP_STEPS = 600

SAVE_DIR = "disc-submission-files"
MODEL_PATH = os.path.join(SAVE_DIR, "sac-model-upright-governed.zip")
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------
# Helpers
# -------------------------
def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

# -------------------------
# Base environment (dynamics "do not edit" preserved)
# -------------------------
class UnbalancedDisk(gym.Env):
    """
    Unbalanced disk with native swing-up reward. We'll subclass this to add the upright governor.
    """
    metadata = {"render_modes": ["human"], "render_fps": 1 / 0.025}

    def __init__(self, umax=3.0, dt=0.025, render_mode=None):
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

        # continuous action (scalar)
        self.action_space = spaces.Box(low=-umax, high=umax, shape=tuple())
        low = [-float('inf'), -40.0]
        high = [float('inf'), 40.0]
        self.observation_space = spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            shape=(2,)
        )

        # swing-up + balance reward (base)
        self.k_omega = 0.10
        self.k_u = 0.01
        self.upright_deg = 12.0
        self.upright_bonus = 0.5
        self.reward_fun = self._reward_swing_balance

        self.render_mode = render_mode
        self.viewer = None
        self.u = 0.0
        self.reset()

    def _reward_swing_balance(self):
        th_err = wrap_to_pi(self.th - np.pi)   # 0 at top
        base = math.cos(th_err)
        pen_spin = self.k_omega * (self.omega ** 2)
        pen_u = self.k_u * ((float(self.u) / self.umax) ** 2)
        bonus = self.upright_bonus if (abs(th_err) < math.radians(self.upright_deg) and abs(self.omega) < 1.0) else 0.0
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

        r = self.reward_fun()
        return self.get_obs(), r, False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.th = np.random.normal(loc=0, scale=0.001)   # near bottom
        self.omega = np.random.normal(loc=0, scale=0.001)
        self.u = 0.0
        return self.get_obs(), {}

    def get_obs(self):
        th_noise = self.th + np.random.normal(loc=0, scale=0.001)
        om_noise = self.omega + np.random.normal(loc=0, scale=0.001)
        return np.array([th_noise, om_noise], dtype=np.float32)

    def render(self):
        # Minimal pygame rendering (no image files)
        import pygame
        from pygame import gfxdraw
        W, H = 500, 500
        th = float(self.th)

        if self.viewer is None:
            pygame.init()
            pygame.display.init()
            self.viewer = pygame.display.set_mode((W, H))

        surf = pygame.Surface((W, H))
        surf.fill((255, 255, 255))

        # background rings
        gfxdraw.filled_circle(surf, W // 2, H // 2, int(W * 0.65 * 0.5), (32, 60, 92))
        gfxdraw.filled_circle(surf, W // 2, H // 2, int(W * 0.06 * 0.5), (132, 132, 126))

        # moving mass
        r = int(W * 0.40 * 0.5)
        cx, cy = W // 2, H // 2
        mx = int(cx - math.sin(th) * r)
        my = int(cy - math.cos(th) * r)
        gfxdraw.filled_circle(surf, mx, my, int(W * 0.22 * 0.5), (155, 140, 108))
        gfxdraw.filled_circle(surf, mx, my, int(W * 0.22 * 0.5 / 8), (71, 63, 48))

        # torque bar
        if self.u:
            u_norm = float(np.clip(self.u, -self.umax, self.umax)) / self.umax
            bar_len = int(abs(u_norm) * W * 0.25)
            color = (220, 50, 50) if u_norm > 0 else (50, 50, 220)
            pygame.draw.rect(surf, color, pygame.Rect(cx - bar_len // 2, cy - 8, bar_len, 16))

        # flip vertical to match usual display
        surf = pygame.transform.flip(surf, False, True)
        self.viewer.blit(surf, (0, 0))
        pygame.event.pump()
        pygame.display.flip()
        return True

    def close(self):
        if self.viewer is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.viewer = None

# -------------------------
# Upright-governed env (prevents fast spin at upright)
# -------------------------
class UnbalancedDiskUprightGov(UnbalancedDisk):
    """
    Adds an "upright governor" that:
      * When near upright (|theta - pi| < hold_deg), strongly limits torque that would
        accelerate in the current spin direction, but allows braking torque.
      * Adds a hold bonus + anti-spin penalty in that region to encourage balancing.
    Dynamics block remains untouched (we only change the action fed to it and the reward).
    """
    def __init__(
        self,
        umax=3.0,
        dt=0.025,
        render_mode=None,
        hold_deg=10.0,           # window around upright (deg)
        omega_lo=0.3,            # rad/s where clamp starts
        omega_hi=1.2,            # rad/s where clamp is strongest
        u_hold_frac=0.25,        # max |u| allowed (as fraction of umax) when clamping
        lambda_spin=0.2,         # extra penalty *|omega| when inside upright window
        hold_bonus=0.5           # extra bonus when inside window and slow
    ):
        super().__init__(umax=umax, dt=dt, render_mode=render_mode)
        # override reward to include hold shaping
        self.hold_deg = float(hold_deg)
        self.omega_lo = float(omega_lo)
        self.omega_hi = float(omega_hi)
        self.u_hold = float(u_hold_frac) * self.umax
        self.lambda_spin = float(lambda_spin)
        self.hold_bonus_extra = float(hold_bonus)
        self.reward_fun = self._reward_with_hold

    # --- action governor ---
    def _upright_governor(self, u_raw: float) -> float:
        err = wrap_to_pi(self.th - np.pi)
        in_upright = abs(err) < math.radians(self.hold_deg)
        if not in_upright:
            return u_raw

        # If spinning, limit torque that matches spin direction; allow braking.
        s_om = np.sign(self.omega) if abs(self.omega) > 1e-6 else 0.0
        s_u  = np.sign(u_raw) if abs(u_raw)  > 1e-6 else 0.0

        # Speed-dependent clamp: interpolate from full to u_hold as |omega| grows in window
        spd = abs(self.omega)
        if spd <= self.omega_lo:
            clamp = self.umax
        elif spd >= self.omega_hi:
            clamp = self.u_hold
        else:
            alpha = (spd - self.omega_lo) / max(self.omega_hi - self.omega_lo, 1e-6)
            clamp = (1 - alpha) * self.umax + alpha * self.u_hold

        if s_u == s_om and s_u != 0.0:
            # same direction as spin → clamp strongly
            u_cmd = float(np.clip(u_raw, -clamp, clamp))
        else:
            # braking or zero torque allowed up to full limit
            u_cmd = float(np.clip(u_raw, -self.umax, self.umax))

        return u_cmd

    # --- reward shaping near upright ---
    def _reward_with_hold(self):
        # base reward (swing-up)
        base = super()._reward_swing_balance()

        err = wrap_to_pi(self.th - np.pi)
        if abs(err) < math.radians(self.hold_deg):
            # anti-spin + small extra bonus when slow in window
            base -= self.lambda_spin * abs(self.omega)
            if abs(self.omega) < 0.5:
                base += self.hold_bonus_extra
        return base

    # override step to apply governor *before* entering do-not-edit block in parent
    def step(self, action):
        # Accept scalar or (1,)
        if isinstance(action, (list, tuple, np.ndarray)):
            u_raw = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        else:
            u_raw = float(action)
        # gate the torque when near upright
        u_cmd = self._upright_governor(u_raw)
        return super().step(u_cmd)

# -------------------------
# Training & Demo
# -------------------------
def train(args):
    env = Monitor(TimeLimit(UnbalancedDiskUprightGov(umax=args.umax, dt=args.dt, render_mode=None),
                            max_episode_steps=MAX_EP_STEPS))
    eval_env = Monitor(TimeLimit(UnbalancedDiskUprightGov(umax=args.umax, dt=args.dt, render_mode=None),
                                 max_episode_steps=MAX_EP_STEPS))

    policy_kwargs = dict(net_arch=BEST_HP["net_arch"], activation_fn=nn.ReLU)
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=BEST_HP["learning_rate"],
        buffer_size=BEST_HP["buffer_size"],
        batch_size=BEST_HP["batch_size"],
        tau=BEST_HP["tau"],
        gamma=BEST_HP["gamma"],
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=SEED,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.dirname(MODEL_PATH),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1,
    )
    cbs = CallbackList([eval_cb])

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cbs, log_interval=10)

    # Save final model (and keep best from callback directory)
    model.save(MODEL_PATH)
    print(f"[Saved] Final model -> {MODEL_PATH}")
    best_zip = os.path.join(os.path.dirname(MODEL_PATH), "best_model.zip")
    if os.path.exists(best_zip):
        shutil.copy(best_zip, MODEL_PATH)
        print(f"[Saved] Replaced with best eval model -> {MODEL_PATH}")

def demo(args):
    # Load trained model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Train first with --train.")
    model = SAC.load(MODEL_PATH, device="cpu", print_system_info=True)

    # Demo env with rendering
    env = TimeLimit(UnbalancedDiskUprightGov(umax=args.umax, dt=args.dt, render_mode="human"),
                    max_episode_steps=MAX_EP_STEPS)
    obs, _ = env.reset(seed=SEED)
    fps = 60
    try:
        for t in range(2 * MAX_EP_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            th = wrap_to_pi(float(obs[0]))
            om = float(obs[1])
            print(f"t={t:04d} | u={float(action[0]): .3f} | r={r: .3f} | th={th: .3f} | om={om: .3f}")
            env.render()
            time.sleep(1.0 / fps)
            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train with best hyperparameters + upright governor")
    parser.add_argument("--demo",  action="store_true", help="Render a demo rollout using the saved model")
    parser.add_argument("--dt", type=float, default=0.025, help="Integrator step (must match training for best results)")
    parser.add_argument("--umax", type=float, default=3.0, help="Torque saturation")
    args = parser.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)

    if args.train:
        train(args)
    if args.demo:
        demo(args)
    if not args.train and not args.demo:
        print("Nothing to do. Run with --train or --demo.")

if __name__ == "__main__":
    main()