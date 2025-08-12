# demo_sac_policy.py
import os
import math
import time
import argparse
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from scipy.integrate import solve_ivp

from stable_baselines3 import SAC
from stable_baselines3.common.save_util import load_from_zip_file


# ---------------------
# Utilities
# ---------------------
def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def phi_features(theta, omega, u):
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


# ---------------------
# Unbalanced Disk Env
# ---------------------
class UnbalancedDisk(gym.Env):
    """
    Same dynamics as training. Reward can be:
      - native swing-up (default)
      - IRL reward if provided via set_irl_reward(...)
    """
    metadata = {"render_modes": ["human"], "render_fps": 1 / 0.025}

    def __init__(self, umax=3.0, dt=0.025, render_mode="human"):
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
        self.action_space = spaces.Box(low=np.array([-umax], np.float32),
                                       high=np.array([umax],  np.float32),
                                       shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -40.0], dtype=np.float32),
            high=np.array([+np.pi, +40.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )
        # native reward params
        self.k_omega = 0.10
        self.k_u = 0.01
        self.upright_deg = 12.0
        self.upright_bonus = 0.5

        # reward plumbing
        self._irl = None  # dict with w,b,mu,sig if set
        self.reward_fun = self._reward_swing_balance

        self.render_mode = render_mode
        self.viewer = None
        self.u = 0.0
        self.reset()

    def set_irl_reward(self, w, b, mu, sig):
        """Provide IRL params (np arrays); switches reward to IRL."""
        self._irl = {"w": np.asarray(w, np.float32),
                     "b": float(b),
                     "mu": np.asarray(mu, np.float32),
                     "sig": np.asarray(sig, np.float32)}
        self.reward_fun = self._reward_irl

    def clear_irl_reward(self):
        self._irl = None
        self.reward_fun = self._reward_swing_balance

    # ---- rewards ----
    def _reward_swing_balance(self):
        th_err = wrap_to_pi(self.th - np.pi)
        base = math.cos(th_err)
        pen_spin = self.k_omega * (self.omega ** 2)
        pen_u = self.k_u * ((float(self.u) / self.umax) ** 2)
        bonus = self.upright_bonus if (abs(th_err) < math.radians(self.upright_deg) and abs(self.omega) < 1.0) else 0.0
        return base - pen_spin - pen_u + bonus

    def _reward_irl(self):
        assert self._irl is not None, "IRL reward not configured"
        u = float(self.u)
        z = (phi_features(float(self.th), float(self.omega), u) - self._irl["mu"]) / (self._irl["sig"] + 1e-8)
        return float(self._irl["w"] @ z + self._irl["b"])

    # ---- env API ----
    def step(self, action):
        # SAC passes shape (1,) array; convert to scalar
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

        sol = solve_ivp(f, [0, self.dt], [self.th, self.omega])
        self.th, self.omega = sol.y[:, -1]
        ##### End do not edit   #####

        r = self.reward_fun()
        return self.get_obs(), r, False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.th = np.random.normal(loc=0, scale=0.001)
        self.omega = np.random.normal(loc=0, scale=0.001)
        self.u = 0.0
        return self.get_obs(), {}

    def get_obs(self):
        th_noise = self.th + np.random.normal(loc=0, scale=0.001)
        om_noise = self.omega + np.random.normal(loc=0, scale=0.001)
        return np.array([th_noise, om_noise], dtype=np.float32)

    def render(self):
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
        gfxdraw.filled_circle(surf, W // 2, H // 2, int(W*0.65*0.5), (32, 60, 92))
        gfxdraw.filled_circle(surf, W // 2, H // 2, int(W*0.06*0.5), (132, 132, 126))
        # moving mass
        r = int(W * 0.40 * 0.5)
        cx, cy = W // 2, H // 2
        mx = int(cx - math.sin(th) * r)
        my = int(cy - math.cos(th) * r)
        gfxdraw.filled_circle(surf, mx, my, int(W*0.22*0.5), (155, 140, 108))
        gfxdraw.filled_circle(surf, mx, my, int(W*0.22*0.5/8), (71, 63, 48))
        # torque bar
        if self.u:
            u_norm = float(np.clip(self.u, -self.umax, self.umax)) / self.umax
            bar_len = int(abs(u_norm) * W * 0.25)
            color = (220, 50, 50) if u_norm > 0 else (50, 50, 220)
            import pygame as pg
            pg.draw.rect(surf, color, pg.Rect(cx - bar_len // 2, cy - 8, bar_len, 16))
        surf = pygame.transform.flip(surf, False, True)
        self.viewer.blit(surf, (0, 0))
        import pygame as pg
        pg.event.pump()
        pg.display.flip()
        return True

    def close(self):
        if self.viewer is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.viewer = None


# ---------------------
# Loader helpers (cross-version safe: load policy only)
# ---------------------
def load_sac_policy_only(model_path: str, dt: float, umax: float):
    """
    Load a SAC zip robustly across SB3 versions:
      - read saved policy_kwargs and spaces
      - rebuild fresh SAC with SAME policy architecture
      - load only the policy state_dict (ignore optimizer/schedules)
    Returns (model, obs_space, act_space).
    """
    data, params, _ = load_from_zip_file(
        model_path, device="cpu",
        custom_objects={"optimizer_state_dict": None},  # strip optimizer
        print_system_info=True
    )

    policy_kwargs = data.get("policy_kwargs", {}) or {}
    saved_obs_space = data.get("observation_space", None)
    saved_act_space = data.get("action_space", None)
    if saved_obs_space is None or saved_act_space is None:
        # Fallback to default spaces
        saved_obs_space = gym.spaces.Box(
            low=np.array([-np.pi, -40.0], dtype=np.float32),
            high=np.array([+np.pi, +40.0], dtype=np.float32),
            shape=(2,), dtype=np.float32
        )
        saved_act_space = gym.spaces.Box(low=np.array([-umax], np.float32),
                                         high=np.array([umax],  np.float32),
                                         shape=(1,), dtype=np.float32)

    # Build an env with matching spaces
    env = UnbalancedDisk(umax=umax, dt=dt, render_mode=None)

    # Create model with same policy architecture; LR etc. are dummies for inference
    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-3,
        buffer_size=10_000,
        batch_size=256,
        tau=0.02,
        gamma=0.98,
        verbose=0,
        seed=88,
    )

    # Try direct load first (may work if versions close)
    try:
        model = SAC.load(model_path, custom_objects={"optimizer_state_dict": None}, device="cpu", print_system_info=True)
        return model, saved_obs_space, saved_act_space
    except Exception as e:
        print(f"[Loader] Direct load failed: {e}\n[Loader] Falling back to policy-only load...")

    # Load ONLY the policy state dict
    key = "policy" if "policy" in params else "policy_state_dict"
    model.policy.load_state_dict(params[key], strict=False)
    return model, saved_obs_space, saved_act_space


def maybe_load_irl(npz_path):
    if not npz_path:
        return None
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"IRL reward npz not found: {npz_path}")
    m = np.load(npz_path)
    return dict(w=m["weights"], b=float(m["bias"]), mu=m["phi_mean"], sig=m["phi_std"])


# ---------------------
# Main
# ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to SAC .zip (upright or ref)")
    parser.add_argument("--reward_npz", type=str, default=None, help="Optional IRL reward model npz (to log IRL reward)")
    parser.add_argument("--steps", type=int, default=1500, help="Total steps to run")
    parser.add_argument("--fps", type=int, default=60, help="Render FPS")
    parser.add_argument("--dt", type=float, default=0.025, help="Env integrator dt (should match training)")
    parser.add_argument("--umax", type=float, default=3.0, help="Torque limit (should match training)")
    parser.add_argument("--no-render", action="store_true", help="Disable pygame window")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    # Load model robustly
    model, saved_obs_space, saved_act_space = load_sac_policy_only(args.model, dt=args.dt, umax=args.umax)

    # Build rendering env and (optionally) set IRL reward for logging
    env = UnbalancedDisk(umax=args.umax, dt=args.dt, render_mode=None if args.no_render else "human")
    if args.reward_npz:
        irl = maybe_load_irl(args.reward_npz)
        env.set_irl_reward(irl["w"], irl["b"], irl["mu"], irl["sig"])
        print("[Info] Using IRL reward for logging.")
    env = TimeLimit(env, max_episode_steps=10_000)
    model.set_env(env)

    # Rollout
    obs, _ = env.reset()
    try:
        for t in range(args.steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            th = wrap_to_pi(float(obs[0]))
            om = float(obs[1])
            a = float(np.asarray(action).reshape(-1)[0])
            print(f"t={t:04d} | a={a:+.3f} | r={reward: .4f} | th={th: .3f} | om={om: .3f}")
            if not args.no_render:
                env.render()
                time.sleep(1.0 / max(args.fps, 1))
            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()