# demo_dqn_policy.py
import os
import math
import time
import argparse
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from scipy.integrate import solve_ivp

from stable_baselines3 import DQN
from stable_baselines3.common.save_util import load_from_zip_file


# ---------------------
# Unbalanced Disk Env
# ---------------------
def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

class UnbalancedDisk(gym.Env):
    """
    Same dynamics & reward used during training (no external assets).
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
        self.action_space = spaces.Box(low=-umax, high=umax, shape=tuple())
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -40.0], dtype=np.float32),
            high=np.array([+np.pi, +40.0], dtype=np.float32),
            shape=(2,)
        )
        # reward params
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
        th_err = wrap_to_pi(self.th - np.pi)
        base = math.cos(th_err)
        pen_spin = self.k_omega * (self.omega ** 2)
        pen_u = self.k_u * ((float(self.u) / self.umax) ** 2)
        bonus = self.upright_bonus if (abs(th_err) < math.radians(self.upright_deg) and abs(self.omega) < 1.0) else 0.0
        return base - pen_spin - pen_u + bonus

    def step(self, action):
        self.u = action
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
            pygame.draw.rect(surf, color, pygame.Rect(cx - bar_len // 2, cy - 8, bar_len, 16))
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


class DQN_UnbalancedDisk(UnbalancedDisk):
    """
    Discretize torque into n bins in [-umax, umax].
    """
    def __init__(self, umax=3.0, dt=0.025, n_actions=10, render_mode="human"):
        super().__init__(umax=umax, dt=dt, render_mode=render_mode)
        self.actions = np.linspace(-umax, umax, n_actions, dtype=np.float32)
        self.action_space = spaces.Discrete(n_actions)
        # keep bounded obs box for SB3
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -40.0], dtype=np.float32),
            high=np.array([+np.pi, +40.0], dtype=np.float32),
            shape=(2,),
        )

    def step(self, action):
        u = float(self.actions[int(action)])
        return super().step(u)


# ---------------------
# Loader helpers
# ---------------------
def load_dqn_policy_only(model_path: str, dt: float, umax: float):
    """
    Load a DQN zip across SB3 versions by:
      - reading saved policy_kwargs and action_space from the zip
      - rebuilding a fresh DQN with matching architecture/output size
      - loading only the policy state dict (no optimizer)
    Returns (model, n_actions).
    """
    # Read raw contents (no deserialization of schedules/optimizer)
    data, params, _ = load_from_zip_file(
        model_path, device="cpu", custom_objects={"optimizer_state_dict": None}, print_system_info=True
    )

    # Pull what we need
    policy_kwargs = data.get("policy_kwargs", {}) or {}
    saved_act_space = data.get("action_space", None)
    if saved_act_space is None or not hasattr(saved_act_space, "n"):
        raise RuntimeError("Saved action_space missing or not Discrete; cannot reconstruct DQN.")

    n_actions = int(saved_act_space.n)

    # Build env with matching number of actions
    env = DQN_UnbalancedDisk(n_actions=n_actions, dt=dt, umax=umax, render_mode=None)

    # Recreate model with the SAME policy architecture
    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-3,        # dummy; not training
        exploration_fraction=0.0,  # dummy; not training
        exploration_final_eps=0.0, # dummy; not training
        verbose=0,
        seed=88,
    )

    # Load policy weights only
    key = "policy" if "policy" in params else "policy_state_dict"
    model.policy.load_state_dict(params[key], strict=True)

    return model, n_actions


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="disc-submission-files/dqn-model.zip", help="Path to DQN .zip")
    parser.add_argument("--steps", type=int, default=1500, help="Total steps to run")
    parser.add_argument("--fps", type=int, default=60, help="Render FPS")
    parser.add_argument("--dt", type=float, default=0.025, help="Env integrator dt (must match training if you changed it)")
    parser.add_argument("--umax", type=float, default=3.0, help="Torque limit (must match training if you changed it)")
    parser.add_argument("--no-render", action="store_true", help="Disable pygame window")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    # Robust cross-version load (policy only, architecture matched)
    model, n_actions = load_dqn_policy_only(args.model, dt=args.dt, umax=args.umax)

    # Bind a rendering env that matches the action count
    env = DQN_UnbalancedDisk(n_actions=n_actions, dt=args.dt, umax=args.umax, render_mode=None if args.no_render else "human")
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
            print(f"t={t:04d} | a={int(action):02d}/{n_actions} | r={reward: .4f} | th={th: .3f} | om={om: .3f}")
            if not args.no_render:
                env.render()
                time.sleep(1.0 / max(args.fps, 1))
            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()