# quick_train_dqn.py
import argparse, math, numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from env.UnbalancedDisk import UnbalancedDisk

def wrap_to_pi(x): return (x + math.pi) % (2 * math.pi) - math.pi

class ProgressRewardWrapper(gym.Wrapper):
    """Add k_progress * (cos(err_t) - cos(err_{t-1})), err = wrap_to_pi(theta - pi)."""
    def __init__(self, env, k_progress=0.8):
        super().__init__(env)
        self.k = float(k_progress)
        self._prev_err = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        th = float(obs[0])
        self._prev_err = wrap_to_pi(th - math.pi)
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        th = float(obs[0])
        err = wrap_to_pi(th - math.pi)
        progress = math.cos(err) - math.cos(self._prev_err)
        r += self.k * progress
        self._prev_err = err
        return obs, r, term, trunc, info

class SuccessBonusWrapper(gym.Wrapper):
    """
    If agent gets upright & calm for 'hold_steps' consecutive steps:
      - give +success_bonus once
      - end episode early (helps DQN learn)
    """
    def __init__(self, env, deg=12.0, omega_thresh=1.0, hold_steps=50, success_bonus=10.0):
        super().__init__(env)
        self.err_rad = math.radians(deg)
        self.om_thr = float(omega_thresh)
        self.hold_steps = int(hold_steps)
        self.bonus = float(success_bonus)
        self._count = 0

    def reset(self, **kwargs):
        self._count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        err = abs(wrap_to_pi(float(obs[0]) - math.pi))
        ok = (err < self.err_rad) and (abs(float(obs[1])) < self.om_thr)
        self._count = self._count + 1 if ok else 0
        if self._count >= self.hold_steps:
            r += self.bonus
            term = True  # early success termination
            self._count = 0
        return obs, r, term, trunc, info

class ActionRepeatWrapper(gym.Wrapper):
    """Repeat each chosen action for 'repeat' env steps (training only)."""
    def __init__(self, env, repeat=3):
        super().__init__(env)
        self.repeat = int(repeat)

    def step(self, action):
        total_r = 0.0
        term = trunc = False
        info = {}
        for _ in range(self.repeat):
            obs, r, term, trunc, info = self.env.step(action)
            total_r += float(r)
            if term or trunc:
                break
        return obs, total_r, term, trunc, info

class DQN_UnbalancedDisk(UnbalancedDisk):
    """Discretize torque for DQN; use reward from UnbalancedDisk (+ wrappers)."""
    def __init__(self, umax=3.0, dt=0.005, n_actions=31, render_mode=None):
        super().__init__(umax=umax, dt=dt, render_mode=render_mode)
        # Odd grid so there’s exactly one zero action, plus strong pushes
        self.actions = np.linspace(-umax, umax, n_actions).astype(np.float32)
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -40.0], dtype=np.float32),
            high=np.array([+np.pi, +40.0], dtype=np.float32),
            shape=(2,),
        )

    def step(self, action):
        u = float(self.actions[int(action)])
        obs, reward, term, trunc, info = super().step(u)
        return obs, reward, term, trunc, info

def eval_rollout(env, model, steps=1500, render=False):
    obs, _ = env.reset()
    ret, length = 0.0, 0
    u_hist = []
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        ret += float(r); length += 1
        if hasattr(env.unwrapped, "actions"):
            u_hist.append(float(env.unwrapped.actions[int(action)]))
        if render and getattr(env, "render_mode", None) == "human":
            env.render()
        if term or trunc:
            obs, _ = env.reset()
    if len(u_hist) == 0: u_hist = [0.0]
    u_hist = np.array(u_hist)
    return {
        "return": ret,
        "steps": length,
        "u_mean": float(np.mean(u_hist)),
        "u_std": float(np.std(u_hist)),
        "u_min": float(np.min(u_hist)),
        "u_max": float(np.max(u_hist)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=80000)
    ap.add_argument("--n_actions", type=int, default=31)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--umax", type=float, default=3.0)
    ap.add_argument("--eval_steps", type=int, default=2000)
    ap.add_argument("--render_eval", action="store_true")
    ap.add_argument("--k_progress", type=float, default=0.8)
    ap.add_argument("--hold_steps", type=int, default=50)
    args = ap.parse_args()

    # --- Train env: progress shaping + success bonus + action repeat ---
    base_train = DQN_UnbalancedDisk(n_actions=args.n_actions, dt=args.dt, umax=args.umax, render_mode=None)
    wrapped_train = ProgressRewardWrapper(base_train, k_progress=args.k_progress)
    wrapped_train = SuccessBonusWrapper(wrapped_train, deg=12.0, omega_thresh=1.0,
                                        hold_steps=args.hold_steps, success_bonus=10.0)
    wrapped_train = ActionRepeatWrapper(wrapped_train, repeat=3)
    train_env = Monitor(TimeLimit(wrapped_train, max_episode_steps=600))

    # DQN with slower epsilon decay so it actually tries strong torques
    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=2e-3,
        batch_size=256,
        gamma=0.98,
        exploration_fraction=0.8,     # slower decay
        exploration_final_eps=0.05,
        target_update_interval=1000,
        verbose=1,
        seed=88,
        policy_kwargs=dict(net_arch=[128, 128]),
    )

    print(f"Training DQN for {args.timesteps} steps...")
    model.learn(total_timesteps=args.timesteps)

    # --- Eval env (same shaping, no action repeat; optional render) ---
    base_eval = DQN_UnbalancedDisk(n_actions=args.n_actions, dt=args.dt, umax=args.umax,
                                   render_mode=("human" if args.render_eval else None))
    wrapped_eval = ProgressRewardWrapper(base_eval, k_progress=args.k_progress)
    wrapped_eval = SuccessBonusWrapper(wrapped_eval, deg=12.0, omega_thresh=1.0,
                                       hold_steps=args.hold_steps, success_bonus=10.0)
    eval_env = TimeLimit(wrapped_eval, max_episode_steps=600)

    stats = eval_rollout(eval_env, model, steps=args.eval_steps, render=args.render_eval)
    print("\n=== Quick Eval ===")
    print(f"Return: {stats['return']:.3f} over {stats['steps']} steps")
    print(f"u: mean={stats['u_mean']:.3f} std={stats['u_std']:.3f} "
          f"min={stats['u_min']:.3f} max={stats['u_max']:.3f}")
    print("If it's still timid, bump --timesteps 120000 and --k_progress 1.0; or ping me for a TD3 version.")

if __name__ == "__main__":
    main()