import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN

from UnbalancedDisk import UnbalancedDisk

# ---------- Compat wrapper for reset(seed=None, options=None) ----------
class ResetCompatWrapper(gym.Wrapper):
    """Makes old-style env.reset() compatible with Gymnasium calling pattern."""
    def reset(self, *, seed=None, options=None):
        # Try the modern signature first
        try:
            return self.env.reset(seed=seed, options=options)
        except TypeError:
            pass

        # Fallbacks if underlying env doesn't support options/seed
        if seed is not None:
            # Try reset(seed=seed)
            try:
                return self.env.reset(seed=seed)
            except TypeError:
                # Last resort: try to set seed via method, then plain reset()
                if hasattr(self.env, "seed"):
                    self.env.seed(seed)
        out = self.env.reset()
        # Ensure (obs, info) tuple
        if isinstance(out, tuple) and len(out) == 2:
            return out
        return out, {}

# ---------- Action & Observation wrappers (unchanged) ----------
class DiscretizeAction(gym.ActionWrapper):
    def __init__(self, env, actions: np.ndarray):
        super().__init__(env)
        self._actions = np.array(actions, dtype=np.float32).ravel()
        self.action_space = gym.spaces.Discrete(len(self._actions))
    def action(self, act_idx: int):
        return float(self._actions[int(act_idx)])

class ObsBoxWrapper(gym.ObservationWrapper):
    def __init__(self, env, low: np.ndarray, high: np.ndarray):
        super().__init__(env)
        self.low = np.array(low, dtype=np.float32)
        self.high = np.array(high, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=self.low, high=self.high, dtype=np.float32)
        self._wrap_angle = (
            np.isclose(self.low[0], -np.pi, atol=1e-4)
            and np.isclose(self.high[0], np.pi, atol=1e-4)
        )
    def observation(self, obs):
        obs = np.array(obs, dtype=np.float32).copy()
        if self._wrap_angle:
            obs[0] = (obs[0] + np.pi) % (2 * np.pi) - np.pi
        return np.clip(obs, self.low, self.high)

def build_torque_grid(n_actions: int, umax: float) -> np.ndarray:
    return np.linspace(-umax, umax, n_actions, dtype=np.float32)

def make_env(dt=0.025, umax=3.0, obs_low=None, obs_high=None, n_actions=None):
    base = UnbalancedDisk(dt=dt, umax=umax, algo="DQN")
    env = ResetCompatWrapper(base)               # <-- put this OUTERMOST so it catches reset()
    if (obs_low is not None) and (obs_high is not None):
        env = ObsBoxWrapper(env, obs_low, obs_high)
    if n_actions is not None:
        env = DiscretizeAction(env, build_torque_grid(n_actions, umax))
    return env

# ---------- Load model first to read saved spaces ----------
model_path = "disc-submission-files/dqn-model.zip"
custom_objects = {"lr_schedule": lambda *_: 0.0, "exploration_schedule": lambda *_: 0.0}
model = DQN.load(model_path, custom_objects=custom_objects, print_system_info=True)

saved_obs_space = model.observation_space  # Box([...], [...], (2,), float32)
saved_act_space = model.action_space       # Discrete(23)
n_actions = int(saved_act_space.n)
obs_low, obs_high = saved_obs_space.low, saved_obs_space.high

# ---------- Build matching env ----------
dt, umax = 0.005, 3.0  # adjust if your training used different values
env = make_env(dt=dt, umax=umax, obs_low=obs_low, obs_high=obs_high, n_actions=n_actions)

# Bind and roll
model.set_env(env)

obs, info = env.reset()
fps = 60
try:
    for t in range(700):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"t={t:03d} | a={int(action):02d} | r={reward: .4f} | theta={obs[0]: .3f} | omega={obs[1]: .3f}")
        env.render()
        time.sleep(1.0 / fps)
        if terminated or truncated:
            obs, info = env.reset()
finally:
    env.close()