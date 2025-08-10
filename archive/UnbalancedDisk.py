import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
from os import path

class UnbalancedDisk(gym.Env):
    """
    Unbalanced disk with IRL reward.
    - algo="DQN" => discrete torques
    - algo in {"SAC","PPO","TD3",...} => continuous torque
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, umax=3.0, dt=0.025, algo="SAC", n_actions=23,
                 reward_model_path="disc-submission-files/reward-model.npz",
                 render_mode="human"):
        super().__init__()

        # Dynamics constants (do not edit)
        self.omega0 = 11.339846957335382
        self.delta_th = 0
        self.gamma = 1.3328339309394384
        self.Ku = 28.136158407237073
        self.Fc = 6.062729509386865
        self.coulomb_omega = 0.001

        # Simulation parameters
        self.umax = float(umax)
        self.dt = float(dt)
        self.algo = algo.upper()
        self.render_mode = render_mode

        # --- Action space ---
        if self.algo == "DQN":
            self.actions = np.linspace(-self.umax, self.umax, int(n_actions), dtype=np.float32)
            self.action_space = spaces.Discrete(int(n_actions))
        else:
            # Continuous torque
            self.actions = None
            self.action_space = spaces.Box(
                low=np.array([-self.umax], dtype=np.float32),
                high=np.array([ self.umax], dtype=np.float32),
                shape=(1,), dtype=np.float32
            )

        # Observation space: wrapped/clipped to training bounds
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -40.0], dtype=np.float32),
            high=np.array([ np.pi,  40.0], dtype=np.float32),
            shape=(2,), dtype=np.float32
        )

        # Load IRL reward model
        m = np.load(reward_model_path)
        self.w = m["weights"].astype(np.float64)
        self.b = float(m["bias"])
        self.phi_mean = m["phi_mean"].astype(np.float64)
        self.phi_std = m["phi_std"].astype(np.float64)
        self.phi_std[self.phi_std == 0.0] = 1.0  # avoid /0

        # state vars
        self.th = 0.0
        self.omega = 0.0
        self.u = 0.0
        self.viewer = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        self.th = float(rng.normal(0.0, 1e-3))
        self.omega = float(rng.normal(0.0, 1e-3))
        self.u = 0.0
        return self._get_obs(), {}

    def step(self, action):
        # map discrete index -> torque for DQN
        if self.algo == "DQN":
            u = float(self.actions[int(action)])
        else:
            u = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        self.u = np.clip(u, -self.umax, self.umax)

        # integrate dynamics
        def f(t, y):
            th, om = y
            dthdt = om
            friction = self.gamma*om + self.Fc*np.tanh(om/self.coulomb_omega)
            domegadt = -self.omega0**2*np.sin(th + self.delta_th) - friction + self.Ku*self.u
            return np.array([dthdt, domegadt], dtype=np.float64)

        sol = solve_ivp(f, [0, self.dt], [self.th, self.omega])
        self.th, self.omega = float(sol.y[0, -1]), float(sol.y[1, -1])

        obs = self._get_obs()
        th, om = float(obs[0]), float(obs[1])
        reward = self._irl_reward(th, om, self.u)

        return obs, reward, False, False, {}

    def _get_obs(self):
        th = self.th + np.random.normal(0.0, 1e-3)
        om = self.omega + np.random.normal(0.0, 1e-3)
        th = (th + np.pi) % (2*np.pi) - np.pi
        om = np.clip(om, -40.0, 40.0)
        return np.array([th, om], dtype=np.float32)

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

    def render(self):
        import pygame
        from pygame import gfxdraw
        screen_width = 500
        screen_height = 500

        if self.viewer is None:
            pygame.init()
            pygame.display.init()
            self.viewer = pygame.display.set_mode((screen_width, screen_height))

        surf = pygame.Surface((screen_width, screen_height))
        surf.fill((255, 255, 255))

        # draw disks
        gfxdraw.filled_circle(surf, screen_width//2, screen_height//2, int(screen_width/2*0.65*1.3), (32, 60, 92))
        gfxdraw.filled_circle(surf, screen_width//2, screen_height//2, int(screen_width/2*0.06*1.3), (132, 132, 126))

        r = screen_width//2*0.40*1.3
        x = int(screen_width//2 - np.sin(self.th) * r)
        y = int(screen_height//2 - np.cos(self.th) * r)
        gfxdraw.filled_circle(surf, x, y, int(screen_width/2*0.22*1.3), (155, 140, 108))
        gfxdraw.filled_circle(surf, x, y, int(screen_width/2*0.22/8*1.3), (71, 63, 48))

        # torque arrow
        fname = path.join(path.dirname(__file__), "clockwise.png")
        if self.u:
            arrow = pygame.image.load(fname)
            arrow_size = abs(float(self.u)/self.umax*screen_height)*0.25
            Z = (arrow_size, arrow_size)
            arrow_rot = pygame.transform.scale(arrow, Z)
            if self.u < 0:
                arrow_rot = pygame.transform.flip(arrow_rot, True, False)

        surf = pygame.transform.flip(surf, False, True)
        self.viewer.blit(surf, (0, 0))
        if self.u:
            self.viewer.blit(arrow_rot, (screen_width//2 - arrow_size//2, screen_height//2 - arrow_size//2))

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
            
if __name__ == '__main__':
    import time
    env = UnbalancedDisk(dt=0.025)

    obs = env.reset()
    Y = [obs]
    env.render()
    try:
        for i in range(100):
            time.sleep(1/24)
            u = 3#env.action_space.sample()
            obs, reward, done, info = env.step(u)
            Y.append(obs)
            env.render()
    finally:
        env.close()
    from matplotlib import pyplot as plt
    import numpy as np
    Y = np.array(Y)
    plt.plot(Y[:,0])
    plt.title(f'max(Y[:,0])={max(Y[:,0])}')
    plt.show()