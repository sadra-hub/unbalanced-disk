import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
from os import path
import math

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

    def reset(self, seed=None):
        self.th = np.random.normal(loc=0, scale=0.001)     # near bottom
        self.omega = np.random.normal(loc=0, scale=0.001)
        self.u = 0.0
        return self.get_obs(), {}

    def get_obs(self):
        # keep tiny sensor noise
        self.th_noise = self.th + np.random.normal(loc=0, scale=0.001)
        self.omega_noise = self.omega + np.random.normal(loc=0, scale=0.001)
        return np.array([self.th_noise, self.omega_noise])

    def render(self):
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

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        gfxdraw.filled_circle(  # central blue disk
            self.surf,
            screen_width // 2,
            screen_height // 2,
            int(screen_width / 2 * 0.65 * 1.3),
            (32, 60, 92),
        )
        gfxdraw.filled_circle(  # small middle disk
            self.surf,
            screen_width // 2,
            screen_height // 2,
            int(screen_width / 2 * 0.06 * 1.3),
            (132, 132, 126),
        )

        from math import cos, sin
        r = screen_width // 2 * 0.40 * 1.3
        gfxdraw.filled_circle(  # disk
            self.surf,
            int(screen_width // 2 - sin(th) * r),
            int(screen_height // 2 - cos(th) * r),
            int(screen_width / 2 * 0.22 * 1.3),
            (155, 140, 108),
        )
        gfxdraw.filled_circle(  # small nut
            self.surf,
            int(screen_width // 2 - sin(th) * r),
            int(screen_height // 2 - cos(th) * r),
            int(screen_width / 2 * 0.22 / 8 * 1.3),
            (71, 63, 48),
        )

        fname = path.join(path.dirname(__file__), "clockwise.png")
        self.arrow = pygame.image.load(fname)
        if self.u:
            if isinstance(self.u, (np.ndarray, list)):
                if np.ndim(self.u) == 1:
                    u = self.u[0]
                elif np.ndim(self.u) == 0:
                    u = self.u
                else:
                    raise ValueError(f'u={self.u} is not the correct shape')
            else:
                u = self.u
            arrow_size = abs(float(u) / self.umax * screen_height) * 0.25
            Z = (arrow_size, arrow_size)
            arrow_rot = pygame.transform.scale(self.arrow, Z)
            if self.u < 0:
                arrow_rot = pygame.transform.flip(arrow_rot, True, False)

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.viewer.blit(self.surf, (0, 0))
        if self.u:
            self.viewer.blit(arrow_rot, (screen_width // 2 - arrow_size // 2, screen_height // 2 - arrow_size // 2))
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