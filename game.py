import os
import sys
import csv
import time
import math
import numpy as np
import pygame

# import env (adjust path/module if needed)
sys.path.append(os.path.join(os.path.dirname(__file__), "envs"))
from UnbalancedDisk import UnbalancedDisk


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def now_str():
    return time.strftime("%Y%m%d-%H%M%S")


class HumanDiskSim:
    def __init__(
        self,
        umax=3.0,
        dt=0.025,
        hz=40,      # 1/dt ≈ 40 Hz
        u_step=0.5, # how much LEFT/RIGHT change voltage
        save_dir="logs",
    ):
        self.umax = float(umax)
        self.dt = float(dt)
        self.hz = int(hz)
        self.u_step = float(u_step)
        self.save_dir = ensure_dir(save_dir)

        # env expects obs = [theta, omega]
        self.env = UnbalancedDisk(umax=self.umax, dt=self.dt, render_mode="human")

        # control state
        self.u = 0.0
        self.running = True
        self.paused = False
        self.auto_zero = True  # if True, slowly returns u->0 when no key pressed

        # logging
        self.session_id = now_str()
        self.log_path_csv = os.path.join(self.save_dir, f"disk_human_{self.session_id}.csv")
        self.log_path_npz = os.path.join(self.save_dir, f"disk_human_{self.session_id}.npz")
        self.log = []  # rows of dicts

        # pygame
        pygame.init()
        pygame.font.init()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)

        # env reset
        self.obs, _ = self.env.reset()

        # UI colors
        self.COLOR_BAR = (30, 144, 255)
        self.COLOR_BG = (255, 255, 255)
        self.COLOR_TXT = (15, 15, 15)
        self.COLOR_FRAME = (220, 220, 220)

    def handle_events(self):
        delta = 0.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                elif event.key == pygame.K_r:
                    self.obs, _ = self.env.reset()
                    self.u = 0.0
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_s:
                    self.save_logs()
                elif event.key == pygame.K_z:
                    self.auto_zero = not self.auto_zero

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            if abs(self.obs[0]) < math.radians(70):  # far from top
                self.u = self.umax  # full power
            else:
                delta -= self.u_step        # gentle push
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            if abs(self.obs[0]) < math.radians(70):
                self.u = -self.umax
            else:
                delta += self.u_step
        else:
            self.u = 0.0

        if delta == 0.0 and self.auto_zero:
            self.u *= 0.95

        self.u = float(np.clip(self.u + delta, -self.umax, self.umax))

    def draw_overlay(self):
        surface = pygame.display.get_surface()
        if surface is None:
            return

        W, H = surface.get_size()

        # voltage bar
        bar_w = int(W * 0.6)
        bar_h = 16
        bar_x = (W - bar_w) // 2
        bar_y = H - bar_h - 16
        pygame.draw.rect(surface, self.COLOR_FRAME, (bar_x, bar_y, bar_w, bar_h), border_radius=8)
        frac = (self.u + self.umax) / (2 * self.umax)
        fill_w = int(bar_w * frac)
        pygame.draw.rect(surface, self.COLOR_BAR, (bar_x, bar_y, fill_w, bar_h), border_radius=8)

        th = float(self.obs[0])
        om = float(self.obs[1])
        info_lines = [
            f"Hz: {self.hz}   dt: {self.dt:.3f}",
            f"u: {self.u:+.3f} V   umax: ±{self.umax:.1f} V degree: {math.degrees(self.obs[0])}",
            f"theta: {th:+.3f} rad  ({math.degrees(th):+.1f} deg)",
            f"omega: {om:+.3f} rad/s",
            f"adjust u   [0] zero   [R] reset   [Space] pause/resume",
            f"[Z] auto-zero: {'ON' if self.auto_zero else 'OFF'}   [S] save   [Esc/Q] quit",
        ]
        y = 10
        for line in info_lines:
            txt = self.font.render(line, True, self.COLOR_TXT)
            surface.blit(txt, (10, y))
            y += 20

        pygame.display.flip()

    def log_step(self, t, obs, u, reward):
        th = float(obs[0])
        omega = float(obs[1])
        self.log.append(
            {
                "t": t,
                "theta": th,
                "omega": omega,
                "u": float(u),
                "reward": float(reward),
            }
        )

    def save_logs(self):
        if not self.log:
            return
        with open(self.log_path_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["t", "theta", "omega", "u", "reward"])
            w.writeheader()
            w.writerows(self.log)
        arr = {k: np.array([row[k] for row in self.log]) for k in self.log[0].keys()}
        np.savez(self.log_path_npz, **arr)
        print(f"[saved] {self.log_path_csv}  |  {self.log_path_npz}")

    def run(self, max_steps=None):
        t = 0.0
        steps = 0
        try:
            while self.running:
                self.clock.tick(self.hz)
                self.handle_events()

                if not self.paused:
                    self.obs, reward, terminated, truncated, _ = self.env.step(self.u)
                    t += self.dt
                    self.log_step(t, self.obs, self.u, reward)
                    if terminated or truncated:
                        self.obs, _ = self.env.reset()
                        self.u = 0.0

                    if max_steps is not None:
                        steps += 1
                        if steps >= max_steps:
                            break

                self.env.render()
                self.draw_overlay()
        finally:
            self.env.close()
            self.save_logs()


if __name__ == "__main__":
    sim = HumanDiskSim(
        umax=3.0,
        dt=0.005,
        hz=40,
        u_step=0.15,
        save_dir="logs",
    )
    sim.run()