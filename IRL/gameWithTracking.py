import os
import sys
import csv
import time
import math
import numpy as np
import pygame

# import env (adjust path/module if needed)
sys.path.append(os.path.join(os.path.dirname(__file__), "envs"))
from env.UnbalancedDisk import UnbalancedDisk


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def now_str():
    return time.strftime("%Y%m%d-%H%M%S")


def wrap_angle(a):
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class HumanDiskSim:
    def __init__(
        self,
        umax=3.0,
        dt=0.025,
        hz=40,      # 1/dt ≈ 40 Hz
        u_step=0.5, # how much LEFT/RIGHT change voltage
        save_dir="logs",
        ref_amp_deg=15.0,   # ±15°
        ref_freq_hz=0.2,    # 5 s period
    ):
        self.umax = float(umax)
        self.dt = float(dt)
        self.hz = int(hz)
        self.u_step = float(u_step)
        self.save_dir = ensure_dir(save_dir)

        # Let the env own the window (avoids flicker)
        self.env = UnbalancedDisk(umax=self.umax, dt=self.dt, render_mode="human")

        # control state
        self.u = 0.0
        self.running = True
        self.paused = False
        self.auto_zero = True

        # reference signal
        self.ref_on = True
        self.ref_amp = math.radians(ref_amp_deg)
        self.ref_freq = float(ref_freq_hz)
        self.ref_phase = 0.0

        # logging
        self.session_id = now_str()
        self.log_path_csv = os.path.join(self.save_dir, f"disk_human_{self.session_id}.csv")
        self.log_path_npz = os.path.join(self.save_dir, f"disk_human_{self.session_id}.npz")
        self.log = []

        # pygame
        pygame.init()
        pygame.font.init()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)

        # env reset
        self.obs, _ = self.env.reset()

        # UI colors
        self.COLOR_BAR = (30, 144, 255)   # voltage bar
        self.COLOR_TXT = (15, 15, 15)
        self.COLOR_FRAME = (220, 220, 220)
        self.COLOR_REF_RING = (80, 80, 80)
        self.COLOR_REF_DOT = (46, 204, 113)   # green dot
        self.COLOR_THETA_DOT = (231, 76, 60)  # red dot (current theta marker, optional)

    def ref_angle(self, t):
        if not self.ref_on:
            return 0.0
        return self.ref_amp * math.sin(2 * math.pi * self.ref_freq * t + self.ref_phase)

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
                elif event.key == pygame.K_t:
                    self.ref_on = not self.ref_on
                elif event.key == pygame.K_LEFTBRACKET:   # [
                    self.ref_freq = max(0.0, self.ref_freq - 0.02)
                elif event.key == pygame.K_RIGHTBRACKET:  # ]
                    self.ref_freq = min(2.0, self.ref_freq + 0.02)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            if abs(self.obs[0]) < math.radians(65):
                self.u = self.umax
            else:
                delta -= self.u_step
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            if abs(self.obs[0]) < math.radians(65):
                self.u = -self.umax
            else:
                delta += self.u_step
        else:
            self.u = 0.0

        if delta == 0.0 and self.auto_zero:
            self.u *= 0.95

        self.u = float(np.clip(self.u + delta, -self.umax, self.umax))

    def draw_overlay(self, t, theta_ref, err):
        """Transparent HUD with:
           - big ring at screen center
           - green dot at ref angle (±15°)
           - voltage bar + text.  No red theta dot.
        """
        surface = pygame.display.get_surface()
        if surface is None:
            return []
        W, H = surface.get_size()
        hud = pygame.Surface((W, H), pygame.SRCALPHA)
        dirty = []

        # ---- Reference ring + green dot ----
        cx, cy = W // 2, H // 2
        R = int(0.35 * min(W, H))   # ring radius
        ring_thickness = 2
        pygame.draw.circle(hud, self.COLOR_REF_RING, (cx, cy), R, ring_thickness)

        def angle_to_xy(angle_rad, r):
            x = cx + int(r * math.sin(angle_rad))
            y = cy - int(r * math.cos(angle_rad))
            return x, y

        # green ref dot
        ref_x, ref_y = angle_to_xy(theta_ref, R)
        pygame.draw.circle(hud, self.COLOR_REF_DOT, (ref_x, ref_y), 6)

        # small crosshair at top (upright reference line)
        top_x, top_y = angle_to_xy(0.0, R)
        pygame.draw.circle(hud, self.COLOR_REF_RING, (top_x, top_y), 3)

        # ---- Voltage bar ----
        bar_w = int(W * 0.6); bar_h = 16
        bar_x = (W - bar_w) // 2
        bar_y = H - bar_h - 16
        pygame.draw.rect(hud, self.COLOR_FRAME, (bar_x, bar_y, bar_w, bar_h), border_radius=8)
        frac = (self.u + self.umax) / (2 * self.umax)
        fill_w = int(bar_w * frac)
        pygame.draw.rect(hud, self.COLOR_BAR, (bar_x, bar_y, fill_w, bar_h), border_radius=8)
        dirty.append(pygame.Rect(bar_x, bar_y, bar_w, bar_h))

        # ---- Text ----
        info_lines = [
            f"u: {self.u:+.3f} V   umax: ±{self.umax:.1f} V   Hz: {self.hz} (dt={self.dt:.3f})",
            f"theta_ref: {theta_ref:+.3f} rad ({math.degrees(theta_ref):+.1f}°)   "
            f"err: {err:+.3f} rad ({math.degrees(err):+.1f}°)",
            f"[A/LEFT]=+u  [D/RIGHT]=-u   [R]=reset   [Space]=pause   [Z] auto-zero: {'ON' if self.auto_zero else 'OFF'}",
            f"[T] ref: {'ON' if self.ref_on else 'OFF'}   [ [ ] ] freq: {self.ref_freq:.2f} Hz   [S]=save   [Esc/Q]=quit",
            "Chase the green dot around the ring.",
        ]
        y = 10
        for line in info_lines:
            txt = self.font.render(line, True, self.COLOR_TXT)
            r = hud.blit(txt, (10, y))
            dirty.append(r)
            y += 20

        r = surface.blit(hud, (0, 0))
        dirty.append(r)
        return dirty
    
    def log_step(self, t, obs, u, env_reward, theta_ref, err, track_reward):
        th = float(obs[0]); omega = float(obs[1])
        self.log.append(
            {
                "t": t,
                "theta": th,
                "omega": omega,
                "u": float(u),
                "env_reward": float(env_reward),
                "theta_ref": float(theta_ref),
                "track_err": float(err),
                "track_reward": float(track_reward),
            }
        )

    def save_logs(self):
        if not self.log:
            return
        with open(self.log_path_csv, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["t", "theta", "omega", "u", "env_reward", "theta_ref", "track_err", "track_reward"],
            )
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

                theta_ref = self.ref_angle(t)

                if not self.paused:
                    self.obs, env_reward, terminated, truncated, _ = self.env.step(self.u)
                    self.env.render()

                    err = wrap_angle(self.obs[0] - theta_ref)
                    track_reward = -abs(err)

                    t += self.dt
                    self.log_step(t, self.obs, self.u, env_reward, theta_ref, err, track_reward)

                    if terminated or truncated:
                        self.obs, _ = self.env.reset()
                        self.u = 0.0
                else:
                    self.env.render()
                    err = wrap_angle(self.obs[0] - theta_ref)

                dirty = self.draw_overlay(t, theta_ref, err)
                if dirty:
                    pygame.display.update(dirty)

                if max_steps is not None:
                    steps += 1
                    if steps >= max_steps:
                        break
        finally:
            self.env.close()
            self.save_logs()


if __name__ == "__main__":
    sim = HumanDiskSim(
        umax=3.0,
        dt=0.0025,
        hz=40,
        u_step=0.15,
        save_dir="logs",
        ref_amp_deg=15.0,
        ref_freq_hz=0.2,
    )
    sim.run()