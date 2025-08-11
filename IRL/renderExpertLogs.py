import time
import numpy as np
from env.UnbalancedDisk import UnbalancedDisk

LOG_PATH = "/Users/sadra/Documents/M.Sc./5sc28-machineLearningSystemsControl/new/IRL/logs/ExpertLogsWithTracking.csv"
REALTIME = True

def load_log(path):
    d = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    t  = np.asarray(d["t"], float)
    th = np.asarray(d["theta"], float)
    om = np.asarray(d["omega"], float)
    u  = np.asarray(d["u"], float)
    return t, th, om, u

t, th, om, u = load_log(LOG_PATH)
dt = float(np.median(np.diff(t)))

env = UnbalancedDisk(dt=dt, umax=3.0, render_mode=("human" if REALTIME else None))

try:
    for k in range(len(u)-1):
        # Force the exact logged state before applying the logged action
        env.th = float(th[k])
        env.omega = float(om[k])

        uk = float(np.clip(u[k], -env.umax, env.umax))
        obs, reward, terminated, truncated, _ = env.step(uk)

        # Print for verification
        print(f"k={k:04d} | forced θ={th[k]:+.6f} rad ({np.degrees(th[k]):+7.3f}°) "
              f"ω={om[k]:+.6f} rad/s | u={uk:+.3f}")

        if REALTIME:
            env.render()
            time.sleep(dt)

finally:
    env.close()