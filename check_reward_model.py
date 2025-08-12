# check_reward_model.py
import argparse
import numpy as np
import matplotlib.pyplot as plt

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

def load_reward(npz_path):
    m = np.load(npz_path)
    required = ["weights", "bias", "phi_mean", "phi_std"]
    for k in required:
        if k not in m:
            raise ValueError(f"Missing key '{k}' in {npz_path}")
    w = m["weights"].astype(np.float32).reshape(-1)
    b = float(np.array(m["bias"]).reshape(()))
    mu = m["phi_mean"].astype(np.float32).reshape(-1)
    sig = m["phi_std"].astype(np.float32).reshape(-1)
    # Basic shape/finite checks
    assert w.shape == mu.shape == sig.shape, "weights/phi_mean/phi_std must have same shape"
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sig)) or not np.isfinite(b):
        raise ValueError("Non-finite values found in reward model arrays.")
    if np.any(sig <= 0):
        raise ValueError("phi_std contains non-positive entries.")
    return w, b, mu, sig

def reward_fn(w, b, mu, sig, theta, omega, u):
    z = (phi_features(theta, omega, u) - mu) / (sig + 1e-8)
    return float(w @ z + b)

def batch_reward(w, b, mu, sig, TH, OM, U):
    # TH, OM, U can be broadcastable arrays
    # vectorized computation over grid
    theta = np.asarray(TH, dtype=np.float32)
    omega = np.asarray(OM, dtype=np.float32)
    u = np.asarray(U, dtype=np.float32)
    # build features per point
    # we’ll loop for clarity; grid sizes are small
    out = np.empty_like(theta, dtype=np.float32)
    it = np.nditer(theta, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        out[idx] = reward_fn(w, b, mu, sig, theta[idx], omega[idx], u[idx] if u.shape == theta.shape else float(u))
        it.iternext()
    return out

def load_expert_csv(csv_path):
    # expects at least columns: theta, omega, u
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    # tolerate extra columns (e.g., theta_ref, track_err); we only use these three
    th = np.asarray(data["theta"], dtype=np.float32)
    om = np.asarray(data["omega"], dtype=np.float32)
    u = np.asarray(data["u"], dtype=np.float32)
    return th, om, u

def sample_random(N, umax=3.0, seed=0):
    rng = np.random.default_rng(seed)
    th = rng.uniform(-np.pi, np.pi, size=N).astype(np.float32)
    om = rng.uniform(-10.0, 10.0, size=N).astype(np.float32)
    u = rng.uniform(-umax, umax, size=N).astype(np.float32)
    return th, om, u

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to reward_model_ref.npz")
    ap.add_argument("--expert_csv", required=True, help="Expert CSV (ref tracking): t,theta,omega,u,theta_ref,track_err")
    ap.add_argument("--umax", type=float, default=3.0)
    ap.add_argument("--show_plots", action="store_true")
    args = ap.parse_args()

    print(f"[Load] {args.npz}")
    w, b, mu, sig = load_reward(args.npz)
    print(f"[OK] weights dim = {w.shape[0]}, ||w|| = {np.linalg.norm(w):.4f}, bias = {b:.4f}")

    # Quick per-feature sanity (signs give hints, not absolute truths)
    names = [
        "theta", "theta^2", "sin(theta)", "cos(theta)",
        "omega", "omega^2", "u", "u^2",
        "|theta*omega|", "|u|*|theta|", "|u|*|omega|", "sign(u)*|theta|"
    ]
    print("\n[Feature weights] (after standardization)")
    for i, (nm, wi) in enumerate(zip(names, w)):
        print(f"  {i:02d} {nm:>14s}: {wi:+.4f}")

    # 1) Expert-vs-random score test
    th_e, om_e, u_e = load_expert_csv(args.expert_csv)
    N = min(len(th_e), 5000)  # cap for speed
    th_e, om_e, u_e = th_e[:N], om_e[:N], u_e[:N]
    th_r, om_r, u_r = sample_random(N, umax=args.umax, seed=123)

    re = np.array([reward_fn(w, b, mu, sig, th_e[i], om_e[i], u_e[i]) for i in range(N)], dtype=np.float32)
    rr = np.array([reward_fn(w, b, mu, sig, th_r[i], om_r[i], u_r[i]) for i in range(N)], dtype=np.float32)

    print("\n[Expert vs Random]")
    print(f"  Expert mean   = {re.mean(): .4f} ± {re.std(): .4f}")
    print(f"  Random  mean  = {rr.mean(): .4f} ± {rr.std(): .4f}")
    gap = re.mean() - rr.mean()
    print(f"  Mean gap (Expert - Random) = {gap: .4f}  --> {'GOOD' if gap > 0 else 'CHECK!'}")

    # 2) 1D slices
    th_grid = np.linspace(-np.pi, np.pi, 721, dtype=np.float32)
    r_th = np.array([reward_fn(w, b, mu, sig, th, 0.0, 0.0) for th in th_grid], dtype=np.float32)

    om_grid = np.linspace(-12, 12, 481, dtype=np.float32)
    r_om = np.array([reward_fn(w, b, mu, sig, np.pi, om, 0.0) for om in om_grid], dtype=np.float32)

    u_grid = np.linspace(-args.umax, args.umax, 201, dtype=np.float32)
    r_u = np.array([reward_fn(w, b, mu, sig, np.pi, 0.0, uu) for uu in u_grid], dtype=np.float32)

    print("\n[1D sanity]")
    print(f"  θ-slice max at θ≈? (index): {th_grid[np.argmax(r_th)]: .3f} rad (expect near +π for upright tracking)")
    print(f"  ω-slice min at |ω| large?   mean at |ω|>8 = {r_om[(np.abs(om_grid)>8)].mean(): .4f}, center ω=0 -> {reward_fn(w,b,mu,sig,np.pi,0.0,0.0): .4f}")
    print(f"  u-slice trend (should usually prefer small |u|):  mean |u|>2 = {r_u[(np.abs(u_grid)>2)].mean(): .4f}")

    # 3) 2D landscape over (theta, omega) at u=0
    if args.show_plots:
        TH, OM = np.meshgrid(np.linspace(-np.pi, np.pi, 181, dtype=np.float32),
                             np.linspace(-10, 10, 121, dtype=np.float32))
        RR = batch_reward(w, b, mu, sig, TH, OM, U=0.0)

        # Heatmap
        plt.figure(figsize=(7, 4))
        plt.title("Reward landscape (u=0)")
        plt.imshow(RR, extent=[-np.pi, np.pi, -10, 10], origin="lower", aspect="auto")
        plt.colorbar(label="r")
        plt.xlabel("theta [rad]")
        plt.ylabel("omega [rad/s]")

        # 1D plots
        plt.figure(figsize=(7, 4))
        plt.plot(th_grid, r_th)
        plt.title("Reward vs theta (omega=0, u=0)")
        plt.xlabel("theta [rad]")
        plt.ylabel("r")

        plt.figure(figsize=(7, 4))
        plt.plot(om_grid, r_om)
        plt.title("Reward vs omega (theta≈pi, u=0)")
        plt.xlabel("omega [rad/s]")
        plt.ylabel("r")

        plt.figure(figsize=(7, 4))
        plt.plot(u_grid, r_u)
        plt.title("Reward vs action (theta≈pi, omega=0)")
        plt.xlabel("u [V]")
        plt.ylabel("r")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()