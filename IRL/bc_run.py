# bc_run.py
# Load a behavior-cloned policy and run it in UnbalancedDisk (optionally render).

import argparse, math, numpy as np, torch
from env.UnbalancedDisk import UnbalancedDisk

class BCPolicy(torch.nn.Module):
    def __init__(self, in_dim=3, hidden=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x)

@torch.no_grad()
def act(model, obs, x_mean, x_std, y_mean, y_std, umax):
    th, om = float(obs[0]), float(obs[1])
    x = np.array([math.sin(th), math.cos(th), om], dtype=np.float32)[None, :]
    xz = (x - x_mean) / x_std
    uz = model(torch.tensor(xz, dtype=torch.float32)).cpu().numpy()[0, 0]
    u  = float(uz * y_std[0] + y_mean[0])
    return float(np.clip(u, -umax, umax))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="path to bc_policy.pt")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--umax", type=float, default=3.0)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    # load checkpoint
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    hidden = int(ckpt.get("hidden", 128))
    model = BCPolicy(in_dim=int(ckpt.get("in_dim", 3)), hidden=hidden)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    x_mean = ckpt["x_mean"]; x_std = ckpt["x_std"]
    y_mean = ckpt["y_mean"]; y_std = ckpt["y_std"]
    umax = float(ckpt.get("umax", args.umax))

    env = UnbalancedDisk(dt=args.dt, umax=umax,
                         render_mode=("human" if args.render else None))
    obs, _ = env.reset()

    ret = 0.0
    for k in range(args.steps):
        u = act(model, obs, x_mean, x_std, y_mean, y_std, umax)
        obs, r, term, trunc, _ = env.step(u)
        ret += float(r)
        if args.render:
            env.render()
        if term or trunc:
            obs, _ = env.reset()

    env.close()
    print(f"Total return: {ret:.3f} over {args.steps} steps")

if __name__ == "__main__":
    main()