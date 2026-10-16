"""
Controlled safe-RL environment: 2D navigation with momentum and obstacles.

State  s = (x, y, vx, vy)      position and velocity
Action a = (ax, ay) in [-1,1]^2 scaled by A_MAX to an acceleration
Safety h(s) <= 0 means safe (signed distance to obstacles / walls, positive = unsafe)

Ground-truth feasible set (same idea as FISOR / HJ reachability):
    V(s) = max( h(s), min_a V(f(s, a)) )      (value iteration on a grid)
    feasible  <=>  V(s) <= 0
    boundary band  <=>  |V(s)| < EPS_BAND

Run:  python controlled_env.py
Outputs (in ./env_outputs): ground_truth.npz, dataset.npz, plots
"""

import os

import numpy as np
from scipy.ndimage import map_coordinates

# ----------------------------------------------------------------------------
# Config (all parameters in one place)
# ----------------------------------------------------------------------------
CFG = dict(
    dt=0.1,
    a_max=2.0,  # max acceleration magnitude per axis
    v_max=1.0,  # velocity clip
    arena=1.0,  # arena is [-arena, arena]^2
    obstacles=[(-0.2, 0.3, 0.30), (0.4, -0.3, 0.25)],  # (cx, cy, radius)
    goal=(0.8, 0.8),
    goal_radius=0.15,
    start_low=(-0.9, -0.9),
    start_high=(-0.5, -0.5),
    max_steps=200,
    n_pos=50,
    n_vel=11,  # grid resolution for ground truth
    eps_band=0.02,  # boundary band |V| < eps_band
)

OUT_DIR = "env_outputs"


# ----------------------------------------------------------------------------
# Safety function and dynamics (vectorised over leading dimensions)
# ----------------------------------------------------------------------------
def h_fn(s, cfg=CFG):
    """Signed safety value: h > 0 means unsafe. s[..., 4]."""
    x, y = s[..., 0], s[..., 1]
    h = np.maximum(np.abs(x), np.abs(y)) - cfg["arena"]  # walls
    for cx, cy, r in cfg["obstacles"]:
        h = np.maximum(h, r - np.hypot(x - cx, y - cy))  # obstacles
    return h


def dynamics(s, a, cfg=CFG):
    """Deterministic dynamics. a is in [-1,1]^2 (scaled internally)."""
    dt = cfg["dt"]
    acc = np.clip(a, -1.0, 1.0) * cfg["a_max"]
    p = s[..., :2] + s[..., 2:] * dt
    v = np.clip(s[..., 2:] + acc * dt, -cfg["v_max"], cfg["v_max"])
    return np.concatenate([p, v], axis=-1)


class ControlledEnv:
    """Gym-like environment. step returns (obs, reward, cost, done, info)."""

    def __init__(self, cfg=CFG, seed=0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.s = None
        self.t = 0

    def reset(self, state=None, random_start=False):
        c = self.cfg
        if state is not None:
            self.s = np.array(state, dtype=np.float64)
        elif random_start:
            while True:  # random safe start anywhere
                p = self.rng.uniform(-0.95, 0.95, 2)
                s = np.concatenate([p, self.rng.uniform(-0.6, 0.6, 2)])
                if h_fn(s, c) < -0.05:
                    self.s = s
                    break
        else:
            p = self.rng.uniform(c["start_low"], c["start_high"])
            self.s = np.concatenate([p, self.rng.uniform(-0.2, 0.2, 2)])
        self.t = 0
        return self.s.copy()

    def step(self, a):
        c = self.cfg
        self.s = dynamics(self.s, np.asarray(a, dtype=np.float64), c)
        self.t += 1
        dist = np.linalg.norm(self.s[:2] - np.array(c["goal"]))
        reached = dist < c["goal_radius"]
        cost = float(h_fn(self.s, c) > 0)
        reward = -dist + (5.0 if reached else 0.0)
        out_of_arena = np.max(np.abs(self.s[:2])) > c["arena"]

        timeout = self.t >= c["max_steps"] and not (reached or out_of_arena)

        done = bool(reached or out_of_arena or timeout)
        return (
            self.s.copy(),
            reward,
            cost,
            done,
            dict(
                reached=bool(reached),
                out_of_arena=bool(out_of_arena),
                timeout=bool(timeout),
            ),
        )


# ----------------------------------------------------------------------------
# Ground-truth feasible set via grid value iteration
# ----------------------------------------------------------------------------
def make_grid(cfg=CFG):
    ax = cfg["arena"]
    pos = np.linspace(-ax, ax, cfg["n_pos"])
    vel = np.linspace(-cfg["v_max"], cfg["v_max"], cfg["n_vel"])
    return pos, vel


def _to_index(s, pos, vel):
    """Continuous state -> fractional grid indices for map_coordinates."""
    ip = (s[..., :2] - pos[0]) / (pos[1] - pos[0])
    iv = (s[..., 2:] - vel[0]) / (vel[1] - vel[0])
    return np.stack([ip[..., 0], ip[..., 1], iv[..., 0], iv[..., 1]], axis=0)


def compute_ground_truth(cfg=CFG, max_iter=400, tol=1e-5, verbose=True):
    pos, vel = make_grid(cfg)
    P0, P1, V0, V1 = np.meshgrid(pos, pos, vel, vel, indexing="ij")
    S = np.stack([P0, P1, V0, V1], axis=-1)  # (np,np,nv,nv,4)
    h = h_fn(S, cfg)
    V = h.copy()
    actions = [(ax, ay) for ax in (-1.0, 0.0, 1.0) for ay in (-1.0, 0.0, 1.0)]
    inside = lambda s2: (
        (np.abs(s2[..., 0]) <= cfg["arena"]) & (np.abs(s2[..., 1]) <= cfg["arena"])
    )
    # successor states and which of them leave the arena do not change across iterations
    nxt = [dynamics(S, np.array(a), cfg) for a in actions]
    idx = [_to_index(s2, pos, vel) for s2 in nxt]
    ins = [inside(s2) for s2 in nxt]
    h_next = [h_fn(s2, cfg) for s2 in nxt]
    for it in range(max_iter):
        best = np.full(V.shape, np.inf)
        for k in range(len(actions)):
            Vk = map_coordinates(V, idx[k], order=1, mode="nearest")
            Vk = np.where(
                ins[k], Vk, np.maximum(h_next[k], 1e-3)
            )  # leaving arena = unsafe
            best = np.minimum(best, Vk)
        V_new = np.maximum(h, best)
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if verbose and (it % 20 == 0 or delta < tol):
            print(
                f"  value iteration {it:3d}  max change {delta:.2e}  feasible frac {(V <= 0).mean():.3f}"
            )
        if delta < tol:
            break
    return dict(V=V, pos=pos, vel=vel, h=h)


class GroundTruth:
    """Query the ground-truth value function at arbitrary states."""

    def __init__(self, gt, cfg=CFG):
        self.V, self.pos, self.vel, self.cfg = gt["V"], gt["pos"], gt["vel"], cfg

    def value(self, s):
        s = np.asarray(s, dtype=np.float64)
        flat = s.reshape(-1, 4)
        v = map_coordinates(
            self.V, _to_index(flat, self.pos, self.vel), order=1, mode="nearest"
        )
        v = np.where(np.max(np.abs(flat[:, :2]), axis=1) > self.cfg["arena"], 1.0, v)
        return v.reshape(s.shape[:-1])

    def feasible(self, s):
        return self.value(s) <= 0

    def in_band(self, s, eps=None):
        eps = self.cfg["eps_band"] if eps is None else eps
        return np.abs(self.value(s)) < eps


# ----------------------------------------------------------------------------
# Data collection with several behaviour policies
# ----------------------------------------------------------------------------
def goal_policy(s, rng, cfg=CFG, noise=0.3):
    g = np.array(cfg["goal"])
    a = 3.0 * (g - s[:2]) - 2.0 * s[2:]
    return np.clip(a + rng.normal(0, noise, 2), -1, 1)


def random_policy(s, rng, cfg=CFG):
    return rng.uniform(-1, 1, 2)


def boundary_policy_factory(rng, cfg=CFG):
    """Drives towards a point just outside a random obstacle, then wanders."""
    cx, cy, r = cfg["obstacles"][rng.integers(len(cfg["obstacles"]))]
    ang = rng.uniform(0, 2 * np.pi)
    target = np.array([cx + (r + 0.05) * np.cos(ang), cy + (r + 0.05) * np.sin(ang)])

    def pol(s, rng, cfg=cfg):
        a = 3.0 * (target - s[:2]) - 0.5 * s[2:]
        return np.clip(a + rng.normal(0, 0.2, 2), -1, 1)

    return pol


def collect(
    n_goal=300, n_random=200, n_boundary=300, seed=0, cfg=CFG, start_mode="random"
):
    assert start_mode in ("random", "fixed")
    rng = np.random.default_rng(seed)
    env = ControlledEnv(cfg, seed)
    rows = {
        k: []
        for k in [
            "obs",
            "act",
            "next_obs",
            "rew",
            "cost",
            "done",
            "src",
            "terminal",
            "timeout",
        ]
    }

    def run(policy_fn, n_ep, src, random_start, per_episode_policy=False):
        for _ in range(n_ep):
            pol = policy_fn(rng, cfg) if per_episode_policy else policy_fn
            s = env.reset(random_start=random_start)
            done = False
            while not done:
                a = pol(s, rng)
                s2, r, c, done, info = env.step(a)
                for k, v in zip(
                    rows,
                    [
                        s,
                        a,
                        s2,
                        r,
                        c,
                        float(done),
                        src,
                        float(info["out_of_arena"]),
                        float(info["timeout"]),
                    ],
                ):
                    rows[k].append(v)
                s = s2

    run(goal_policy, n_goal, 0, start_mode == "random")
    run(random_policy, n_random, 1, True)
    run(boundary_policy_factory, n_boundary, 2, True, per_episode_policy=True)
    return {k: np.array(v) for k, v in rows.items()}


def annotate_dataset(data, gt, cfg=CFG):
    """Add ground-truth annotations ('in_band', 'feasible') to a collected dataset."""
    q = GroundTruth(gt, cfg)
    return dict(
        data,
        in_band=q.in_band(data["obs"]),
        feasible=q.feasible(data["obs"]),
    )


# ----------------------------------------------------------------------------
# Sanity checks and plots
# ----------------------------------------------------------------------------
def sanity_checks(gt, cfg=CFG):
    q = GroundTruth(gt, cfg)
    far_rest = np.array([-0.8, 0.8, 0.0, 0.0])
    near_fast = np.array(
        [0.4, -0.3 + 0.25 + 0.15, 0.0, -1.0]
    )  # 0.15 above obstacle, driving down at max speed
    near_slow = np.array([0.4, -0.3 + 0.25 + 0.15, 0.0, 0.0])
    wall_fast = np.array([0.8, 0.0, 1.0, 0.0])  # near wall, moving towards it fast
    print("sanity checks")
    print(
        f"  far from obstacles, at rest      : V={q.value(far_rest):+.3f}  (expect feasible)"
    )
    print(
        f"  near obstacle, at rest           : V={q.value(near_slow):+.3f}  h={h_fn(near_slow):+.3f}  (expect feasible)"
    )
    print(
        f"  near obstacle, driving at it fast: V={q.value(near_fast):+.3f}  h={h_fn(near_fast):+.3f}  (expect infeasible although h<0)"
    )
    print(
        f"  near wall, moving at it fast     : V={q.value(wall_fast):+.3f}  h={h_fn(wall_fast):+.3f}  (expect infeasible although h<0)"
    )
    ok = (
        q.value(far_rest) <= 0
        and q.value(near_slow) <= 0
        and q.value(near_fast) > 0
        and q.value(wall_fast) > 0
    )
    return bool(ok)


def rollout_check(gt, cfg=CFG, n=300, seed=1):
    """Starting from feasible states, a greedy grid-policy must stay safe; from
    infeasible-but-currently-safe states, safety should (almost) always be lost."""
    q = GroundTruth(gt, cfg)
    rng = np.random.default_rng(seed)
    actions = [(ax, ay) for ax in (-1.0, 0.0, 1.0) for ay in (-1.0, 0.0, 1.0)]

    def greedy(s):
        return min(actions, key=lambda a: q.value(dynamics(s, np.array(a), cfg)))

    res = {"feasible": [0, 0], "infeasible": [0, 0]}
    tries = 0
    while min(res["feasible"][1], res["infeasible"][1]) < n and tries < 200000:
        tries += 1
        s = np.concatenate([rng.uniform(-0.95, 0.95, 2), rng.uniform(-1, 1, 2)])
        if h_fn(s, cfg) > 0:
            continue
        key = "feasible" if q.feasible(s) else "infeasible"
        if res[key][1] >= n:
            continue
        ok = True
        for _ in range(60):
            s = dynamics(s, np.array(greedy(s)), cfg)
            if h_fn(s, cfg) > 0:
                ok = False
                break
        res[key][0] += int(ok)
        res[key][1] += 1
    print("rollout check (greedy policy on V, 60 steps)")
    for k, (ok, tot) in res.items():
        print(f"  start {k:10s}: stayed safe in {ok}/{tot}")
    return res


def plot_all(gt, data, cfg=CFG, out=OUT_DIR):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos, vel = gt["pos"], gt["vel"]
    q = GroundTruth(gt, cfg)
    n = 120
    g = np.linspace(-cfg["arena"], cfg["arena"], n)
    X, Y = np.meshgrid(g, g, indexing="ij")
    vels = [(0.0, 0.0), (0.6, 0.0), (0.0, -0.6), (0.6, 0.6)]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    for ax, (vx, vy) in zip(axes, vels):
        S = np.stack([X, Y, np.full_like(X, vx), np.full_like(X, vy)], axis=-1)
        V = q.value(S)
        hh = h_fn(S, cfg)
        ax.imshow(
            (V <= 0).T,
            origin="lower",
            extent=[-1, 1, -1, 1],
            cmap="Greens",
            alpha=0.6,
            vmin=0,
            vmax=1.3,
        )
        ax.contour(X, Y, V, levels=[0], colors="k", linewidths=1.5)
        ax.contour(X, Y, hh, levels=[0], colors="r", linewidths=1.2, linestyles="--")
        ax.plot(*cfg["goal"], "b*", ms=12)
        ax.set_title(f"v=({vx}, {vy})  green=feasible, black=boundary, red--=obstacle")
        ax.set_aspect("equal")
        ax.title.set_fontsize(8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "feasible_region_slices.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4.6))
    o = data["obs"]
    ax.scatter(
        o[:, 0],
        o[:, 1],
        s=1,
        c=np.where(data["in_band"], "tab:red", "tab:blue"),
        alpha=0.4,
    )
    for cx, cy, r in cfg["obstacles"]:
        ax.add_patch(plt.Circle((cx, cy), r, fill=False, color="k"))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_title(
        f"Dataset coverage\n{len(o)} transitions, {int(data['in_band'].sum())} in boundary band (red)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out, "dataset_coverage.png"), dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("computing ground-truth feasible set ...")
    gt = compute_ground_truth()
    np.savez_compressed(os.path.join(OUT_DIR, "ground_truth.npz"), **gt)
    ok = sanity_checks(gt)
    print("sanity checks passed" if ok else "WARNING: sanity checks failed")
    rollout_check(gt)

    print("collecting data ...")
    data = collect()
    data = annotate_dataset(data, gt)
    np.savez_compressed(os.path.join(OUT_DIR, "dataset.npz"), **data)
    unsafe_frac = (data["cost"] > 0).mean()
    print(
        f"dataset: {len(data['obs'])} transitions, "
        f"{int(data['in_band'].sum())} in boundary band, "
        f"{unsafe_frac:.3f} with cost=1, "
        f"{(~data['feasible']).mean():.3f} infeasible states"
    )
    plot_all(gt, data)
    print(f"saved outputs to ./{OUT_DIR}")
