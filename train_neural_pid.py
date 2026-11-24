# ============================================================
# NEURAL PID: AMORTISED EXACT GAUSSIAN PID (CCS redundancy, Williams-Beer lattice)
#
# What the network learns: ONLY the Monte Carlo CCS redundancy (the one PID term with
# no closed form), as a fraction of the joint MI. I(X1;T), I(X2;T) and I(X1,X2;T) are
# computed exactly, and the lattice (incl. forcenn) is applied afterwards, exactly as
# calc_pi_mvn does. So all atoms are exactly 0 when the target is unrelated, and
# R + U1 = I1, R + U2 = I2, R + U1 + U2 + S = JMI hold by construction.
#
# Significance: the permutation null depends only on (r12, N). It is computed at
# inference by Monte Carlo through the same network, so there are no null heads.
#
# Sources: gcmi.copnorm. Binary targets: 0/1 indicator (NOT copula-normalised).
# Continuous targets: gcmi.copnorm.
# ============================================================

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import gcmi
import scipy.special as sps    # used as sps.ndtri (a bare ufunc import does not pickle into joblib workers)

try:
    from joblib import Parallel, delayed
    HAVE_JOBLIB = True
except ImportError:
    HAVE_JOBLIB = False

sys.path.insert(0, "/content/PyPID_mvn")
try:
    import pid_mvn          # only used to cross-check the teacher; not required
except ImportError:
    pid_mvn = None

LN2 = math.log(2.0)

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    "output_dir": "/content/drive/MyDrive/neural_pid_output",
    "n_train": 100_000,
    "n_val": 10_000,
    "n_test": 10_000,
    # Teacher: exact CCS redundancy, Monte Carlo samples per example.
    # pid_mvn uses 100,000; more samples = cleaner training targets.
    "teacher_nmc": 200_000,
    # Mixture of correlation-matrix generators: (uniform, weak-target, from simulated data).
    "mix": (0.3, 0.5, 0.2),
    "s_min": 1e-3,                  # smallest target strength sqrt(r1t^2 + r2t^2) sampled
    "min_det": 1e-8,                # reject near-singular correlation matrices
    "data_n_min": 10,
    "data_n_max": 20_000,
    # Network.
    "hidden_dim": 128,
    "n_hidden_layers": 4,
    "batch_size": 1024,
    "epochs": 300,
    "learning_rate": 1e-3,
    "weight_decay": 1e-6,
    "model_error_floor": 0.05,      # floor on the per-example noise SD (in R/JMI units) for loss weights
    "checkpoint_every": 10,
    # Generation.
    "n_jobs": -1,
    "chunk": 250,
    "seed": 42,
    # Inference defaults.
    "n_null": 1000,
    "p_value_thresh": 0.05,
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 1. CLOSED-FORM PIECES AND THE EXACT CCS REDUNDANCY (TEACHER)
# ============================================================

def corr_det(r12, r1t, r2t):
    return 1.0 - r12 ** 2 - r1t ** 2 - r2t ** 2 + 2.0 * r12 * r1t * r2t


def closed_form_mi(r12, r1t, r2t):
    """I(X1;T), I(X2;T), I(X1,X2;T) in bits for unit-variance Gaussians (vectorised)."""
    r12, r1t, r2t = (np.asarray(v, dtype=np.float64) for v in (r12, r1t, r2t))
    I1 = -0.5 * np.log1p(-r1t ** 2) / LN2
    I2 = -0.5 * np.log1p(-r2t ** 2) / LN2
    # JMI = -0.5 log2(1 - R^2) with R^2 the squared multiple correlation of T on (X1, X2);
    # log1p keeps full precision for weak targets and gives exactly 0 when r1t = r2t = 0.
    rsq = (r1t ** 2 + r2t ** 2 - 2 * r12 * r1t * r2t) / np.maximum(1.0 - r12 ** 2, 1e-15)
    JMI = -0.5 * np.log1p(-np.clip(rsq, 0.0, 1.0 - 1e-15)) / LN2
    return I1, I2, JMI


def iccs_redundancy(r12, r1t, r2t, nmc, rng):
    """Raw CCS redundancy (bits) and its Monte Carlo standard error.

    Same estimator as pid_mvn.Iccs_mvn for varsizes [1, 1, 1] (local co-information,
    kept where sign(dhx) = sign(dhy) = sign(dhxy) = sign(lnii), averaged over all samples),
    with the Gaussian log densities written in closed form.
    """
    a, b, c = float(r1t), float(r2t), float(r12)
    if a == 0.0 or b == 0.0:
        return 0.0, 0.0                       # exact: dhx (or dhy) is identically 0
    C = np.array([[1.0, c, a], [c, 1.0, b], [a, b, 1.0]])
    L = np.linalg.cholesky(C)
    z = rng.standard_normal((nmc, 3)) @ L.T
    x1, x2, t = z[:, 0], z[:, 1], z[:, 2]

    va, vb = 1.0 - a * a, 1.0 - b * b
    dhx = -0.5 * np.log(va) - (x1 - a * t) ** 2 / (2 * va) + 0.5 * x1 ** 2
    dhy = -0.5 * np.log(vb) - (x2 - b * t) ** 2 / (2 * vb) + 0.5 * x2 ** 2

    s11, s22, s12 = va, vb, c - a * b         # Cov(X1, X2 | T)
    det_s = s11 * s22 - s12 * s12
    det_x = 1.0 - c * c
    e1, e2 = x1 - a * t, x2 - b * t
    q_cond = (s22 * e1 * e1 - 2 * s12 * e1 * e2 + s11 * e2 * e2) / det_s
    q_marg = (x1 * x1 - 2 * c * x1 * x2 + x2 * x2) / det_x
    dhxy = -0.5 * np.log(det_s / det_x) - 0.5 * q_cond + 0.5 * q_marg

    lnii = dhx + dhy - dhxy
    sx = np.sign(dhx)
    keep = (sx == np.sign(lnii)) & (sx == np.sign(dhy)) & (sx == np.sign(dhxy))
    lnii = np.where(keep, lnii, 0.0)
    return lnii.mean() / LN2, lnii.std(ddof=1) / math.sqrt(nmc) / LN2


def wishart_corr(Sigma0, n_samples, n_draws, rng):
    """Sample correlation matrices of N Gaussian samples drawn from Sigma0 (Bartlett
    decomposition of Wishart(N-1, Sigma0)). Sigma0 (k,d,d), n_samples (k,) -> (k,n_draws,d,d)."""
    Sigma0 = np.asarray(Sigma0, dtype=np.float64)
    k, d, _ = Sigma0.shape
    df = np.maximum(np.asarray(n_samples, dtype=np.float64).reshape(k), d + 1.0) - 1.0
    L = np.linalg.cholesky(Sigma0)
    A = np.zeros((k, n_draws, d, d))
    il = np.tril_indices(d, -1)
    A[:, :, il[0], il[1]] = rng.standard_normal((k, n_draws, len(il[0])))
    dof = np.broadcast_to(df[:, None, None] - np.arange(d)[None, None, :], (k, n_draws, d))
    A[:, :, np.arange(d), np.arange(d)] = np.sqrt(rng.chisquare(dof))
    LA = np.einsum("kij,knjl->knil", L, A)
    W = LA @ np.swapaxes(LA, -1, -2)
    s = np.sqrt(np.einsum("...ii->...i", W))
    return W / (s[..., :, None] * s[..., None, :])


def atom_pvalues(obs, null_m1, null_m2, null_c1, null_c2, n_null):
    """Atom-specific p-values from four null ensembles of atoms (k, n_null, 4):
    m1/m2: source 1/2 has no marginal relation to T (R = 0 under either -> p_R = max);
    c1/c2: T independent of source 1/2 given the other (U1 from c1, U2 from c2,
    S needs both sources to add information -> p_S = max).
    Returns p (k,4) and the mean/SD of the null that gave each p."""
    def pv(null, o):
        return (1 + np.sum(null >= o[:, None] - 1e-15, axis=1)) / (n_null + 1)
    pr1, pr2 = pv(null_m1[..., 0], obs[:, 0]), pv(null_m2[..., 0], obs[:, 0])
    ps1, ps2 = pv(null_c1[..., 3], obs[:, 3]), pv(null_c2[..., 3], obs[:, 3])
    p = np.column_stack([np.maximum(pr1, pr2), pv(null_c1[..., 1], obs[:, 1]),
                         pv(null_c2[..., 2], obs[:, 2]), np.maximum(ps1, ps2)])
    nR = np.where((pr1 >= pr2)[:, None], null_m1[..., 0], null_m2[..., 0])
    nS = np.where((ps1 >= ps2)[:, None], null_c1[..., 3], null_c2[..., 3])
    nulls = np.stack([nR, null_c1[..., 1], null_c2[..., 2], nS], axis=-1)
    return p, nulls.mean(1), nulls.std(1)


def assemble_atoms(R_raw, I1, I2, JMI, forcenn=True):
    """Williams-Beer lattice exactly as calc_pi_mvn: R, U1, U2 clipped at 0 if forcenn, S never."""
    R = np.maximum(R_raw, 0.0) if forcenn else R_raw
    U1, U2 = I1 - R, I2 - R
    if forcenn:
        U1, U2 = np.maximum(U1, 0.0), np.maximum(U2, 0.0)
    S = JMI - R - U1 - U2
    return np.stack([R, U1, U2, S], axis=-1)


# ============================================================
# 2. TURNING DATA INTO A CORRELATION MATRIX (USE THIS IN YOUR PIPELINES)
# ============================================================

def copnorm_1d(x):
    """gcmi copula normalisation (ordinal ranks; ties broken by sample order)."""
    x = np.asarray(x, dtype=np.float64)
    return np.asarray(gcmi.copnorm(x[None, :]), dtype=np.float64).ravel()


def binary_indicator(y):
    """Binary target as a 0/1 indicator (NOT copula-normalised).
    Centring/scaling does not change the correlation matrix, so it is left as 0/1."""
    y = np.asarray(y)
    levels = np.unique(y)
    if len(levels) != 2:
        raise ValueError(f"Binary target needs exactly 2 levels, got {len(levels)}.")
    return (y == levels[1]).astype(np.float64)


def correlation_from_data(x1, x2, y, target="binary"):
    """3x3 correlation matrix [x1, x2, target]: sources gcmi-copnormed, binary target as a
    0/1 indicator, continuous target gcmi-copnormed."""
    z1, z2 = copnorm_1d(x1), copnorm_1d(x2)
    zt = binary_indicator(y) if target == "binary" else copnorm_1d(y)
    return np.corrcoef(np.vstack([z1, z2, zt]))


# ============================================================
# 3. TRAINING DISTRIBUTION OF CORRELATION MATRICES
# ============================================================

def _valid(r12, r1t, r2t):
    return corr_det(r12, r1t, r2t) > CONFIG["min_det"] and abs(r12) < 1 and abs(r1t) < 1 and abs(r2t) < 1


def uniform_example(rng):
    while True:
        A = rng.standard_normal((3, 3))
        C = A @ A.T
        d = np.sqrt(np.diag(C))
        C = C / np.outer(d, d)
        r = (C[0, 1], C[0, 2], C[1, 2])
        if _valid(*r):
            return r


def weak_example(rng):
    """Target strength s log-uniform down to s_min; extra density near collinear sources
    and near the axes (one source almost unrelated to the target)."""
    while True:
        c = rng.uniform(-1, 1) if rng.random() < 0.8 else rng.choice([-1, 1]) * (1 - 10 ** rng.uniform(-4, -1))
        u = rng.random()
        if u < 0.7:
            th = rng.uniform(0, np.pi / 2)
        else:
            small = 10 ** rng.uniform(-4, math.log10(0.2))
            th = small if u < 0.85 else np.pi / 2 - small
        s = 10 ** rng.uniform(math.log10(CONFIG["s_min"]), 0)
        a, b = s * math.cos(th), s * math.sin(th)
        s1, s2 = rng.choice([-1.0, 1.0], size=2)
        r = (c * s1 * s2, a * s1, b * s2)
        if _valid(*r):
            return r


def _synthetic_signal(n, rng, freqs, t):
    phase = rng.uniform(0, 2 * np.pi, len(freqs))
    amp = rng.uniform(0.5, 1.0, len(freqs))
    sig = (amp[:, None] * np.sin(2 * np.pi * freqs[:, None] * t[None, :] + phase[:, None])).sum(0)
    sig -= sig.mean()
    sd = sig.std()
    return sig / sd if sd > 0 else sig


def simulate_emg(n, rng):
    """Your EMG simulator (vectorised). Target is the binary task parameter y1."""
    sig_corr = rng.uniform(-0.99, 0.99)
    noise_corr = rng.uniform(-0.99, 0.99)
    trials = max(1, min(100, n // 100))
    sub = max(10, n // trials)
    freqs = np.arange(1, 151, dtype=np.float64)
    t = np.arange(sub, dtype=np.float64) / sub
    e1 = np.concatenate([_synthetic_signal(sub, rng, freqs, t) for _ in range(trials)])
    e2 = np.concatenate([_synthetic_signal(sub, rng, freqs, t) for _ in range(trials)])
    L = np.linalg.cholesky(np.array([[1.0, noise_corr], [noise_corr, 1.0]]) + 1e-12 * np.eye(2))
    noise = rng.standard_normal((sub * trials, 2)) @ L.T
    y1 = np.tile(np.concatenate([np.zeros(sub), np.ones(sub)]), trials)[: sub * trials]
    agree = rng.random(len(y1)) < abs(sig_corr)
    if sig_corr >= 0:
        y2 = np.where(agree, y1, 1 - y1)
    else:
        y2 = np.where(agree, 1 - y1, y1)
    x1 = e1 * y1 + noise[:, 0]
    x2 = e2 * y2 + noise[:, 1]
    return x1[:n], x2[:n], y1[:n]


def data_example(rng):
    """Correlation matrix computed from simulated data, finite N (sampling noise included)."""
    while True:
        n = int(10 ** rng.uniform(math.log10(CONFIG["data_n_min"]), math.log10(CONFIG["data_n_max"])))
        kind = rng.integers(3)
        if kind == 0:
            x1, x2, y = simulate_emg(n, rng)
            target = "binary"
        else:
            r12, r1t, r2t = uniform_example(rng)
            k = 10 ** rng.uniform(-2, 0)          # weaken the target
            r1t, r2t = r1t * k, r2t * k
            C = np.array([[1, r12, r1t], [r12, 1, r2t], [r1t, r2t, 1.0]])
            Z = rng.standard_normal((n, 3)) @ np.linalg.cholesky(C).T
            x1, x2 = Z[:, 0], Z[:, 1]
            if kind == 1:                          # binary target, random prevalence
                p = 10 ** rng.uniform(-2, math.log10(0.5))
                y = (Z[:, 2] > sps.ndtri(1 - p)).astype(np.int8)
                target = "binary"
            else:                                  # continuous target
                y = Z[:, 2]
                target = "continuous"
        if target == "binary" and min(np.sum(y == 0), np.sum(y != 0)) < 2:
            continue
        C = correlation_from_data(x1, x2, y, target)
        r = (C[0, 1], C[0, 2], C[1, 2])
        if np.all(np.isfinite(r)) and _valid(*r):
            return r


GENERATORS = (uniform_example, weak_example, data_example)


def generate_chunk(n, seed, nmc, mix):
    """Rows: r12, r1t, r2t, R_raw, R_se, I1, I2, JMI, generator id."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, 9))
    cum = np.cumsum(mix)
    for i in range(n):
        kind = int(np.searchsorted(cum, rng.random() * cum[-1], side="right"))
        r12, r1t, r2t = GENERATORS[kind](rng)
        R, se = iccs_redundancy(r12, r1t, r2t, nmc, rng)
        I1, I2, JMI = closed_form_mi(r12, r1t, r2t)
        out[i] = (r12, r1t, r2t, R, se, I1, I2, JMI, kind)
    return out


def generate_dataset(name, n_examples, seed_offset):
    out_dir = Path(CONFIG["output_dir"])
    final = out_dir / f"{name}.npz"
    if final.exists():
        print(f"Loading existing {name} set: {final}")
        return np.load(final)["rows"]

    partial = out_dir / f"{name}_partial.npz"
    chunk = CONFIG["chunk"]
    sizes = [min(chunk, n_examples - i) for i in range(0, n_examples, chunk)]
    rows = np.load(partial)["rows"] if partial.exists() else np.zeros((0, 9))
    done = int(np.searchsorted(np.cumsum(sizes), len(rows), side="right"))
    rows = rows[: int(np.sum(sizes[:done]))]
    if done:
        print(f"Resuming {name}: {len(rows):,}/{n_examples:,}")

    n_jobs = CONFIG["n_jobs"] if HAVE_JOBLIB else 1
    per_round = 4 * (max(1, torch.get_num_threads()) if n_jobs == -1 else max(1, n_jobs))
    t0 = time.time()
    base = CONFIG["seed"] * 1_000_003 + seed_offset
    for start in range(done, len(sizes), per_round):
        idx = range(start, min(start + per_round, len(sizes)))
        args = [(sizes[k], base + k, CONFIG["teacher_nmc"], CONFIG["mix"]) for k in idx]
        if HAVE_JOBLIB and n_jobs != 1:
            parts = Parallel(n_jobs=n_jobs)(delayed(generate_chunk)(*a) for a in args)
        else:
            parts = [generate_chunk(*a) for a in args]
        rows = np.vstack([rows] + parts)
        np.savez_compressed(partial, rows=rows)
        rate = (len(rows) - sum(sizes[:done])) / max(time.time() - t0, 1e-9)
        print(f"\r  {name}: {len(rows):,}/{n_examples:,} ({rate:,.0f}/s)", end="")
    print()
    np.savez_compressed(final, rows=rows)
    if partial.exists():
        partial.unlink()
    return rows


# ============================================================
# 4. FEATURES AND NETWORK
# ============================================================

def pid_features(r12, r1t, r2t):
    """Sign-invariant, scale-separated inputs:
    [direction of (|r1t|,|r2t|) in [-1,1], source correlation (sign-aligned),
     target strength s, squared multiple correlation of T on (X1,X2) in [-1,1]].
    Swapping the sources only negates column 0."""
    r12, r1t, r2t = (np.asarray(v, dtype=np.float64) for v in (r12, r1t, r2t))
    a, b = np.abs(r1t), np.abs(r2t)
    c = r12 * np.where(r1t < 0, -1.0, 1.0) * np.where(r2t < 0, -1.0, 1.0)
    s = np.hypot(a, b)
    phi = np.where(s > 0, np.arctan2(b, a) / (np.pi / 2), 0.5)
    rsq = 1.0 - corr_det(c, a, b) / np.maximum(1.0 - c ** 2, 1e-15)
    return np.stack([2 * phi - 1, c, s, 2 * np.clip(rsq, 0, 1) - 1], axis=-1)


class RedundancyNet(nn.Module):
    """Predicts rho = R_raw / JMI. Exactly symmetric in the two sources."""

    def __init__(self, hidden_dim=128, n_hidden_layers=4):
        super().__init__()
        layers = [nn.Linear(4, hidden_dim), nn.GELU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, f):
        g = torch.cat([-f[:, :1], f[:, 1:]], dim=1)
        return 0.5 * (self.mlp(f) + self.mlp(g)).squeeze(-1)


def training_arrays(rows):
    r12, r1t, r2t, R, se, I1, I2, JMI = rows[:, :8].T
    ok = (JMI > 1e-12) & (r1t != 0) & (r2t != 0) & np.isfinite(R) & np.isfinite(se)
    F = pid_features(r12[ok], r1t[ok], r2t[ok])
    rho = R[ok] / JMI[ok]
    se_rho = se[ok] / JMI[ok]
    w = 1.0 / (se_rho ** 2 + CONFIG["model_error_floor"] ** 2)
    return F, rho, w / w.mean()


def train_model(model, train, val, device):
    out_dir = Path(CONFIG["output_dir"])
    to_t = lambda a: torch.tensor(a, dtype=torch.float64)
    train_ds = torch.utils.data.TensorDataset(*(to_t(a) for a in train))
    val_ds = torch.utils.data.TensorDataset(*(to_t(a) for a in val))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=8192)

    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["epochs"], eta_min=1e-6)
    ckpt_path = out_dir / "checkpoint.pt"
    best, history, start = np.inf, [], 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        best, history, start = ck["best"], ck["history"], ck["epoch"] + 1
        print(f"Resuming training at epoch {start + 1}")

    def wmse(xb, yb, wb):
        return torch.mean(wb * (model(xb) - yb) ** 2)

    for epoch in range(start, CONFIG["epochs"]):
        model.train()
        tl = []
        for xb, yb, wb in train_loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad()
            loss = wmse(xb, yb, wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl.append(loss.item())
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = np.mean([wmse(x.to(device), y.to(device), w.to(device)).item() for x, y, w in val_loader])
        history.append((epoch, float(np.mean(tl)), float(vl)))
        if (epoch + 1) % 10 == 0 or epoch == start:
            print(f"Epoch {epoch + 1:4d}/{CONFIG['epochs']} | train {np.mean(tl):.5e} | val {vl:.5e}")
        if vl < best:
            best = vl
            torch.save(model.state_dict(), out_dir / "redundancy_net_best.pt")
        if (epoch + 1) % CONFIG["checkpoint_every"] == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "best": best, "history": history, "epoch": epoch}, ckpt_path)
    np.save(out_dir / "training_history.npy", np.asarray(history))
    model.load_state_dict(torch.load(out_dir / "redundancy_net_best.pt", map_location=device))
    return history


# ============================================================
# 5. ESTIMATOR (DROP-IN: predict_from_covariance(C, n_samples))
# ============================================================

class NeuralPIDEstimator:

    def __init__(self, model, device="cpu", n_null=1000, p_value_thresh=0.05, seed=0):
        self.model = model.to(device=device, dtype=torch.float64).eval()
        self.device = device
        self.n_null = n_null
        self.p_value_thresh = p_value_thresh
        self.rng = np.random.default_rng(seed)

    @classmethod
    def load(cls, output_dir, device=None, **kw):
        output_dir = Path(output_dir)
        meta = json.loads((output_dir / "model_metadata.json").read_text())
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = RedundancyNet(meta["hidden_dim"], meta["n_hidden_layers"]).to(torch.float64)
        model.load_state_dict(torch.load(output_dir / "redundancy_net.pt", map_location=device))
        return cls(model, device, **kw)

    def _raw_redundancy(self, r12, r1t, r2t, JMI, batch=262_144):
        F = pid_features(r12, r1t, r2t).reshape(-1, 4)
        rho = np.empty(len(F))
        with torch.no_grad():
            for i in range(0, len(F), batch):
                x = torch.tensor(F[i:i + batch], dtype=torch.float64, device=self.device)
                rho[i:i + batch] = self.model(x).cpu().numpy()
        rho = rho.reshape(np.shape(JMI))
        zero = (np.asarray(r1t) == 0) | (np.asarray(r2t) == 0) | (JMI <= 0)
        return np.where(zero, 0.0, rho * JMI)

    def atoms_from_correlations(self, r12, r1t, r2t, forcenn=True):
        r12, r1t, r2t = (np.asarray(v, dtype=np.float64) for v in (r12, r1t, r2t))
        I1, I2, JMI = closed_form_mi(r12, r1t, r2t)
        R_raw = self._raw_redundancy(r12, r1t, r2t, JMI)
        return assemble_atoms(R_raw, I1, I2, JMI, forcenn), R_raw, I1, I2, JMI

    def null_correlations(self, r12, n_samples, n_null):
        """Permutation null of (r1t, r2t) given r12 and N: corr(r1t, r2t) = r12 under
        permutation; Fisher-z marginals with variance 1/(N-3)."""
        r12 = np.asarray(r12, dtype=np.float64)[:, None]
        n = np.asarray(n_samples, dtype=np.float64).reshape(-1, 1)
        scale = 1.0 / np.sqrt(np.maximum(n - 3.0, 1.0))
        z1 = self.rng.standard_normal((len(r12), n_null))
        z2 = r12 * z1 + np.sqrt(np.maximum(1 - r12 ** 2, 0)) * self.rng.standard_normal(z1.shape)
        a, b = np.tanh(z1 * scale), np.tanh(z2 * scale)
        rr = np.broadcast_to(r12, a.shape)
        for _ in range(50):
            bad = corr_det(rr, a, b) <= CONFIG["min_det"]
            if not bad.any():
                break
            a, b = np.where(bad, 0.9 * a, a), np.where(bad, 0.9 * b, b)
        return rr, a, b

    def marginal_null(self, r12, r1t, r2t, zero, n_samples, n_null):
        """H0: source `zero` (1 or 2) is uncorrelated with T, with r12 and the other source's
        correlation with T at their observed values (the latter shrunk only if needed to
        keep the matrix positive definite). Parametric bootstrap from Wishart(N-1, Sigma0)."""
        keep = r1t if zero == 2 else r2t
        lim = np.sqrt(np.maximum(0.999 - r12 ** 2, 0.0))
        keep = np.clip(keep, -lim, lim)
        k = len(r12)
        S = np.zeros((k, 3, 3))
        S[:, np.arange(3), np.arange(3)] = 1.0
        S[:, 0, 1] = S[:, 1, 0] = r12
        j = 0 if zero == 2 else 1
        S[:, j, 2] = S[:, 2, j] = keep
        W = wishart_corr(S, n_samples, n_null, self.rng)
        return W[..., 0, 1], W[..., 0, 2], W[..., 1, 2]

    def conditional_null(self, r12, r1t, r2t, test, n_samples, n_null):
        """H0: T is independent of source `test` (1 or 2) given the other source. Exact for
        Gaussian data conditional on the observed sources and the other source's relation
        to T: r_test,t = r12 r_other,t + sqrt(1-r12^2) sqrt(1-r_other,t^2) g, with g the
        partial correlation, g^2 ~ Beta(1/2, (N-3)/2) and a random sign."""
        other = r1t if test == 2 else r2t
        k = len(r12)
        n = np.maximum(np.asarray(n_samples, dtype=np.float64).reshape(-1, 1), 4.0)
        g = np.sqrt(self.rng.beta(0.5, (n - 3.0) / 2.0, size=(k, n_null)))
        g *= self.rng.choice([-1.0, 1.0], size=(k, n_null))
        new = (r12 * other)[:, None] + (np.sqrt(np.maximum(1 - r12 ** 2, 0)) * np.sqrt(np.maximum(1 - other ** 2, 0)))[:, None] * g
        rr = np.broadcast_to(r12[:, None], new.shape)
        ob = np.broadcast_to(other[:, None], new.shape)
        return (rr, ob, new) if test == 2 else (rr, new, ob)

    def predict_batch(self, Cs, n_samples, forcenn=True, n_null=None, p_value_thresh=None,
                      null="atom", max_rows=1_000_000):
        """Cs: (k, 3, 3) covariance/correlation matrices ordered [x1, x2, target].
        n_samples: scalar or (k,) effective sample sizes (only used for the null).
        null="atom" (default): each atom is tested against a null under which it is 0
          (R: either source unrelated to T; U1/U2: T independent of that source given the
          other; S: either source adds nothing given the other). JMI: global null.
        null="global": every atom against "T unrelated to both sources"."""
        if null not in ("atom", "global"):
            raise ValueError("null must be 'atom' or 'global'")
        Cs = np.asarray(Cs, dtype=np.float64).reshape(-1, 3, 3)
        d = np.sqrt(np.einsum("kii->ki", Cs))
        R = Cs / (d[:, :, None] * d[:, None, :])
        r12, r1t, r2t = R[:, 0, 1], R[:, 0, 2], R[:, 1, 2]
        atoms, R_raw, I1, I2, JMI = self.atoms_from_correlations(r12, r1t, r2t, forcenn)
        out = {"atoms": atoms, "atoms_raw": assemble_atoms(R_raw, I1, I2, JMI, False),
               "mi_source1": I1, "mi_source2": I2, "jmi": JMI}

        n_null = self.n_null if n_null is None else n_null
        thr = self.p_value_thresh if p_value_thresh is None else p_value_thresh
        if n_null and n_null > 0:
            k = len(r12)
            n_samples = np.broadcast_to(np.asarray(n_samples, dtype=np.float64), (k,))
            p = np.empty((k, 4)); p_jmi = np.empty(k)
            mu = np.empty((k, 4)); sd = np.empty((k, 4))
            step = max(1, max_rows // (n_null * (4 if null == "atom" else 1)))
            for i in range(0, k, step):
                sl = slice(i, min(k, i + step))
                ns = n_samples[sl]
                rr, a, b = self.null_correlations(r12[sl], ns, n_null)     # global null (JMI)
                null_jmi = closed_form_mi(rr, a, b)[2]
                p_jmi[sl] = (1 + np.sum(null_jmi >= JMI[sl][:, None] - 1e-15, axis=1)) / (n_null + 1)
                if null == "global":
                    null_atoms = self.atoms_from_correlations(rr, a, b, forcenn)[0]
                    p[sl] = (1 + np.sum(null_atoms >= atoms[sl][:, None, :] - 1e-15, axis=1)) / (n_null + 1)
                    mu[sl], sd[sl] = null_atoms.mean(1), null_atoms.std(1)
                else:
                    c = (r12[sl], r1t[sl], r2t[sl])
                    ens = [self.marginal_null(*c, 1, ns, n_null), self.marginal_null(*c, 2, ns, n_null),
                           self.conditional_null(*c, 1, ns, n_null), self.conditional_null(*c, 2, ns, n_null)]
                    ens = [self.atoms_from_correlations(*e, forcenn)[0] for e in ens]
                    p[sl], mu[sl], sd[sl] = atom_pvalues(atoms[sl], *ens, n_null)
            out.update(null_mu=mu, null_sigma=sd, p_values=p, p_jmi=p_jmi,
                       significant_atoms=np.where(p < thr, atoms, 0.0))
        return out

    def predict_from_covariance(self, C, n_samples, **kw):
        res = self.predict_batch(np.asarray(C)[None], n_samples, **kw)
        return {k: v[0] for k, v in res.items()}


# ============================================================
# 6. EVALUATION
# ============================================================

def evaluate(est, test_rows, out_dir):
    report = {}
    names = ["R", "U1", "U2", "S"]
    r12, r1t, r2t, R_t, se_t, I1, I2, JMI = test_rows[:, :8].T

    # 6a. Test set vs teacher, binned by the strongest source-target correlation.
    pred, R_raw_p, *_ = est.atoms_from_correlations(r12, r1t, r2t, forcenn=True)
    teach = assemble_atoms(R_t, I1, I2, JMI, True)
    strength = np.maximum(np.abs(r1t), np.abs(r2t))
    bins = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    print("\nTest set: network vs single teacher run (bits). 'excess' removes the teacher's own MC noise.")
    print(f"{'max|r_t| bin':>14} {'n':>6} {'mean JMI':>10} {'teacher se(R)':>13} "
          + " ".join(f"{'MAE ' + n:>10}" for n in names) + f" {'excess RMSE R':>14}")
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (strength > lo) & (strength <= hi)
        if m.sum() == 0:
            continue
        mae = np.abs(pred[m] - teach[m]).mean(0)
        mse_raw = np.mean((R_raw_p[m] - R_t[m]) ** 2)
        excess = math.sqrt(max(mse_raw - np.mean(se_t[m] ** 2), 0.0))
        print(f"{f'({lo}, {hi}]':>14} {m.sum():6d} {JMI[m].mean():10.2e} {se_t[m].mean():13.2e} "
              + " ".join(f"{v:10.2e}" for v in mae) + f" {excess:14.2e}")
        rows.append(dict(bin=f"({lo},{hi}]", n=int(m.sum()), mean_jmi=float(JMI[m].mean()),
                         teacher_se=float(se_t[m].mean()), excess_rmse_R=excess,
                         **{f"mae_{n}": float(v) for n, v in zip(names, mae)}))
    report["test_by_strength"] = rows

    # 6b. Unrelated target: every atom must be exactly 0.
    for r12_0 in (0.0, 0.3, 0.7, 0.95):
        z = est.predict_from_covariance(np.array([[1, r12_0, 0], [r12_0, 1, 0], [0, 0, 1.0]]), 900, n_null=0)
        assert np.all(z["atoms"] == 0), z["atoms"]
    print("\nUnrelated target -> all atoms exactly 0: OK")

    # 6c. High-precision reference on weak targets: network vs one pid_mvn-sized exact run.
    rng = np.random.default_rng(123)
    ref_rows = []
    for _ in range(150):
        r = weak_example(rng)
        if max(abs(r[1]), abs(r[2])) > 0.3:
            continue
        I1_, I2_, J_ = closed_form_mi(*r)
        ref = np.mean([iccs_redundancy(*r, 200_000, rng)[0] for _ in range(10)])   # 2M samples
        single = iccs_redundancy(*r, 100_000, rng)[0]                                # like pid_mvn
        net = est._raw_redundancy(*(np.array([v]) for v in r), np.array([J_]))[0]
        ref_rows.append((J_, ref, single, net))
    ref_rows = np.array(ref_rows)
    e_single = np.abs(ref_rows[:, 2] - ref_rows[:, 1])
    e_net = np.abs(ref_rows[:, 3] - ref_rows[:, 1])
    print(f"\nWeak targets (max|r_t| <= 0.3, n={len(ref_rows)}), raw redundancy vs a 2M-sample reference:")
    print(f"  one exact run (100k samples, as pid_mvn): MAE {e_single.mean():.2e} bits, "
          f"median relative to JMI {np.median(e_single / ref_rows[:, 0]):.3f}")
    print(f"  network:                                  MAE {e_net.mean():.2e} bits, "
          f"median relative to JMI {np.median(e_net / ref_rows[:, 0]):.3f}")
    report["weak_reference"] = dict(n=len(ref_rows), mae_single_exact=float(e_single.mean()),
                                    mae_network=float(e_net.mean()))

    # 6d. Cross-check of the teacher against pid_mvn itself.
    if pid_mvn is not None:
        diffs, sds = [], []
        for _ in range(10):
            r = uniform_example(rng)
            C = np.array([[1, r[0], r[1]], [r[0], 1, r[2]], [r[1], r[2], 1.0]])
            runs = np.array([pid_mvn.calc_pi_mvn(pid_mvn.Lattice2D(), C, [1, 1, 1], pid_mvn.Iccs_mvn,
                                                 forcenn=True).PI for _ in range(3)])
            Rr = np.mean([iccs_redundancy(*r, 200_000, rng)[0] for _ in range(5)])
            ours = assemble_atoms(Rr, *closed_form_mi(*r), True)
            diffs.append(np.abs(runs.mean(0) - ours).max()); sds.append(runs.std(0).max())
        print(f"\npid_mvn cross-check: max |pid_mvn mean - teacher| = {max(diffs):.2e} bits "
              f"(pid_mvn run-to-run SD up to {max(sds):.2e})")
        report["pid_mvn_max_diff"] = float(max(diffs))

    # 6e. Null calibration: independent targets, false-positive rate at alpha = 0.05.
    print("\nNull calibration (independent target, 300 datasets each): false-positive rate at 0.05")
    print(f"{'N':>6} {'r12':>5} {'target':>12}  " + " ".join(f"{n:>6}" for n in names + ["JMI"]))
    cal = []
    for N in (30, 300, 3000):
        for r12_0 in (0.0, 0.5, 0.9):
            for tgt, prev in (("gaussian", None), ("binary", 0.5), ("binary", 0.1)):
                Cs = []
                for _ in range(300):
                    X = rng.standard_normal((N, 2)) @ np.linalg.cholesky(np.array([[1, r12_0], [r12_0, 1.0]])).T
                    if tgt == "gaussian":
                        y = rng.standard_normal(N); Cs.append(correlation_from_data(X[:, 0], X[:, 1], y, "continuous"))
                    else:
                        y = (rng.random(N) < prev).astype(int)
                        if y.min() == y.max():
                            y[0] = 1 - y[0]
                        Cs.append(correlation_from_data(X[:, 0], X[:, 1], y, "binary"))
                res = est.predict_batch(np.array(Cs), N, n_null=500)
                fpr = list((res["p_values"] < 0.05).mean(0)) + [float((res["p_jmi"] < 0.05).mean())]
                label = tgt if prev is None else f"binary p={prev}"
                print(f"{N:6d} {r12_0:5.1f} {label:>12}  " + " ".join(f"{v:6.3f}" for v in fpr))
                cal.append(dict(N=N, r12=r12_0, target=label, **{f"fpr_{n}": float(v) for n, v in zip(names + ["JMI"], fpr)}))
    report["null_calibration"] = cal

    with open(Path(out_dir) / "evaluation.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def save_package(model, out_dir):
    out_dir = Path(out_dir)
    torch.save(model.state_dict(), out_dir / "redundancy_net.pt")
    # NumPy copy of the weights for torch-free inference (neural_pid.py).
    linears = [m for m in model.mlp if isinstance(m, nn.Linear)]
    np.savez(out_dir / "redundancy_net.npz",
             **{f"W{i}": m.weight.detach().cpu().numpy() for i, m in enumerate(linears)},
             **{f"b{i}": m.bias.detach().cpu().numpy() for i, m in enumerate(linears)})
    meta = {
        "network_input": "pid_features(r12, r1t, r2t): [2*phi-1, sign-aligned r12, s, 2*R^2_multiple-1]",
        "network_output": "rho = raw CCS redundancy / I(X1,X2;T)",
        "atoms": ["redundancy", "unique_source1", "unique_source2", "synergy"],
        "lattice": "Williams-Beer as pid_mvn.calc_pi_mvn; forcenn clips R, U1, U2 (not S)",
        "binary_target": "0/1 indicator (no copula normalisation)",
        "continuous_variables": "gcmi.copnorm (sources, and continuous targets)",
        "null": "Monte Carlo permutation null of (r1t, r2t) given r12 and N, through the network",
        "teacher_nmc": CONFIG["teacher_nmc"],
        "hidden_dim": CONFIG["hidden_dim"],
        "n_hidden_layers": CONFIG["n_hidden_layers"],
        "precision": "float64",
    }
    (out_dir / "model_metadata.json").write_text(json.dumps(meta, indent=2))


# ============================================================
# 7. RUN
# ============================================================

if __name__ == "__main__":
    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(CONFIG["seed"])
    torch.set_default_dtype(torch.float64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, "| joblib:", HAVE_JOBLIB, "| pid_mvn cross-check:", pid_mvn is not None)

    train_rows = generate_dataset("train", CONFIG["n_train"], 0)
    val_rows = generate_dataset("validation", CONFIG["n_val"], 10_000_000)
    test_rows = generate_dataset("test", CONFIG["n_test"], 20_000_000)

    model = RedundancyNet(CONFIG["hidden_dim"], CONFIG["n_hidden_layers"]).to(device=device, dtype=torch.float64)
    train_model(model, training_arrays(train_rows), training_arrays(val_rows), device)
    save_package(model, out_dir)

    pid_estimator = NeuralPIDEstimator(model, device, n_null=CONFIG["n_null"],
                                       p_value_thresh=CONFIG["p_value_thresh"])
    evaluate(pid_estimator, test_rows, out_dir)
    print("\nSaved model package to", out_dir)
