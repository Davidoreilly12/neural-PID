# ============================================================
# NEURAL PID [2,2,1]: AMORTISED EXACT GAUSSIAN PID FOR TWO 2-VARIABLE SOURCES AND ONE TARGET
#
# Same design as the [1,1,1] Neural PID:
#   - The network learns ONLY the Monte Carlo CCS redundancy (Iccs_mvn with varsizes
#     [2,2,1]) as a fraction of the joint MI. I(X1;T), I(X2;T), I(X1,X2;T) are exact
#     closed forms and the lattice is applied as calc_pi_mvn does (forcenn clips R, U1, U2,
#     not S). All atoms are in bits and sum exactly to I(X1,X2;T).
#   - The teacher is the CORRECTED Iccs_mvn: the pid_mvn Python port filled covariance
#     blocks with Cfull[idx, idx] (NumPy pairs the indices -> only the diagonal), which is
#     wrong for multivariate sources. Here every block is the full sub-matrix.
#
# Canonical input (exact, verified): Iccs is invariant to any invertible linear map within
# a source, so each source is whitened and rotated so that its target vector is (rho_k, 0).
# A 5x5 covariance then reduces to 6 numbers: rho1, rho2 (each source's canonical
# correlation with T) and the 2x2 whitened cross-source block B, with signs fixed so that
# B[1,0] >= 0 and B[0,1] >= 0.
#
# Significance: Monte Carlo permutation null at inference. With the source blocks fixed,
# the whitened target cross-correlations of a permuted target are a uniformly oriented
# vector with R^2 ~ Beta(2, (N-5)/2). They are passed through the same network.
#
# Sources: gcmi.copnorm on every variable. Binary targets: 0/1 indicator (NOT
# copula-normalised). Continuous targets: gcmi.copnorm.
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
import scipy.special as sps        # used as sps.ndtri (a bare ufunc import does not pickle into joblib workers)
import scipy.linalg as sla

try:
    from joblib import Parallel, delayed
    HAVE_JOBLIB = True
except ImportError:
    HAVE_JOBLIB = False

sys.path.insert(0, "/content/PyPID_mvn")
try:
    import iccs_mvn_fixed          # corrected pid_mvn port; only used to cross-check the teacher
except ImportError:
    iccs_mvn_fixed = None

LN2 = math.log(2.0)

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    "output_dir": "/content/drive/MyDrive/neural_pid_221_output",
    "n_train": 100_000,
    "n_val": 10_000,
    "n_test": 10_000,
    "teacher_nmc": 200_000,         # Monte Carlo samples per teacher example (pid_mvn uses 100,000)
    "mix": (0.3, 0.5, 0.2),         # (uniform, weak-target, from simulated data)
    "s_min": 1e-3,                  # smallest target strength hypot(rho1, rho2) sampled
    "min_det": 1e-8,                # reject near-singular covariance matrices
    "data_n_min": 20,
    "data_n_max": 20_000,
    # Network.
    "hidden_dim": 128,
    "n_hidden_layers": 4,
    "batch_size": 1024,
    "epochs": 300,
    "learning_rate": 1e-3,
    "weight_decay": 1e-6,
    "model_error_floor": 0.05,      # floor on the per-example noise SD (R/JMI units) in the loss weights
    "checkpoint_every": 10,
    # Generation.
    "n_jobs": -1,
    "chunk": 100,
    "seed": 42,
    # Inference defaults.
    "n_null": 1000,
    "p_value_thresh": 0.05,
}

N_FEATURES = 9
SWAP_PERM = [0, 1, 2, 3, 5, 4, 6, 7, 8]          # source swap: B01 <-> B10 ...
SWAP_SIGN = [-1, 1, 1, 1, 1, 1, 1, 1, 1]          # ... and the direction feature changes sign


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 1. CANONICAL FORM, CLOSED-FORM MI AND THE EXACT CCS REDUNDANCY (TEACHER)
# ============================================================

def whitened_blocks(Cs):
    """(k,5,5) covariances ordered [x1a, x1b, x2a, x2b, t] -> whitened cross block Bw (k,2,2)
    and whitened target vectors a1, a2 (k,2). Any whitening works (exact invariance)."""
    Cs = np.asarray(Cs, dtype=np.float64).reshape(-1, 5, 5)
    d = np.sqrt(np.einsum("kii->ki", Cs))
    R = Cs / (d[:, :, None] * d[:, None, :])
    try:
        W1 = np.linalg.inv(np.linalg.cholesky(R[:, 0:2, 0:2]))
        W2 = np.linalg.inv(np.linalg.cholesky(R[:, 2:4, 2:4]))
    except np.linalg.LinAlgError as e:
        raise ValueError("A source's 2x2 covariance block is singular (its two variables are collinear).") from e
    a1 = np.einsum("kij,kj->ki", W1, R[:, 0:2, 4])
    a2 = np.einsum("kij,kj->ki", W2, R[:, 2:4, 4])
    Bw = W1 @ R[:, 0:2, 2:4] @ np.swapaxes(W2, 1, 2)
    return Bw, a1, a2


def canonical_from_whitened(Bw, a1, a2):
    """Rotate each whitened source so its target vector is (rho, 0); fix the second-axis signs
    so that B[1,0] >= 0 and B[0,1] >= 0. Returns (..., 6) = [rho1, rho2, B00, B01, B10, B11]."""
    r1, r2 = np.hypot(a1[..., 0], a1[..., 1]), np.hypot(a2[..., 0], a2[..., 1])
    u1 = np.where(r1 > 0, a1[..., 0] / np.where(r1 > 0, r1, 1), 1.0)
    v1 = np.where(r1 > 0, a1[..., 1] / np.where(r1 > 0, r1, 1), 0.0)
    u2 = np.where(r2 > 0, a2[..., 0] / np.where(r2 > 0, r2, 1), 1.0)
    v2 = np.where(r2 > 0, a2[..., 1] / np.where(r2 > 0, r2, 1), 0.0)
    b00, b01, b10, b11 = Bw[..., 0, 0], Bw[..., 0, 1], Bw[..., 1, 0], Bw[..., 1, 1]
    # B = Q1^T Bw Q2 with Q = [[u, -v], [v, u]]
    t00, t01 = u1 * b00 + v1 * b10, u1 * b01 + v1 * b11
    t10, t11 = -v1 * b00 + u1 * b10, -v1 * b01 + u1 * b11
    B00, B01 = t00 * u2 + t01 * v2, -t00 * v2 + t01 * u2
    B10, B11 = t10 * u2 + t11 * v2, -t10 * v2 + t11 * u2
    s1 = np.where(B10 < 0, -1.0, 1.0)              # flip source 1's second axis
    s2 = np.where(B01 < 0, -1.0, 1.0)              # flip source 2's second axis
    return np.stack([r1, r2, B00, B01 * s2, B10 * s1, B11 * s1 * s2], axis=-1)


def canonical_from_cov(Cs):
    return canonical_from_whitened(*whitened_blocks(Cs))


def cov_from_canonical(p):
    r1, r2, B00, B01, B10, B11 = p
    return np.array([[1, 0, B00, B01, r1],
                     [0, 1, B10, B11, 0],
                     [B00, B10, 1, 0, r2],
                     [B01, B11, 0, 1, 0],
                     [r1, 0, r2, 0, 1.0]])


def _cross_terms(p):
    r1, r2, B00, B01, B10, B11 = np.moveaxis(np.asarray(p, dtype=np.float64), -1, 0)
    # S = I - B^T B  (det S = det of the 4x4 source covariance)
    M00 = 1.0 - (B00 ** 2 + B10 ** 2)
    M11 = 1.0 - (B01 ** 2 + B11 ** 2)
    M01 = -(B00 * B01 + B10 * B11)
    det_s = M00 * M11 - M01 ** 2
    # R^2 = a'a + (b - B'a)' S^-1 (b - B'a), a = (r1, 0), b = (r2, 0)
    d0, d1 = r2 - B00 * r1, -B01 * r1
    q = (M11 * d0 ** 2 - 2 * M01 * d0 * d1 + M00 * d1 ** 2) / np.where(det_s > 0, det_s, np.inf)
    rsq = r1 ** 2 + q
    return r1, r2, rsq, det_s


def closed_form_mi(p):
    """I(X1;T), I(X2;T), I(X1,X2;T) in bits from canonical parameters (vectorised)."""
    r1, r2, rsq, _ = _cross_terms(p)
    I1 = -0.5 * np.log1p(-r1 ** 2) / LN2
    I2 = -0.5 * np.log1p(-r2 ** 2) / LN2
    JMI = -0.5 * np.log1p(-np.clip(rsq, 0.0, 1.0 - 1e-15)) / LN2
    return I1, I2, JMI


def valid_canonical(p):
    r1, r2, rsq, det_s = _cross_terms(p)
    return (det_s > CONFIG["min_det"]) & (det_s * (1.0 - rsq) > CONFIG["min_det"]) & (r1 < 1) & (r2 < 1)


def _log_ratio(a, t, Caa, cat):
    """log p(a | t) - log p(a) for Gaussian a (n x k) and unit-variance target t (n,)."""
    La = np.linalg.cholesky(Caa)
    Lc = np.linalg.cholesky(Caa - np.outer(cat, cat))
    za = sla.solve_triangular(La, a.T, lower=True, check_finite=False)
    zc = sla.solve_triangular(Lc, (a - t[:, None] * cat).T, lower=True, check_finite=False)
    return (-0.5 * (zc * zc).sum(0) + 0.5 * (za * za).sum(0)
            - np.log(np.diag(Lc)).sum() + np.log(np.diag(La)).sum())


def iccs_221_cov(C, nmc, rng, z=None):
    """Raw CCS redundancy (bits) and Monte Carlo SE for a 5x5 covariance, varsizes [2,2,1].
    Same estimator as Iccs_mvn (local co-information kept where sign(dhx) = sign(dhy) =
    sign(dhxy) = sign(lnii), averaged over all samples), with full covariance blocks."""
    C = np.asarray(C, dtype=np.float64)
    d = np.sqrt(np.diag(C))
    if z is None:
        z = rng.standard_normal((nmc, 5)) @ np.linalg.cholesky(C).T
    z = z / d                                   # correlation scale (exactly invariant)
    Rm = C / np.outer(d, d)
    t = z[:, 4]
    f = lambda idx: _log_ratio(z[:, idx], t, Rm[np.ix_(idx, idx)], Rm[idx, 4])
    dhx, dhy, dhxy = f([0, 1]), f([2, 3]), f([0, 1, 2, 3])
    lnii = dhx + dhy - dhxy
    sx = np.sign(dhx)
    keep = (sx == np.sign(lnii)) & (sx == np.sign(dhy)) & (sx == np.sign(dhxy))
    lnii = np.where(keep, lnii, 0.0)
    return lnii.mean() / LN2, lnii.std(ddof=1) / math.sqrt(len(lnii)) / LN2


def iccs_redundancy(p, nmc, rng, z=None):
    """Teacher on canonical parameters (same estimator as iccs_221_cov, ~5x faster).
    In canonical coordinates each source's target vector is (rho, 0) with identity
    covariance, so dhx and dhy only involve the first axis. Exactly 0 when either
    source is unrelated to T."""
    r1, r2 = float(p[0]), float(p[1])
    if r1 == 0.0 or r2 == 0.0:
        return 0.0, 0.0                       # dhx (or dhy) is identically 0
    C = cov_from_canonical(p)
    if z is None:
        z = rng.standard_normal((nmc, 5)) @ np.linalg.cholesky(C).T
    x, t = z[:, :4], z[:, 4]
    v1, v2 = 1.0 - r1 * r1, 1.0 - r2 * r2
    dhx = -0.5 * math.log(v1) - (x[:, 0] - r1 * t) ** 2 / (2 * v1) + 0.5 * x[:, 0] ** 2
    dhy = -0.5 * math.log(v2) - (x[:, 2] - r2 * t) ** 2 / (2 * v2) + 0.5 * x[:, 2] ** 2
    Cxx, c = C[:4, :4], C[:4, 4]
    Ccond = Cxx - np.outer(c, c)
    e = x - t[:, None] * c
    dhxy = (-0.5 * (np.linalg.slogdet(Ccond)[1] - np.linalg.slogdet(Cxx)[1])
            - 0.5 * np.einsum("ni,ij,nj->n", e, np.linalg.inv(Ccond), e, optimize=True)
            + 0.5 * np.einsum("ni,ij,nj->n", x, np.linalg.inv(Cxx), x, optimize=True))
    lnii = dhx + dhy - dhxy
    sx = np.sign(dhx)
    keep = (sx == np.sign(lnii)) & (sx == np.sign(dhy)) & (sx == np.sign(dhxy))
    lnii = np.where(keep, lnii, 0.0)
    return lnii.mean() / LN2, lnii.std(ddof=1) / math.sqrt(len(lnii)) / LN2


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
# 2. TURNING DATA INTO A COVARIANCE MATRIX (USE THIS IN YOUR PIPELINES)
# ============================================================

def copnorm_cols(X):
    """gcmi copula normalisation of each column of X (N x k)."""
    X = np.asarray(X, dtype=np.float64).reshape(len(X), -1)
    return np.asarray(gcmi.copnorm(X.T), dtype=np.float64).T


def binary_indicator(y):
    """Binary target as a 0/1 indicator (NOT copula-normalised)."""
    y = np.asarray(y)
    levels = np.unique(y)
    if len(levels) != 2:
        raise ValueError(f"Binary target needs exactly 2 levels, got {len(levels)}.")
    return (y == levels[1]).astype(np.float64)


def correlation_from_data(X1, X2, y, target="binary"):
    """5x5 correlation matrix [X1[:,0], X1[:,1], X2[:,0], X2[:,1], target]: every source
    variable gcmi-copnormed, binary target as a 0/1 indicator, continuous target copnormed."""
    Z = copnorm_cols(np.column_stack([X1, X2]))
    if Z.shape[1] != 4:
        raise ValueError("X1 and X2 must each have 2 columns.")
    zt = binary_indicator(y) if target == "binary" else copnorm_cols(y)[:, 0]
    return np.corrcoef(np.column_stack([Z, zt]), rowvar=False)


# ============================================================
# 3. TRAINING DISTRIBUTION
# ============================================================

def _rotation(rng):
    th = rng.uniform(0, 2 * np.pi)
    Q = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    if rng.random() < 0.5:
        Q[:, 1] *= -1
    return Q


def vine_corr(d, rng, lim=0.99):
    """Random d x d correlation matrix from partial correlations drawn uniformly in
    (-lim, lim) (C-vine; Lewandowski, Kurowicka & Joe 2009). Always valid, and every
    pairwise correlation can reach +-lim. Variable order is shuffled."""
    P = np.zeros((d, d))
    S = np.eye(d)
    for k in range(d - 1):
        for i in range(k + 1, d):
            P[k, i] = rng.uniform(-lim, lim)
            r = P[k, i]
            for l in range(k - 1, -1, -1):
                r = r * math.sqrt((1 - P[l, i] ** 2) * (1 - P[l, k] ** 2)) + P[l, i] * P[l, k]
            S[k, i] = S[i, k] = r
    perm = rng.permutation(d)
    return S[np.ix_(perm, perm)]


def _random_corr5(rng):
    if rng.random() < 0.5:
        return vine_corr(5, rng)
    A = rng.standard_normal((5, int(rng.integers(5, 11))))
    return A @ A.T


def uniform_example(rng):
    """Random 5x5 covariance: vine (full range of every correlation) or A A^T."""
    while True:
        p = canonical_from_cov(_random_corr5(rng)[None])[0]
        if valid_canonical(p):
            return p


def weak_example(rng):
    """Canonical parameters drawn directly: cross block with singular values (often near 1),
    target strength s log-uniform down to s_min, extra density near the axes."""
    while True:
        sv = rng.uniform(0, 1, 2)
        near = rng.random(2) < 0.2
        sv = np.where(near, 1 - 10 ** rng.uniform(-4, -1, 2), sv)
        B = _rotation(rng) @ np.diag(sv) @ _rotation(rng).T
        u = rng.random()
        if u < 0.7:
            th = rng.uniform(0, np.pi / 2)
        else:
            small = 10 ** rng.uniform(-4, math.log10(0.2))
            th = small if u < 0.85 else np.pi / 2 - small
        s = 10 ** rng.uniform(math.log10(CONFIG["s_min"]), 0)
        a1 = np.array([s * math.cos(th), 0.0]); a2 = np.array([s * math.sin(th), 0.0])
        p = canonical_from_whitened(B, a1, a2)
        if valid_canonical(p):
            return p


def _synthetic_signal(sub, rng, freqs, t):
    phase = rng.uniform(0, 2 * np.pi, len(freqs))
    amp = rng.uniform(0.5, 1.0, len(freqs))
    sig = (amp[:, None] * np.sin(2 * np.pi * freqs[:, None] * t[None, :] + phase[:, None])).sum(0)
    sig -= sig.mean()
    sd = sig.std()
    return sig / sd if sd > 0 else sig


def simulate_emg_221(n, rng):
    """EMG simulator with two channels per source (the [1,1,1] simulator extended to four
    channels). Signal correlations: one channel's envelope is gated by the binary task y1;
    each other channel is gated by a copy of y1 that agrees with it with signal correlation
    drawn uniformly in (-0.99, 0.99). Noise correlations: a 4x4 matrix whose partial
    correlations are drawn uniformly in (-0.99, 0.99). Channel order is shuffled."""
    sig_corr = np.concatenate([[1.0], rng.uniform(-0.99, 0.99, 3)])
    trials = max(1, min(100, n // 100))
    sub = max(10, n // trials)
    freqs = np.arange(1, 151, dtype=np.float64)
    t = np.arange(sub, dtype=np.float64) / sub
    e = np.stack([np.concatenate([_synthetic_signal(sub, rng, freqs, t) for _ in range(trials)])
                  for _ in range(4)], axis=1)
    noise = rng.standard_normal((sub * trials, 4)) @ np.linalg.cholesky(vine_corr(4, rng)).T
    y1 = np.tile(np.concatenate([np.zeros(sub), np.ones(sub)]), trials)[: sub * trials]
    gates = []
    for sc in sig_corr:
        agree = rng.random(len(y1)) < abs(sc)
        gates.append(np.where(agree, y1, 1 - y1) if sc >= 0 else np.where(agree, 1 - y1, y1))
    X = e * np.stack(gates, axis=1)[:, rng.permutation(4)] + noise
    return X[:n, :2], X[:n, 2:], y1[:n]


def data_example(rng):
    """Covariance computed from simulated data at finite N (sampling noise included)."""
    while True:
        n = int(10 ** rng.uniform(math.log10(CONFIG["data_n_min"]), math.log10(CONFIG["data_n_max"])))
        kind = rng.integers(3)
        if kind == 0:
            X1, X2, y = simulate_emg_221(n, rng)
            target = "binary"
        else:
            C = _random_corr5(rng)
            d = np.sqrt(np.diag(C)); C = C / np.outer(d, d)
            k = 10 ** rng.uniform(-2, 0)                  # weaken the target
            C[:4, 4] *= k; C[4, :4] *= k
            Z = rng.standard_normal((n, 5)) @ np.linalg.cholesky(C).T
            X1, X2 = Z[:, :2], Z[:, 2:4]
            if kind == 1:                                  # binary target, random prevalence
                prev = 10 ** rng.uniform(-2, math.log10(0.5))
                y = (Z[:, 4] > sps.ndtri(1 - prev)).astype(np.int8)
                target = "binary"
            else:
                y = Z[:, 4]
                target = "continuous"
        if target == "binary" and min(np.sum(y == 0), np.sum(y != 0)) < 2:
            continue
        C = correlation_from_data(X1, X2, y, target)
        if not np.all(np.isfinite(C)):
            continue
        try:
            p = canonical_from_cov(C[None])[0]
        except ValueError:
            continue
        if valid_canonical(p):
            return p


GENERATORS = (uniform_example, weak_example, data_example)
N_COLS = 12      # rho1, rho2, B00, B01, B10, B11, R_raw, R_se, I1, I2, JMI, generator id


def generate_chunk(n, seed, nmc, mix):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, N_COLS))
    cum = np.cumsum(mix)
    for i in range(n):
        kind = int(np.searchsorted(cum, rng.random() * cum[-1], side="right"))
        p = GENERATORS[kind](rng)
        R, se = iccs_redundancy(p, nmc, rng)
        I1, I2, JMI = closed_form_mi(p)
        out[i] = (*p, R, se, I1, I2, JMI, kind)
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
    rows = np.load(partial)["rows"] if partial.exists() else np.zeros((0, N_COLS))
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
        print(f"\r  {name}: {len(rows):,}/{n_examples:,} ({rate:,.1f}/s)", end="")
    print()
    np.savez_compressed(final, rows=rows)
    if partial.exists():
        partial.unlink()
    return rows


# ============================================================
# 4. FEATURES AND NETWORK
# ============================================================

def pid_features(p):
    """[direction of (rho1, rho2) in [-1,1], strength s, 2*R^2-1, B00, B01, B10, B11,
    singular values of B]. Swapping the sources negates column 0 and swaps B01 <-> B10."""
    p = np.asarray(p, dtype=np.float64).reshape(-1, 6)
    r1, r2, rsq, _ = _cross_terms(p)
    s = np.hypot(r1, r2)
    phi = np.where(s > 0, np.arctan2(r2, r1) / (np.pi / 2), 0.5)
    B = p[:, 2:]
    fro = (B ** 2).sum(1)
    det = B[:, 0] * B[:, 3] - B[:, 1] * B[:, 2]
    disc = np.sqrt(np.maximum(fro ** 2 - 4 * det ** 2, 0.0))
    sv1 = np.sqrt(np.maximum((fro + disc) / 2, 0.0))
    sv2 = np.sqrt(np.maximum((fro - disc) / 2, 0.0))
    return np.column_stack([2 * phi - 1, s, 2 * np.clip(rsq, 0, 1) - 1, B, sv1, sv2])


class RedundancyNet(nn.Module):
    """Predicts rho = R_raw / JMI. Exactly symmetric in the two sources."""

    def __init__(self, hidden_dim=128, n_hidden_layers=4):
        super().__init__()
        layers = [nn.Linear(N_FEATURES, hidden_dim), nn.GELU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)
        self.register_buffer("perm", torch.tensor(SWAP_PERM))
        self.register_buffer("sign", torch.tensor(SWAP_SIGN, dtype=torch.float64))

    def forward(self, f):
        g = f[:, self.perm] * self.sign.to(f.dtype)
        return 0.5 * (self.mlp(f) + self.mlp(g)).squeeze(-1)


class TorchRho:
    """Adapter so the estimator can call the torch model on NumPy features."""

    def __init__(self, model, device):
        self.model, self.device = model.eval(), device

    def __call__(self, F, batch=262_144):
        out = np.empty(len(F))
        with torch.no_grad():
            for i in range(0, len(F), batch):
                x = torch.tensor(F[i:i + batch], dtype=torch.float64, device=self.device)
                out[i:i + batch] = self.model(x).cpu().numpy()
        return out


def training_arrays(rows):
    p, R, se, JMI = rows[:, :6], rows[:, 6], rows[:, 7], rows[:, 10]
    ok = (JMI > 1e-12) & (p[:, 0] > 0) & (p[:, 1] > 0) & np.isfinite(R) & np.isfinite(se)
    F = pid_features(p[ok])
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
# 5. ESTIMATOR (same code as neural_pid_221.py; the model is any callable F -> rho)
# ============================================================

class NeuralPID221Estimator:
    """predict_from_covariance(C, n_samples) for one 5x5 matrix ordered
    [x1a, x1b, x2a, x2b, target]; predict_batch(Cs, n_samples) for many at once."""

    def __init__(self, model, n_null=1000, p_value_thresh=0.05, seed=0):
        self.model = model
        self.n_null = n_null
        self.p_value_thresh = p_value_thresh
        self.rng = np.random.default_rng(seed)

    def _raw_redundancy(self, p, JMI):
        p = np.asarray(p, dtype=np.float64)
        flat = p.reshape(-1, 6)
        rho = self.model(pid_features(flat)) if len(flat) else np.empty(0)
        rho = rho.reshape(np.shape(JMI))
        zero = (p[..., 0] == 0) | (p[..., 1] == 0) | (JMI <= 0)
        return np.where(zero, 0.0, rho * JMI)

    def atoms_from_canonical(self, p, forcenn=True):
        """Returns (atoms [...,4], raw redundancy, I(X1;T), I(X2;T), I(X1,X2;T))."""
        I1, I2, JMI = closed_form_mi(p)
        R_raw = self._raw_redundancy(p, JMI)
        return assemble_atoms(R_raw, I1, I2, JMI, forcenn), R_raw, I1, I2, JMI

    def null_canonical(self, Bw, n_samples, n_null):
        """Permutation null: whitened target cross-correlations c = G u sqrt(R^2), with
        G G^T the observed 4x4 whitened source covariance, u uniform on the sphere and
        R^2 ~ Beta(2, (N-5)/2) (exact for Gaussian sources, any independent target)."""
        k = len(Bw)
        Cxx = np.zeros((k, 4, 4))
        Cxx[:, 0, 0] = Cxx[:, 1, 1] = Cxx[:, 2, 2] = Cxx[:, 3, 3] = 1.0
        Cxx[:, :2, 2:] = Bw
        Cxx[:, 2:, :2] = np.swapaxes(Bw, 1, 2)
        G = np.linalg.cholesky(Cxx)
        u = self.rng.standard_normal((k, n_null, 4))
        u /= np.linalg.norm(u, axis=-1, keepdims=True)
        n = np.maximum(np.asarray(n_samples, dtype=np.float64).reshape(-1, 1), 6.0)
        rsq = self.rng.beta(2.0, (n - 5.0) / 2.0, size=(k, n_null))
        c = np.einsum("kij,knj->kni", G, u) * np.sqrt(rsq)[..., None]
        return canonical_from_whitened(np.broadcast_to(Bw[:, None], (k, n_null, 2, 2)), c[..., :2], c[..., 2:])

    def marginal_null(self, Bw, a1, a2, zero, n_samples, n_null):
        """H0: source `zero` (1 or 2) has no correlation with T, with the other source's
        relation to T and the source blocks at their observed values (the other source's
        target vector is shrunk only if needed to keep the matrix positive definite).
        Parametric bootstrap: sample correlation matrices from Wishart(N-1, Sigma0)."""
        k = len(Bw)
        keep = a1 if zero == 2 else a2
        M = np.eye(2) - (Bw @ np.swapaxes(Bw, 1, 2) if zero == 2 else np.swapaxes(Bw, 1, 2) @ Bw)
        q = np.einsum("ki,ki->k", keep, np.linalg.solve(M, keep[..., None])[..., 0])
        keep = keep * np.where(q > 0.999, np.sqrt(0.999 / np.maximum(q, 1e-300)), 1.0)[:, None]
        S = np.zeros((k, 5, 5))
        S[:, np.arange(5), np.arange(5)] = 1.0
        S[:, :2, 2:4] = Bw
        S[:, 2:4, :2] = np.swapaxes(Bw, 1, 2)
        idx = slice(0, 2) if zero == 2 else slice(2, 4)
        S[:, idx, 4] = keep
        S[:, 4, idx] = keep
        W = wishart_corr(S, n_samples, n_null, self.rng)
        return canonical_from_cov(W.reshape(-1, 5, 5)).reshape(k, n_null, 6)

    def conditional_null(self, Bw, a1, a2, test, n_samples, n_null):
        """H0: T is independent of source `test` (1 or 2) given the other source. Exact for
        Gaussian data conditional on the observed sources and the other source's relation
        to T: the tested source's whitened target vector is B'a_other + L g sqrt(1-|a_other|^2),
        with L L' its residual covariance given the other source and g the partial
        correlation vector (uniform direction, |g|^2 ~ Beta(1, (N-5)/2))."""
        k = len(Bw)
        Bt = np.swapaxes(Bw, 1, 2)
        if test == 2:
            other, mean, Sres = a1, np.einsum("kij,kj->ki", Bt, a1), np.eye(2) - Bt @ Bw
        else:
            other, mean, Sres = a2, np.einsum("kij,kj->ki", Bw, a2), np.eye(2) - Bw @ Bt
        L = np.linalg.cholesky(Sres)
        u = self.rng.standard_normal((k, n_null, 2))
        u /= np.linalg.norm(u, axis=-1, keepdims=True)
        n = np.maximum(np.asarray(n_samples, dtype=np.float64).reshape(-1, 1), 6.0)
        g2 = self.rng.beta(1.0, (n - 5.0) / 2.0, size=(k, n_null))
        scale = np.sqrt(np.maximum(1.0 - (other ** 2).sum(-1), 0.0))[:, None]
        new = mean[:, None] + np.einsum("kij,knj->kni", L, u) * (np.sqrt(g2) * scale)[..., None]
        Bb = np.broadcast_to(Bw[:, None], (k, n_null, 2, 2))
        ob = np.broadcast_to(other[:, None], (k, n_null, 2))
        return canonical_from_whitened(Bb, ob, new) if test == 2 else canonical_from_whitened(Bb, new, ob)

    def predict_batch(self, Cs, n_samples, forcenn=True, n_null=None, p_value_thresh=None,
                      null="atom", max_rows=1_000_000):
        """Cs: (k, 5, 5) covariance/correlation matrices ordered [x1a, x1b, x2a, x2b, target].
        n_samples: scalar or (k,) effective sample sizes (only used for the null).
        null="atom" (default): each atom is tested against a null under which it is 0
          (R: either source unrelated to T; U1/U2: T independent of that source given the
          other; S: either source adds nothing given the other). JMI: global null.
        null="global": every atom against "T unrelated to both sources"."""
        if null not in ("atom", "global"):
            raise ValueError("null must be 'atom' or 'global'")
        Bw, a1, a2 = whitened_blocks(Cs)
        p = canonical_from_whitened(Bw, a1, a2)
        atoms, R_raw, I1, I2, JMI = self.atoms_from_canonical(p, forcenn)
        out = {"atoms": atoms, "atoms_raw": assemble_atoms(R_raw, I1, I2, JMI, False),
               "mi_source1": I1, "mi_source2": I2, "jmi": JMI, "canonical": p}

        n_null = self.n_null if n_null is None else n_null
        thr = self.p_value_thresh if p_value_thresh is None else p_value_thresh
        if n_null and n_null > 0:
            k = len(p)
            n_samples = np.broadcast_to(np.asarray(n_samples, dtype=np.float64), (k,))
            pv = np.empty((k, 4)); p_jmi = np.empty(k)
            mu = np.empty((k, 4)); sd = np.empty((k, 4))
            step = max(1, max_rows // (n_null * (4 if null == "atom" else 1)))
            for i in range(0, k, step):
                sl = slice(i, min(k, i + step))
                ns = n_samples[sl]
                pg = self.null_canonical(Bw[sl], ns, n_null)          # global null (JMI)
                null_jmi = closed_form_mi(pg)[2]
                p_jmi[sl] = (1 + np.sum(null_jmi >= JMI[sl][:, None] - 1e-15, axis=1)) / (n_null + 1)
                if null == "global":
                    null_atoms = self.atoms_from_canonical(pg, forcenn)[0]
                    pv[sl] = (1 + np.sum(null_atoms >= atoms[sl][:, None, :] - 1e-15, axis=1)) / (n_null + 1)
                    mu[sl], sd[sl] = null_atoms.mean(1), null_atoms.std(1)
                else:
                    b, x1, x2 = Bw[sl], a1[sl], a2[sl]
                    ens = [self.marginal_null(b, x1, x2, 1, ns, n_null),
                           self.marginal_null(b, x1, x2, 2, ns, n_null),
                           self.conditional_null(b, x1, x2, 1, ns, n_null),
                           self.conditional_null(b, x1, x2, 2, ns, n_null)]
                    ens = [self.atoms_from_canonical(e, forcenn)[0] for e in ens]
                    pv[sl], mu[sl], sd[sl] = atom_pvalues(atoms[sl], *ens, n_null)
            out.update(null_mu=mu, null_sigma=sd, p_values=pv, p_jmi=p_jmi,
                       significant_atoms=np.where(pv < thr, atoms, 0.0))
        return out

    def predict_from_covariance(self, C, n_samples, **kw):
        res = self.predict_batch(np.asarray(C)[None], n_samples, **kw)
        return {k: v[0] for k, v in res.items()}


# ============================================================
# 6. EVALUATION
# ============================================================

def _random_within_source_map(C, rng):
    """Apply random invertible linear maps within each source and rescale the target."""
    M = np.zeros((5, 5))
    M[:2, :2] = rng.standard_normal((2, 2)) + 2 * np.eye(2)
    M[2:4, 2:4] = rng.standard_normal((2, 2)) + 2 * np.eye(2)
    M[4, 4] = rng.uniform(0.2, 5) * rng.choice([-1, 1])
    return M @ C @ M.T


def evaluate(est, test_rows, out_dir):
    report = {}
    names = ["R", "U1", "U2", "S"]
    p, R_t, se_t, I1, I2, JMI = test_rows[:, :6], test_rows[:, 6], test_rows[:, 7], *test_rows[:, 8:11].T

    # 6a. Test set vs teacher, binned by the strongest source-target canonical correlation.
    pred, R_raw_p, *_ = est.atoms_from_canonical(p, forcenn=True)
    teach = assemble_atoms(R_t, I1, I2, JMI, True)
    strength = np.maximum(p[:, 0], p[:, 1])
    bins = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    print("\nTest set: network vs single teacher run (bits). 'excess' removes the teacher's own MC noise.")
    print(f"{'max rho bin':>14} {'n':>6} {'mean JMI':>10} {'teacher se(R)':>13} "
          + " ".join(f"{'MAE ' + n:>10}" for n in names) + f" {'excess RMSE R':>14}")
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (strength > lo) & (strength <= hi)
        if m.sum() == 0:
            continue
        mae = np.abs(pred[m] - teach[m]).mean(0)
        excess = math.sqrt(max(np.mean((R_raw_p[m] - R_t[m]) ** 2) - np.mean(se_t[m] ** 2), 0.0))
        print(f"{f'({lo}, {hi}]':>14} {m.sum():6d} {JMI[m].mean():10.2e} {se_t[m].mean():13.2e} "
              + " ".join(f"{v:10.2e}" for v in mae) + f" {excess:14.2e}")
        rows.append(dict(bin=f"({lo},{hi}]", n=int(m.sum()), mean_jmi=float(JMI[m].mean()),
                         teacher_se=float(se_t[m].mean()), excess_rmse_R=excess,
                         **{f"mae_{n}": float(v) for n, v in zip(names, mae)}))
    report["test_by_strength"] = rows

    # 6b. Unrelated target -> all atoms exactly 0; one unrelated source -> R exactly 0.
    rng = np.random.default_rng(123)
    for _ in range(20):
        A = rng.standard_normal((5, 7)); C = A @ A.T
        C0 = C.copy(); C0[:4, 4] = C0[4, :4] = 0
        assert np.all(est.predict_from_covariance(C0, 900, n_null=0)["atoms"] == 0)
        C1 = C.copy(); C1[:2, 4] = C1[4, :2] = 0
        assert est.predict_from_covariance(C1, 900, n_null=0)["atoms"][0] == 0
    print("\nUnrelated target -> all atoms exactly 0; unrelated source -> redundancy exactly 0: OK")

    # 6c. Exact invariance to linear maps within each source.
    diffs = []
    for _ in range(200):
        A = rng.standard_normal((5, 7)); C = A @ A.T
        a = est.predict_from_covariance(C, 900, n_null=0)["atoms"]
        b = est.predict_from_covariance(_random_within_source_map(C, rng), 900, n_null=0)["atoms"]
        diffs.append(np.abs(a - b).max())
    print(f"Invariance to within-source linear maps: max |diff| = {max(diffs):.1e} bits")
    report["invariance_max_diff"] = float(max(diffs))

    # 6d. High-precision reference on weak targets: network vs one pid_mvn-sized exact run.
    ref_rows = []
    for _ in range(150):
        pw = weak_example(rng)
        if max(pw[0], pw[1]) > 0.3:
            continue
        J_ = closed_form_mi(pw)[2]
        ref = np.mean([iccs_redundancy(pw, 200_000, rng)[0] for _ in range(10)])   # 2M samples
        single = iccs_redundancy(pw, 100_000, rng)[0]                              # like pid_mvn
        net = est._raw_redundancy(pw[None], np.array([J_]))[0]
        ref_rows.append((J_, ref, single, net))
    ref_rows = np.array(ref_rows)
    e_single = np.abs(ref_rows[:, 2] - ref_rows[:, 1])
    e_net = np.abs(ref_rows[:, 3] - ref_rows[:, 1])
    print(f"\nWeak targets (max rho <= 0.3, n={len(ref_rows)}), raw redundancy vs a 2M-sample reference:")
    print(f"  one exact run (100k samples, as pid_mvn): MAE {e_single.mean():.2e} bits, "
          f"median relative to JMI {np.median(e_single / ref_rows[:, 0]):.3f}")
    print(f"  network:                                  MAE {e_net.mean():.2e} bits, "
          f"median relative to JMI {np.median(e_net / ref_rows[:, 0]):.3f}")
    report["weak_reference"] = dict(n=len(ref_rows), mae_single_exact=float(e_single.mean()),
                                    mae_network=float(e_net.mean()))

    # 6e. Cross-check of the teacher against the corrected pid_mvn port, on raw covariances.
    if iccs_mvn_fixed is not None:
        diffs, sds = [], []
        for _ in range(10):
            A = rng.standard_normal((5, 7)); C = A @ A.T
            runs = np.array([iccs_mvn_fixed.calc_pi_mvn(iccs_mvn_fixed.Lattice2D(), C, [2, 2, 1],
                                                       iccs_mvn_fixed.Iccs_mvn, forcenn=True).PI for _ in range(3)])
            pc = canonical_from_cov(C[None])[0]
            Rr = np.mean([iccs_redundancy(pc, 200_000, rng)[0] for _ in range(5)])
            ours = assemble_atoms(Rr, *closed_form_mi(pc), True)
            diffs.append(np.abs(runs.mean(0) - ours).max()); sds.append(runs.std(0).max())
        print(f"\nCorrected pid_mvn cross-check: max |pid_mvn mean - teacher| = {max(diffs):.2e} bits "
              f"(pid_mvn run-to-run SD up to {max(sds):.2e})")
        report["pid_mvn_fixed_max_diff"] = float(max(diffs))

    # 6f. Null calibration: independent targets, false-positive rate at alpha = 0.05.
    print("\nNull calibration (independent target, 300 datasets each): false-positive rate at 0.05")
    print(f"{'N':>6} {'sources':>10} {'target':>12}  " + " ".join(f"{n:>6}" for n in names + ["JMI"]))
    structures = {
        "indep": np.eye(4),
        "within": np.kron(np.eye(2), np.array([[1, 0.8], [0.8, 1.0]])),
        "mixed": None,
    }
    cal = []
    for N in (30, 300, 3000):
        for sname, Cx in structures.items():
            for tgt, prev in (("gaussian", None), ("binary", 0.5), ("binary", 0.1)):
                Cs = []
                for _ in range(300):
                    if Cx is None:
                        A = rng.standard_normal((4, 6)); Cm = A @ A.T
                    else:
                        Cm = Cx
                    X = rng.standard_normal((N, 4)) @ np.linalg.cholesky(Cm).T
                    if tgt == "gaussian":
                        y = rng.standard_normal(N); kind = "continuous"
                    else:
                        y = (rng.random(N) < prev).astype(int); kind = "binary"
                        if y.min() == y.max():
                            y[0] = 1 - y[0]
                    Cs.append(correlation_from_data(X[:, :2], X[:, 2:], y, kind))
                res = est.predict_batch(np.array(Cs), N, n_null=500)
                fpr = list((res["p_values"] < 0.05).mean(0)) + [float((res["p_jmi"] < 0.05).mean())]
                label = tgt if prev is None else f"binary p={prev}"
                print(f"{N:6d} {sname:>10} {label:>12}  " + " ".join(f"{v:6.3f}" for v in fpr))
                cal.append(dict(N=N, sources=sname, target=label,
                                **{f"fpr_{n}": float(v) for n, v in zip(names + ["JMI"], fpr)}))
    report["null_calibration"] = cal

    with open(Path(out_dir) / "evaluation.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def save_package(model, out_dir):
    out_dir = Path(out_dir)
    torch.save(model.state_dict(), out_dir / "redundancy_net_221.pt")
    # NumPy copy of the weights for torch-free inference (neural_pid_221.py).
    linears = [m for m in model.mlp if isinstance(m, nn.Linear)]
    np.savez(out_dir / "redundancy_net_221.npz",
             **{f"W{i}": m.weight.detach().cpu().numpy() for i, m in enumerate(linears)},
             **{f"b{i}": m.bias.detach().cpu().numpy() for i, m in enumerate(linears)})
    meta = {
        "varsizes": [2, 2, 1],
        "network_input": "pid_features(canonical): [2*phi-1, s, 2*R^2-1, B00, B01, B10, B11, sv1, sv2]",
        "canonical": "each source whitened and rotated so its target vector is (rho_k, 0); B = whitened cross block",
        "network_output": "rho = raw CCS redundancy / I(X1,X2;T)",
        "teacher": "Iccs_mvn with full covariance blocks (np.ix_ fix), varsizes [2,2,1]",
        "atoms": ["redundancy", "unique_source1", "unique_source2", "synergy"],
        "lattice": "Williams-Beer as pid_mvn.calc_pi_mvn; forcenn clips R, U1, U2 (not S)",
        "binary_target": "0/1 indicator (no copula normalisation)",
        "continuous_variables": "gcmi.copnorm (every source variable, and continuous targets)",
        "null": "permutation null: uniformly oriented whitened cross-correlations, R^2 ~ Beta(2, (N-5)/2)",
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
    print("Device:", device, "| joblib:", HAVE_JOBLIB, "| pid_mvn cross-check:", iccs_mvn_fixed is not None)

    train_rows = generate_dataset("train", CONFIG["n_train"], 0)
    val_rows = generate_dataset("validation", CONFIG["n_val"], 10_000_000)
    test_rows = generate_dataset("test", CONFIG["n_test"], 20_000_000)

    model = RedundancyNet(CONFIG["hidden_dim"], CONFIG["n_hidden_layers"]).to(device=device, dtype=torch.float64)
    train_model(model, training_arrays(train_rows), training_arrays(val_rows), device)
    save_package(model, out_dir)

    pid_estimator = NeuralPID221Estimator(TorchRho(model, device), n_null=CONFIG["n_null"],
                                          p_value_thresh=CONFIG["p_value_thresh"])
    evaluate(pid_estimator, test_rows, out_dir)
    print("\nSaved model package to", out_dir)
