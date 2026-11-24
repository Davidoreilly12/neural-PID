"""
Neural PID: amortised exact Gaussian(-copula) Partial Information Decomposition.

The network predicts only the CCS redundancy (Ince, 2017), which is the one PID term
without a closed form, as a fraction of the joint mutual information. I(X1;T), I(X2;T)
and I(X1,X2;T) are computed exactly from the correlation matrix, and the atoms come
from the Williams-Beer lattice, exactly as in pid_mvn.calc_pi_mvn.

Inference is pure NumPy/SciPy (no torch needed).

Usage
-----
    from neural_pid import NeuralPIDEstimator, correlation_from_data

    pid_estimator = NeuralPIDEstimator.load("neural_pid_output")
    C = correlation_from_data(x1, x2, y, target="binary")   # or target="continuous"
    res = pid_estimator.predict_from_covariance(C, n_samples=len(y))
    res["atoms"]       # [redundancy, unique_x1, unique_x2, synergy] in bits
    res["p_values"]    # permutation-null p-value for each atom
"""

import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import erf

try:
    import gcmi
except ImportError:          # only needed for correlation_from_data
    gcmi = None

LN2 = math.log(2.0)
MIN_DET = 1e-8               # smallest determinant accepted for null correlation matrices

__all__ = [
    "NeuralPIDEstimator",
    "RedundancyNet",
    "correlation_from_data",
    "closed_form_mi",
    "assemble_atoms",
    "pid_features",
]


# ============================================================
# Exact pieces
# ============================================================

def corr_det(r12, r1t, r2t):
    return 1.0 - r12 ** 2 - r1t ** 2 - r2t ** 2 + 2.0 * r12 * r1t * r2t


def closed_form_mi(r12, r1t, r2t):
    """I(X1;T), I(X2;T), I(X1,X2;T) in bits for unit-variance Gaussians (vectorised)."""
    r12, r1t, r2t = (np.asarray(v, dtype=np.float64) for v in (r12, r1t, r2t))
    I1 = -0.5 * np.log1p(-r1t ** 2) / LN2
    I2 = -0.5 * np.log1p(-r2t ** 2) / LN2
    # JMI = -0.5 log2(1 - R^2), R^2 = squared multiple correlation of T on (X1, X2).
    rsq = (r1t ** 2 + r2t ** 2 - 2 * r12 * r1t * r2t) / np.maximum(1.0 - r12 ** 2, 1e-15)
    JMI = -0.5 * np.log1p(-np.clip(rsq, 0.0, 1.0 - 1e-15)) / LN2
    return I1, I2, JMI


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
    """Williams-Beer lattice as pid_mvn.calc_pi_mvn: R, U1, U2 clipped at 0 if forcenn, S never.
    Returns [..., 4] = [R, U1, U2, S]."""
    R = np.maximum(R_raw, 0.0) if forcenn else R_raw
    U1, U2 = I1 - R, I2 - R
    if forcenn:
        U1, U2 = np.maximum(U1, 0.0), np.maximum(U2, 0.0)
    S = JMI - R - U1 - U2
    return np.stack([R, U1, U2, S], axis=-1)


# ============================================================
# Preprocessing
# ============================================================

def copnorm_1d(x):
    """gcmi copula normalisation (ordinal ranks; ties broken by sample order)."""
    if gcmi is None:
        raise ImportError("gcmi is required: pip install 'gcmi @ git+https://github.com/robince/gcmi.git#subdirectory=python'")
    x = np.asarray(x, dtype=np.float64)
    return np.asarray(gcmi.copnorm(x[None, :]), dtype=np.float64).ravel()


def binary_indicator(y):
    """Binary target as a 0/1 indicator (NOT copula-normalised)."""
    y = np.asarray(y)
    levels = np.unique(y)
    if len(levels) != 2:
        raise ValueError(f"Binary target needs exactly 2 levels, got {len(levels)}.")
    return (y == levels[1]).astype(np.float64)


def correlation_from_data(x1, x2, y, target="binary"):
    """3x3 correlation matrix [x1, x2, target]: sources gcmi-copnormed, binary target as a
    0/1 indicator, continuous target gcmi-copnormed."""
    z1, z2 = copnorm_1d(x1), copnorm_1d(x2)
    if target == "binary":
        zt = binary_indicator(y)
    elif target == "continuous":
        zt = copnorm_1d(y)
    else:
        raise ValueError("target must be 'binary' or 'continuous'")
    return np.corrcoef(np.vstack([z1, z2, zt]))


# ============================================================
# Network (NumPy forward pass of the trained MLP)
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


def _gelu(x):
    """Exact (erf) GELU, identical to torch.nn.GELU()."""
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


class RedundancyNet:
    """Predicts rho = raw CCS redundancy / I(X1,X2;T). Exactly symmetric in the two sources.
    Weights: redundancy_net.npz with W0, b0, ..., Wk, bk (W in torch layout: out x in)."""

    def __init__(self, weights):
        n = len([k for k in weights if k.startswith("W")])
        self.layers = [(np.asarray(weights[f"W{i}"], np.float64), np.asarray(weights[f"b{i}"], np.float64))
                       for i in range(n)]

    @classmethod
    def load(cls, path):
        with np.load(path) as w:
            return cls({k: w[k] for k in w.files})

    def _mlp(self, x):
        for W, b in self.layers[:-1]:
            x = _gelu(x @ W.T + b)
        W, b = self.layers[-1]
        return (x @ W.T + b)[:, 0]

    def __call__(self, f):
        g = f.copy()
        g[:, 0] = -g[:, 0]
        return 0.5 * (self._mlp(f) + self._mlp(g))


# ============================================================
# Estimator
# ============================================================

class NeuralPIDEstimator:
    """predict_from_covariance(C, n_samples) for one 3x3 matrix ordered [x1, x2, target];
    predict_batch(Cs, n_samples) for many at once (much faster)."""

    def __init__(self, model, n_null=1000, p_value_thresh=0.05, seed=0, n_threads=None):
        self.model = model
        self.n_threads = n_threads or min(8, os.cpu_count() or 1)
        self.n_null = n_null
        self.p_value_thresh = p_value_thresh
        self.rng = np.random.default_rng(seed)

    @classmethod
    def load(cls, model_dir="neural_pid_output", **kw):
        return cls(RedundancyNet.load(Path(model_dir) / "redundancy_net.npz"), **kw)

    def _raw_redundancy(self, r12, r1t, r2t, JMI, batch=8_192):
        F = pid_features(r12, r1t, r2t).reshape(-1, 4)
        starts = range(0, len(F), batch)
        if len(starts) > 1 and self.n_threads > 1:
            # NumPy/SciPy release the GIL, so row blocks run in parallel threads.
            with ThreadPoolExecutor(self.n_threads) as pool:
                rho = np.concatenate(list(pool.map(lambda i: self.model(F[i:i + batch]), starts)))
        else:
            rho = self.model(F) if len(F) else np.empty(0)
        rho = rho.reshape(np.shape(JMI))
        zero = (np.asarray(r1t) == 0) | (np.asarray(r2t) == 0) | (JMI <= 0)
        return np.where(zero, 0.0, rho * JMI)

    def atoms_from_correlations(self, r12, r1t, r2t, forcenn=True):
        """Returns (atoms [...,4], raw redundancy, I(X1;T), I(X2;T), I(X1,X2;T))."""
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
            bad = corr_det(rr, a, b) <= MIN_DET
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
