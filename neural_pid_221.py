"""
Neural PID [2,2,1]: amortised exact Gaussian(-copula) PID for two 2-variable sources and
one scalar target (5x5 covariance, pid_mvn varsizes [2, 2, 1]).

The network predicts only the CCS redundancy (Ince, 2017) as a fraction of the joint
mutual information. I(X1;T), I(X2;T) and I(X1,X2;T) are exact closed forms, and the atoms
come from the Williams-Beer lattice as in pid_mvn.calc_pi_mvn. Atoms are in bits and sum
exactly to I(X1,X2;T).

The teacher is Iccs_mvn with full covariance blocks. (The original Python port of
pid_mvn indexes blocks with Cfull[idx, idx], which keeps only the diagonal of each
multivariate block; see iccs_mvn_fixed.py.)

Inference is pure NumPy/SciPy (no torch needed).

Usage
-----
    from neural_pid_221 import NeuralPID221Estimator, correlation_from_data

    est = NeuralPID221Estimator.load("neural_pid_221_output")
    C = correlation_from_data(X1, X2, y, target="binary")    # X1, X2: (N, 2); or "continuous"
    res = est.predict_from_covariance(C, n_samples=len(y))
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
SWAP_PERM = [0, 1, 2, 3, 5, 4, 6, 7, 8]
SWAP_SIGN = np.array([-1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64)

__all__ = [
    "NeuralPID221Estimator",
    "RedundancyNet",
    "correlation_from_data",
    "canonical_from_cov",
    "closed_form_mi",
    "assemble_atoms",
    "pid_features",
]


# ============================================================
# Canonical form and exact pieces
# ============================================================

def whitened_blocks(Cs):
    """(k,5,5) covariances ordered [x1a, x1b, x2a, x2b, t] -> whitened cross block Bw (k,2,2)
    and whitened target vectors a1, a2 (k,2)."""
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
    t00, t01 = u1 * b00 + v1 * b10, u1 * b01 + v1 * b11
    t10, t11 = -v1 * b00 + u1 * b10, -v1 * b01 + u1 * b11
    B00, B01 = t00 * u2 + t01 * v2, -t00 * v2 + t01 * u2
    B10, B11 = t10 * u2 + t11 * v2, -t10 * v2 + t11 * u2
    s1 = np.where(B10 < 0, -1.0, 1.0)
    s2 = np.where(B01 < 0, -1.0, 1.0)
    return np.stack([r1, r2, B00, B01 * s2, B10 * s1, B11 * s1 * s2], axis=-1)


def canonical_from_cov(Cs):
    """(k,5,5) covariances -> (k,6) canonical parameters [rho1, rho2, B00, B01, B10, B11]."""
    return canonical_from_whitened(*whitened_blocks(Cs))


def _cross_terms(p):
    r1, r2, B00, B01, B10, B11 = np.moveaxis(np.asarray(p, dtype=np.float64), -1, 0)
    M00 = 1.0 - (B00 ** 2 + B10 ** 2)
    M11 = 1.0 - (B01 ** 2 + B11 ** 2)
    M01 = -(B00 * B01 + B10 * B11)
    det_s = M00 * M11 - M01 ** 2
    d0, d1 = r2 - B00 * r1, -B01 * r1
    q = (M11 * d0 ** 2 - 2 * M01 * d0 * d1 + M00 * d1 ** 2) / np.where(det_s > 0, det_s, np.inf)
    return r1, r2, r1 ** 2 + q, det_s


def closed_form_mi(p):
    """I(X1;T), I(X2;T), I(X1,X2;T) in bits from canonical parameters (vectorised)."""
    r1, r2, rsq, _ = _cross_terms(p)
    I1 = -0.5 * np.log1p(-r1 ** 2) / LN2
    I2 = -0.5 * np.log1p(-r2 ** 2) / LN2
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

def copnorm_cols(X):
    """gcmi copula normalisation of each column of X (N x k)."""
    if gcmi is None:
        raise ImportError("gcmi is required: pip install 'gcmi @ git+https://github.com/robince/gcmi.git#subdirectory=python'")
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
    if target == "binary":
        zt = binary_indicator(y)
    elif target == "continuous":
        zt = copnorm_cols(y)[:, 0]
    else:
        raise ValueError("target must be 'binary' or 'continuous'")
    return np.corrcoef(np.column_stack([Z, zt]), rowvar=False)


# ============================================================
# Network (NumPy forward pass of the trained MLP)
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


def _gelu(x):
    """Exact (erf) GELU, identical to torch.nn.GELU()."""
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


class RedundancyNet:
    """Predicts rho = raw CCS redundancy / I(X1,X2;T). Exactly symmetric in the two sources.
    Weights: redundancy_net_221.npz with W0, b0, ..., Wk, bk (W in torch layout: out x in).
    Row blocks run in parallel threads (NumPy/SciPy release the GIL)."""

    def __init__(self, weights, n_threads=None, block=8_192):
        n = len([k for k in weights if k.startswith("W")])
        self.layers = [(np.asarray(weights[f"W{i}"], np.float64), np.asarray(weights[f"b{i}"], np.float64))
                       for i in range(n)]
        self.n_threads = n_threads or min(8, os.cpu_count() or 1)
        self.block = block

    @classmethod
    def load(cls, path, **kw):
        with np.load(path) as w:
            return cls({k: w[k] for k in w.files}, **kw)

    def _mlp(self, x):
        for W, b in self.layers[:-1]:
            x = _gelu(x @ W.T + b)
        W, b = self.layers[-1]
        return (x @ W.T + b)[:, 0]

    def _sym(self, f):
        return 0.5 * (self._mlp(f) + self._mlp(f[:, SWAP_PERM] * SWAP_SIGN))

    def __call__(self, F):
        starts = range(0, len(F), self.block)
        if len(starts) > 1 and self.n_threads > 1:
            with ThreadPoolExecutor(self.n_threads) as pool:
                return np.concatenate(list(pool.map(lambda i: self._sym(F[i:i + self.block]), starts)))
        return self._sym(F) if len(F) else np.empty(0)


# ============================================================
# Estimator
# ============================================================

class NeuralPID221Estimator:
    """predict_from_covariance(C, n_samples) for one 5x5 matrix ordered
    [x1a, x1b, x2a, x2b, target]; predict_batch(Cs, n_samples) for many at once."""

    def __init__(self, model, n_null=1000, p_value_thresh=0.05, seed=0):
        self.model = model
        self.n_null = n_null
        self.p_value_thresh = p_value_thresh
        self.rng = np.random.default_rng(seed)

    @classmethod
    def load(cls, model_dir="neural_pid_221_output", n_threads=None, **kw):
        return cls(RedundancyNet.load(Path(model_dir) / "redundancy_net_221.npz", n_threads=n_threads), **kw)

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
