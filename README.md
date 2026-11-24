# Neural PID

Neural amortisation of Partial Information Decomposition (PID) for rapid estimation of redundant, unique, and synergistic information between pairs of sources about a target variable.

Two models are included:

| Model | Sources and target | `pid_mvn` varsizes | Module |
|--------|--------|--------|--------|
| Neural PID | two scalar sources, scalar target (3 × 3 matrix) | `[1, 1, 1]` | `neural_pid.py` |
| Neural PID [2,2,1] | two 2-variable sources, scalar target (5 × 5 matrix) | `[2, 2, 1]` | `neural_pid_221.py` (see [Multivariate Sources](#multivariate-sources-neural-pid-221)) |

The sections up to Example Usage describe the `[1, 1, 1]` model. The `[2, 2, 1]` model follows the same design.

## Overview

This repository provides a neural estimator of the Gaussian (copula) PID with CCS redundancy (Ref. 1). It takes the 3 × 3 correlation matrix of two sources and a target and returns:

- Redundancy (R)
- Unique Information 1 (U1)
- Unique Information 2 (U2)
- Synergy (S)

plus a p-value for each atom, tested against a null under which that atom is zero, and a p-value for the joint mutual information.

The network learns only the part of the PID that has no closed form: the **CCS redundancy**, a Monte Carlo integral (`Iccs_mvn` in [PyPID_mvn](https://github.com/AlessandroCorsini/PyPID_mvn)). Everything else is computed exactly from the correlation matrix (r₁₂ between the sources, r₁ₜ and r₂ₜ between each source and the target):

\[
I(X_1;T) = -\tfrac12\log_2(1-r_{1t}^2), \qquad I(X_2;T) = -\tfrac12\log_2(1-r_{2t}^2)
\]

\[
I(X_1,X_2;T) = -\tfrac12\log_2(1-R^2_{T|X_1X_2}), \qquad R^2_{T|X_1X_2} = \frac{r_{1t}^2 + r_{2t}^2 - 2r_{12}r_{1t}r_{2t}}{1-r_{12}^2}
\]

The network predicts the raw redundancy as a fraction of the joint mutual information, \(R = \rho \cdot I(X_1,X_2;T)\). The atoms then follow from the Williams–Beer lattice (Ref. 4), exactly as in `pid_mvn.calc_pi_mvn`:

\[
U_1 = I(X_1;T) - R, \qquad U_2 = I(X_2;T) - R, \qquad S = I(X_1,X_2;T) - R - U_1 - U_2
\]

With `forcenn=True` (default), R, U1 and U2 are clipped at zero and S is not, as in `calc_pi_mvn`.

Consequences:

- **All atoms are exactly 0** when the target is unrelated to the sources.
- **R + U1 = I(X1;T), R + U2 = I(X2;T), and all four atoms sum to I(X1,X2;T)**, exactly.
- **The estimate is exactly symmetric:** swapping the two sources swaps U1 and U2 and leaves R and S unchanged.
- Only R carries approximation error. U1 and U2 inherit it with opposite sign, and S with the same sign.

---

## Preprocessing

| Variable | Treatment |
|--------|--------|
| Sources | Gaussian copula normalisation (`gcmi.copnorm`, Ref. 2) |
| Continuous target | Gaussian copula normalisation (`gcmi.copnorm`) |
| Binary target | 0/1 indicator, **not** copula-normalised |

Copula-normalising a binary variable breaks its ties by sample order. That adds noise that does not depend on the data, and it lowers every source–target correlation by a factor that depends on prevalence. The 0/1 indicator gives the point-biserial correlation. A binary target is still treated as Gaussian by the PID, which is an approximation; it is accurate for small effects.

`correlation_from_data(x1, x2, y, target="binary" | "continuous")` builds the matrix this way.

---

## Significance Testing

Each atom is tested against a null hypothesis under which **that atom** is zero. A single "target unrelated to both sources" null is not enough, because of a property of CCS redundancy. A source that is truly unrelated to the target still has a sample correlation of about ±1/√N with it. Its mutual information grows with the square of that correlation, but CCS redundancy grows roughly in proportion to the correlation itself. So when one source is informative and the other is not, the measured R is well above what the global null produces. Under the global null, R was declared significant in 30–73% of such datasets (see Null Calibration).

| Atom | Null hypothesis | Null correlations |
|--------|--------|--------|
| R | Either source has zero correlation with T (CCS R = 0 exactly in that case). p = the larger of the two p-values. | Parametric bootstrap: sample correlation matrices of N Gaussian samples (Wishart) from the observed matrix with that source's correlation to T set to 0 |
| U1 | T independent of X1 given X2 | Exact, conditional on the data: r₁ₜ = r₁₂r₂ₜ + √(1−r₁₂²)√(1−r₂ₜ²)·g, with g the partial correlation, g² ~ Beta(½, (N−3)/2), random sign |
| U2 | T independent of X2 given X1 | As for U1, with the sources swapped |
| S | Either source adds nothing given the other. p = the larger of the U1 and U2 null p-values for S. | As for U1 and U2 |
| I(X1,X2;T) | T unrelated to both sources | Fisher-z draws of (r₁ₜ, r₂ₜ) with variance 1/(N − 3) and correlation r₁₂ |

A conditional-independence null would be wrong for R. When X2 is a noisy copy of X1, X2 carries genuinely redundant information, yet T is independent of X2 given X1.

For each input, the estimator draws `n_null` (default 1,000) null correlation matrices for each hypothesis and passes them through the same network. It returns empirical p-values: \(p = (1 + \#\{\text{null} \ge \text{observed}\}) / (n_{\text{null}} + 1)\). There is no Gaussian assumption about the null, which is strongly skewed for non-negative atoms.

`null="global"` gives the previous behaviour: every atom is tested against "T unrelated to both sources".

`n_samples` should be the **effective** number of independent samples. For autocorrelated data (time series, sleep epochs) the raw sample count gives p-values that are too small.

The tests concern significance only. The atom values themselves are plug-in estimates from the sample correlation matrix, so they are biased upward at small N, R most of all. This is the same as for an exact PID of the same matrix. Atoms that fail their test are set to 0 in `significant_atoms`.

---

## Training

- **Teacher:** the exact CCS redundancy for each correlation matrix, computed with 200,000 Monte Carlo samples (`pid_mvn` uses 100,000). The raw value, before `forcenn` clipping, is stored together with its Monte Carlo standard error.
- **Training data:** 100,000 / 10,000 / 10,000 correlation matrices (train / validation / test):
  - 30% uniform random correlation matrices.
  - 50% weak targets: target strength √(r₁ₜ² + r₂ₜ²) log-uniform from 10⁻³ to 1. Extra examples with near-collinear sources, and with one source almost unrelated to the target.
  - 20% computed from simulated data with N = 10 to 20,000:
    - The synthetic EMG simulator of Ref. 3 (Fig. 2) with its binary task target.
    - Gaussian data with binary targets at 1–50% prevalence.
    - Gaussian data with continuous targets.
- **Network inputs:** four features that don't depend on sign or source order: direction of (|r₁ₜ|, |r₂ₜ|), r₁₂ with signs aligned, target strength, and squared multiple correlation.
- **Network:** 4 × 128 GELU MLP, averaged over source order, float64.
- **Loss:** weighted MSE on ρ, with weights 1 / (teacher SE² + 0.05²).
- **Optimiser:** AdamW, cosine schedule, 300 epochs.

`train_neural_pid.py` reproduces the model.

---

## Test Performance

### PID Atom Prediction Accuracy

Mean absolute error (bits) on the 10,000 held-out matrices, binned by the strongest source–target correlation. "Teacher SE" is the Monte Carlo standard error of the exact value itself, so errors at that level can't be resolved further.

| max \|rₜ\| | n | Mean I(X1,X2;T) | Teacher SE | R | U1 | U2 | S |
|--------|------:|--------|--------|--------|--------|--------|--------|
| (0, 0.01] | 2,096 | 3.24 × 10⁻³ | 2.28 × 10⁻⁶ | 1.69 × 10⁻⁶ | 8.39 × 10⁻⁷ | 9.50 × 10⁻⁷ | 6.34 × 10⁻⁷ |
| (0.01, 0.05] | 1,922 | 2.27 × 10⁻² | 1.69 × 10⁻⁵ | 1.53 × 10⁻⁵ | 1.02 × 10⁻⁵ | 9.13 × 10⁻⁶ | 5.95 × 10⁻⁶ |
| (0.05, 0.1] | 913 | 4.41 × 10⁻² | 5.33 × 10⁻⁵ | 4.69 × 10⁻⁵ | 3.18 × 10⁻⁵ | 3.26 × 10⁻⁵ | 1.77 × 10⁻⁵ |
| (0.1, 0.2] | 895 | 8.09 × 10⁻² | 1.13 × 10⁻⁴ | 9.11 × 10⁻⁵ | 5.95 × 10⁻⁵ | 6.48 × 10⁻⁵ | 3.33 × 10⁻⁵ |
| (0.2, 0.3] | 603 | 1.57 × 10⁻¹ | 2.00 × 10⁻⁴ | 1.75 × 10⁻⁴ | 1.17 × 10⁻⁴ | 1.31 × 10⁻⁴ | 7.43 × 10⁻⁵ |
| (0.3, 0.5] | 976 | 4.09 × 10⁻¹ | 3.27 × 10⁻⁴ | 3.59 × 10⁻⁴ | 2.80 × 10⁻⁴ | 2.76 × 10⁻⁴ | 1.98 × 10⁻⁴ |
| (0.5, 1.0] | 2,595 | 1.61 | 7.68 × 10⁻⁴ | 3.26 × 10⁻³ | 2.79 × 10⁻³ | 2.68 × 10⁻³ | 2.21 × 10⁻³ |

Up to |rₜ| = 0.5, the error is about the same as the teacher's own Monte Carlo noise. In the weakest bin it is below 0.1% of the joint mutual information.

### Weak Targets: Network vs Exact

Raw redundancy for 131 weak-target matrices (max |rₜ| ≤ 0.3), against a 2-million-sample reference:

| Estimator | MAE (bits) | Median error / I(X1,X2;T) |
|--------|--------|--------|
| One exact run (100,000 MC samples, as `pid_mvn`) | 3.10 × 10⁻⁵ | 0.2% |
| Neural PID | 1.41 × 10⁻⁵ | 0.1% |

The network learns the noise-free value of the Monte Carlo integral, so for weak effects it is more accurate than a single exact run.

### Agreement with `pid_mvn`

The training teacher matches `pid_mvn.calc_pi_mvn` (mean of 3 runs, 10 random matrices) to within 9.7 × 10⁻⁴ bits, inside `pid_mvn`'s own run-to-run SD (up to 2.8 × 10⁻³ bits).

### Null Calibration

**Atom-specific nulls (default).** Fraction of datasets with p < 0.05, from 300 simulated datasets per cell. Each range covers N ∈ {100, 1000} × target ∈ {continuous, binary p = 0.5}. The latent target is t = a·X1 + b·X2 + noise.

| Scenario | Atom | True value | Atom null | Global null |
|--------|--------|--------|--------|--------|
| X1 informative, X2 independent | R | 0 | **0.02–0.05** | 0.30–0.73 |
| | U2 | 0 | 0.01–0.06 | 0.00–0.02 |
| | S | 0 | **0.04–0.08** | 0.29–0.46 |
| | U1 | > 0 | 0.76–1.00 | 0.74–1.00 |
| X2 a noisy copy of X1, T from X1 only | U2 | 0 | 0.01–0.05 | 0.01–0.07 |
| | S | 0 | 0.02–0.05 | 0.05–0.09 |
| | R | > 0 | 0.61–1.00 | 0.65–1.00 |
| All independent | R, S | 0 | 0.00–0.01 | 0.03–0.07 |
| | U1, U2 | 0 | 0.04–0.06 | 0.04–0.06 |
| X1, X2 independent, T from both | U1, U2 | > 0 | 0.72–1.00 | 0.56–1.00 |

The tests for R and S combine two p-values by taking the larger. They are therefore conservative when both of their null hypotheses hold (all independent).

**Global null (`null="global"`).** False-positive rate at α = 0.05 with an independent target. 300 simulated datasets per condition, 27 conditions: N ∈ {30, 300, 3000} × r₁₂ ∈ {0, 0.5, 0.9} × target ∈ {Gaussian, binary p = 0.5, binary p = 0.1}.

| | R | U1 | U2 | S | I(X1,X2;T) |
|--------|--------|--------|--------|--------|--------|
| Mean false-positive rate | 0.052 | 0.049 | 0.051 | 0.050 | 0.050 |

Every one of the 135 rates lies between 0.013 and 0.077. The sampling SD for 300 datasets is about 0.013, so this range is what you'd expect for a nominal 0.05. Full results are in `neural_pid_output/evaluation.json`.

### Speed

Per source pair, on a 20-thread desktop CPU, NumPy inference:

| Method | Time |
|--------|--------|
| Exact PID (`pid_mvn`-style, 100,000 MC samples) | 22.9 ms |
| Neural PID, one `predict_from_covariance` call | 0.11 ms |
| Neural PID, `predict_batch` | 18 µs |
| Neural PID, `predict_batch` with 1,000-draw global null | 7.8 ms |
| Neural PID, `predict_batch` with 1,000-draw atom-specific nulls (default) | about 39 ms |

The atom-specific nulls take about 4× longer than the global null (39 vs 9.3 ms in the same run), because they draw four null ensembles. A 1,000-permutation exact test would take about 23 s per pair.

---

## Example Usage

```python
import numpy as np
from neural_pid import NeuralPIDEstimator, correlation_from_data

pid_estimator = NeuralPIDEstimator.load("neural_pid_output", n_null=1000)

# Two sources and a binary target (use target="continuous" for a continuous one)
C = correlation_from_data(x1, x2, y, target="binary")

result = pid_estimator.predict_from_covariance(C, n_samples=len(y))

atoms = result["atoms"]                          # [R, U1, U2, S] in bits
p_values = result["p_values"]                    # p-value per atom (atom-specific null)
significant_atoms = result["significant_atoms"]  # atoms with p >= 0.05 set to 0
jmi = result["jmi"]                              # I(X1,X2;T), exact
null_mu, null_sigma = result["null_mu"], result["null_sigma"]
```

For many pairs, stack the matrices and make one call. This is much faster:

```python
Cs = np.stack([correlation_from_data(X[:, i], X[:, j], y) for i, j in pairs])
results = pid_estimator.predict_batch(Cs, n_samples=n_eff)   # arrays, one row per pair
```

Also returned:
- `atoms_raw`: atoms without the `forcenn` clipping.
- `mi_source1`, `mi_source2`: I(X1;T) and I(X2;T).
- `p_jmi`: p-value for the joint mutual information.

For R and S, `null_mu` and `null_sigma` describe the null that gave the larger p-value. Pass `null="global"` for the single "target unrelated to both sources" null, and `n_null=0` to skip significance testing. See `example.py` for a runnable script.

---

## Multivariate Sources: Neural PID [2,2,1]

`neural_pid_221.py` amortises the same Gaussian CCS PID for **two sources of two variables each** and a scalar target. This is `pid_mvn` with `varsizes = [2, 2, 1]`; an example is a gene's expression and accessibility as one source. The design is the same as the `[1, 1, 1]` model:
- The network predicts only the CCS redundancy, as a fraction of the joint mutual information.
- I(X1;T), I(X2;T) and I(X1,X2;T) are exact closed forms.
- The atoms come from the Williams–Beer lattice. They are in bits and sum exactly to I(X1,X2;T).

### Indexing bug in the Python `Iccs_mvn` for multivariate sources

The Python port of `Iccs_mvn` fills covariance blocks with `Cfull[idx, idx]`. With two index arrays, NumPy pairs the indices element by element and returns only the **diagonal** of the block. MATLAB's `Cfull(idx, idx)` returns the whole block.

For multivariate sources, this drops the within-source covariance from I(X1;T), I(X2;T) and the redundancy terms, along with some cross-source terms. On random 5 × 5 matrices the atoms were wrong by up to ~0.3 bits, and some matrices failed with "not positive definite". For `[1, 1, 1]`, every index array has length 1, so the bug has no effect there.

`iccs_mvn_fixed.py` is the port with every block filled using `np.ix_`. The `[2, 2, 1]` network is trained on this corrected version. Any earlier `[2, 2, 1]` results from the unfixed port should be recomputed.

### Canonical input

CCS redundancy is exactly invariant to any invertible linear map within a source (checked to 1e-16). So each source is whitened, then rotated so that its correlation vector with T becomes (ρₖ, 0). A 5 × 5 matrix then reduces to 6 numbers:
- ρ1 and ρ2: each source's canonical correlation with T;
- the 2 × 2 whitened cross-source block B, with signs fixed so that B₁₀ ≥ 0 and B₀₁ ≥ 0.

The closed forms are I(Xₖ;T) = −½log₂(1 − ρₖ²) and I(X1,X2;T) = −½log₂(1 − R²), where R² is the squared multiple correlation of T on all four source variables.

The network inputs are the direction of (ρ1, ρ2), the strength √(ρ1² + ρ2²), 2R² − 1, the entries of B and its singular values. The network output is averaged over the source swap, so swapping the sources swaps U1 and U2 exactly. All atoms are exactly 0 for an unrelated target, and R is exactly 0 when either source is unrelated.

### Preprocessing and significance

Preprocessing is the same as for `[1, 1, 1]`: `gcmi.copnorm` on each source variable, a binary target as a 0/1 indicator, and `copnorm` on a continuous target.

The atom-specific nulls are also the same, extended to two-variable sources. For U1, U2 and S, the tested source's whitened target-correlation vector is B′a + L·g·√(1 − |a|²), where a is the other source's whitened target-correlation vector and L is the Cholesky factor of the tested source's residual covariance given the other source. The partial-correlation vector g has a uniform direction and |g|² ~ Beta(1, (N − 5)/2). The global null uses a uniformly oriented whitened cross-correlation vector with R² ~ Beta(2, (N − 5)/2).

### Training

- **Teacher:** the corrected `Iccs_mvn` with 200,000 Monte Carlo samples. It matches `iccs_mvn_fixed.py` to within 1e-14 bits on identical random draws.
- **Data:** 100,000 / 10,000 / 10,000 matrices (train / validation / test):
  - 30% random matrices: half from partial correlations drawn uniformly in (−0.99, 0.99) (C-vine), so every correlation reaches ±0.99, and half A·Aᵀ;
  - 50% weak targets, with strength log-uniform from 10⁻³ and cross-block singular values often near 1;
  - 20% computed from simulated data with N = 20 to 20,000, from three generators:
    - a four-channel version of the EMG simulator of Ref. 3, with signal correlations uniform in (−0.99, 0.99) and C-vine noise correlations;
    - Gaussian data with binary targets at 1–50% prevalence;
    - Gaussian data with continuous targets.
- **Network:** 4 × 128 GELU MLP, float64.
- **Loss:** weighted MSE on ρ, with weights 1 / (teacher SE² + 0.05²).
- **Optimiser:** AdamW, cosine schedule, 300 epochs.

`train_neural_pid_221.py` reproduces the model.

### Test performance

Mean absolute error (bits) on the 10,000 held-out matrices, binned by the larger canonical source–target correlation:

| max ρ | n | Mean I(X1,X2;T) | Teacher SE | R | U1 | U2 | S |
|--------|------:|--------|--------|--------|--------|--------|--------|
| (0, 0.01] | 2,080 | 2.57 × 10⁻³ | 2.54 × 10⁻⁶ | 2.97 × 10⁻⁶ | 1.38 × 10⁻⁶ | 1.45 × 10⁻⁶ | 1.72 × 10⁻⁶ |
| (0.01, 0.05] | 1,944 | 3.56 × 10⁻² | 2.12 × 10⁻⁵ | 3.28 × 10⁻⁵ | 1.89 × 10⁻⁵ | 1.86 × 10⁻⁵ | 1.83 × 10⁻⁵ |
| (0.05, 0.1] | 972 | 4.13 × 10⁻² | 6.77 × 10⁻⁵ | 6.81 × 10⁻⁵ | 5.69 × 10⁻⁵ | 5.65 × 10⁻⁵ | 4.53 × 10⁻⁵ |
| (0.1, 0.2] | 868 | 7.83 × 10⁻² | 1.41 × 10⁻⁴ | 1.54 × 10⁻⁴ | 1.31 × 10⁻⁴ | 1.31 × 10⁻⁴ | 1.08 × 10⁻⁴ |
| (0.2, 0.3] | 549 | 1.49 × 10⁻¹ | 2.47 × 10⁻⁴ | 3.28 × 10⁻⁴ | 2.96 × 10⁻⁴ | 2.81 × 10⁻⁴ | 2.48 × 10⁻⁴ |
| (0.3, 0.5] | 875 | 3.63 × 10⁻¹ | 3.89 × 10⁻⁴ | 6.17 × 10⁻⁴ | 5.43 × 10⁻⁴ | 5.32 × 10⁻⁴ | 4.58 × 10⁻⁴ |
| (0.5, 1.0] | 2,712 | 1.54 | 9.75 × 10⁻⁴ | 5.12 × 10⁻³ | 4.58 × 10⁻³ | 4.61 × 10⁻³ | 4.07 × 10⁻³ |

- **Up to ρ = 0.5**, the error is close to the teacher's own Monte Carlo noise.
- **Strong targets (ρ > 0.5)** have the largest errors: about 0.3% of the joint mutual information on average, with a heavier tail. The RMSE of R beyond the teacher's noise is 0.019 bits.
- **Weak targets** (max ρ ≤ 0.3, 139 matrices), raw redundancy against a 2-million-sample reference:
  - one exact run with 100,000 samples: MAE 2.8 × 10⁻⁵ bits, median error 0.9% of I(X1,X2;T);
  - the network: MAE 6.7 × 10⁻⁵ bits, median error 0.5% of I(X1,X2;T).
- **Corrected `pid_mvn` cross-check** (mean of 3 runs, 10 random matrices): the largest difference is 2.6 × 10⁻³ bits. The run-to-run SD of `pid_mvn` itself is up to 1.3 × 10⁻³ bits.
- **Invariance:** atoms change by at most 6 × 10⁻¹³ bits under random invertible linear maps within each source.

### Null calibration

Fraction of datasets with p < 0.05, from 300 simulated datasets per cell. Each range covers N ∈ {100, 1000} × target ∈ {continuous, binary p = 0.5}. Each source has 2 variables.

| Scenario | Atom | True value | Atom null | Global null |
|--------|--------|--------|--------|--------|
| X1 informative, X2 independent | R | 0 | **0.01–0.03** | 0.29–0.93 |
| | U2 | 0 | 0.01–0.06 | 0.00–0.03 |
| | S | 0 | **0.03–0.05** | 0.30–0.52 |
| | U1 | > 0 | 0.67–1.00 | 0.65–1.00 |
| X2 a noisy copy of X1, T from X1 only | U2 | 0 | 0.04–0.05 | 0.01–0.10 |
| | S | 0 | 0.03–0.05 | 0.04–0.10 |
| | R | > 0 | 0.60–1.00 | 0.62–1.00 |
| All independent | R, S | 0 | 0.00–0.01 | 0.02–0.07 |
| | U1, U2 | 0 | 0.04–0.06 | 0.04–0.06 |
| X1, X2 independent, T from both | U1, U2 | > 0 | 0.59–1.00 | 0.43–1.00 |

I also tested the global null alone, with independent targets: 27 conditions (N ∈ {30, 300, 3000}; independent, within-source-correlated and mixed sources; Gaussian and binary targets). The mean false-positive rates were 0.045 for R, 0.052 for U1, 0.051 for U2, 0.049 for S and 0.050 for I(X1,X2;T), all within 0.017–0.093. Full results are in `neural_pid_221_output/evaluation.json`.

### Speed

Per source pair, on a 20-thread desktop CPU, with NumPy inference:

| Method | Time |
|--------|--------|
| Exact PID (corrected `pid_mvn`, 100,000 MC samples) | 245 ms |
| Neural PID [2,2,1], one `predict_from_covariance` call | 0.23 ms |
| Neural PID [2,2,1], `predict_batch` | 12 µs |
| Neural PID [2,2,1], `predict_batch` with 1,000-draw global null | 9.7 ms |
| Neural PID [2,2,1], `predict_batch` with 1,000-draw atom-specific nulls (default) | 51 ms |

### Usage

```python
from neural_pid_221 import NeuralPID221Estimator, correlation_from_data

est = NeuralPID221Estimator.load("neural_pid_221_output", n_null=1000)

# X1, X2: (N, 2) arrays (two variables per source); y binary or continuous
C = correlation_from_data(X1, X2, y, target="binary")     # 5 x 5, order [x1a, x1b, x2a, x2b, target]
res = est.predict_from_covariance(C, n_samples=len(y))
res["atoms"], res["p_values"], res["jmi"]

# Many pairs: stack (k, 5, 5) covariance or correlation matrices
res = est.predict_batch(Cs, n_samples=n_eff)
```

Any 5 × 5 covariance ordered `[x1a, x1b, x2a, x2b, target]` works directly, with no need to standardise it first. `res["canonical"]` returns the 6 canonical parameters. See `example_221.py` for a runnable script.

---

## Changes from the Previous Model

- **Near-zero floor removed.** The previous network output all four atoms directly through a softplus. It could not reach zero, which left a positive floor of about 0.002 bits on redundancy and synergy for weak targets. Atoms are now built from the exact mutual-information terms and are exactly zero when there is no information.
- **Null heads replaced.** The previous Gaussian null-mean and null-SD heads collapsed during training to a constant mean and near-zero SD, so the p-values were unusable. They are replaced by Monte Carlo nulls passed through the same network.
- **Atom-specific nulls.** Each atom is now tested against a null under which it is zero (see Significance Testing). Under the global null, redundancy and synergy were often called significant when only one source was informative. The network is unchanged.
- **Binary target no longer copula-normalised** (see Preprocessing).
- **`1/√N` no longer a network input.** The PID of a correlation matrix does not depend on N; N enters only through the null.
- **Teacher clipping bias removed.** The teacher now stores redundancy before `forcenn` clipping, which removes an upward bias for weak, noisy targets.

---

## Repository Contents

| Path | Contents |
|--------|--------|
| `neural_pid.py` | Inference: `NeuralPIDEstimator`, `correlation_from_data` (NumPy/SciPy only) |
| `example.py` | Runnable examples (binary target, continuous target, many pairs) |
| `train_neural_pid.py` | Data generation, training and evaluation |
| `neural_pid_output/redundancy_net.npz` | Trained weights (NumPy) |
| `neural_pid_output/redundancy_net.pt` | The same weights (PyTorch state dict, used by the training script) |
| `neural_pid_output/model_metadata.json` | Model description |
| `neural_pid_output/evaluation.json` | All test results above |
| `neural_pid_output/training_history.npy` | Train / validation loss per epoch |
| `neural_pid_output/{train,validation,test}.npz` | Training data: r₁₂, r₁ₜ, r₂ₜ, teacher redundancy, its MC SE, I(X1;T), I(X2;T), I(X1,X2;T), generator |
| `neural_pid_221.py` | `[2, 2, 1]` inference: `NeuralPID221Estimator`, `correlation_from_data` (NumPy/SciPy only) |
| `example_221.py` | Runnable `[2, 2, 1]` examples |
| `train_neural_pid_221.py` | `[2, 2, 1]` data generation, training and evaluation |
| `iccs_mvn_fixed.py` | `pid_mvn` port with the multivariate indexing fix (the `[2, 2, 1]` teacher's reference) |
| `neural_pid_221_output/redundancy_net_221.npz` | `[2, 2, 1]` weights (NumPy); `redundancy_net_221.pt` is the same weights for PyTorch |
| `neural_pid_221_output/model_metadata.json`, `evaluation.json`, `training_history.npy` | `[2, 2, 1]` model description, test results, loss history |
| `neural_pid_221_output/{train,validation,test}.npz` | `[2, 2, 1]` training data: ρ1, ρ2, B₀₀, B₀₁, B₁₀, B₁₁, teacher redundancy, its MC SE, I(X1;T), I(X2;T), I(X1,X2;T), generator |

---

## Limitations

- Two sources and a scalar target only, with source sizes of either `[1, 1, 1]` or `[2, 2, 1]`. Other sizes need the exact estimator (`iccs_mvn_fixed.py` for multivariate sources).
- Gaussian (copula) PID: dependence beyond what the copula correlation captures is not measured, and binary targets are approximated as Gaussian.
- The nulls assume exchangeable samples. Use an effective sample size for autocorrelated data. The nulls for U1, U2, S and R are exact or parametric for Gaussian data, and approximate for binary targets.
- Atom values are plug-in estimates and are biased upward at small N. No bias correction is applied.

---

## Dependencies

Inference:

- NumPy
- SciPy
- [GCMI](https://github.com/robince/gcmi) (for `correlation_from_data`)

Training additionally needs PyTorch, and optionally joblib (parallel data generation) and [PyPID_mvn](https://github.com/AlessandroCorsini/PyPID_mvn) (teacher cross-check).

Install with:

```bash
pip install numpy scipy
```

```bash
pip install "gcmi @ git+https://github.com/robince/gcmi.git#subdirectory=python"
```

---

## References

1. Ince, R. A. A. (2017). *Measuring multivariate redundant information with pointwise common change in surprisal*. Entropy, 19(7), 318.

2. Ince, R. A. A., Giordano, B. L., Kayser, C., Rousselet, G. A., Gross, J., & Schyns, P. G. (2017). *A statistical framework for neuroimaging data analysis based on mutual information estimated via a Gaussian copula*. Human Brain Mapping, 38(3), 1541-1573.

3. O’Reilly, D., Shaw, W., Hilt, P., de Castro Aguiar, R., Astill, SL., Delis, I. *Quantifying the diverse contributions of hierarchical muscle interactions to motor function*. Iscience. 2025 Jan 17;28(1).

4. Williams, P. L., & Beer, R. D. (2010). *Nonnegative decomposition of multivariate information*. arXiv:1004.2515.
