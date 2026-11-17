# Neural PID

Neural amortisation of Partial Information Decomposition (PID) for rapid estimation of redundant, unique, and synergistic information between pairs of sources about a target variable.

## Overview

This repository provides a neural estimator of the Gaussian (copula) PID with CCS redundancy (Ref. 1). It takes the 3 × 3 correlation matrix of two sources and a target and returns:

- Redundancy (R)
- Unique Information 1 (U1)
- Unique Information 2 (U2)
- Synergy (S)

plus a permutation-based p-value for each atom and for the joint mutual information.

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

Under a permutation of the target, the source–target correlations (r₁ₜ, r₂ₜ) are approximately jointly normal on the Fisher-z scale, with variance 1/(N − 3) and correlation r₁₂ between them. So the null depends only on r₁₂ and N.

For each input, the estimator:

1. Draws `n_null` (default 1,000) null correlation pairs.
2. Passes them through the same network.
3. Returns empirical p-values: \(p = (1 + \#\{\text{null} \ge \text{observed}\}) / (n_{\text{null}} + 1)\).

There is no Gaussian assumption about the null, which is strongly skewed for non-negative atoms.

`n_samples` should be the **effective** number of independent samples. For autocorrelated data (time series, sleep epochs) the raw sample count gives p-values that are too small.

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

False-positive rate at α = 0.05 with an independent target. 300 simulated datasets per condition, 27 conditions: N ∈ {30, 300, 3000} × r₁₂ ∈ {0, 0.5, 0.9} × target ∈ {Gaussian, binary p = 0.5, binary p = 0.1}.

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
| Neural PID, `predict_batch` with a 1,000-draw null | 7.8 ms |

The 1,000-draw null replaces an exact 1,000-permutation test, which would take about 23 s per pair.

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
p_values = result["p_values"]                    # permutation-null p-value per atom
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

Pass `n_null=0` to skip significance testing. See `example.py` for a runnable script.

---

## Changes from the Previous Model

- **Near-zero floor removed.** The previous network output all four atoms directly through a softplus. It could not reach zero, which left a positive floor of about 0.002 bits on redundancy and synergy for weak targets. Atoms are now built from the exact mutual-information terms and are exactly zero when there is no information.
- **Null heads replaced.** The previous Gaussian null-mean and null-SD heads collapsed during training to a constant mean and near-zero SD, so the p-values were unusable. They are replaced by the Monte Carlo permutation null above.
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

---

## Limitations

- Two scalar sources and a scalar target (`varsizes = [1, 1, 1]`). Multivariate sources need the exact `pid_mvn` estimator.
- Gaussian (copula) PID: dependence beyond what the copula correlation captures is not measured, and binary targets are approximated as Gaussian.
- The permutation null assumes exchangeable samples. Use an effective sample size for autocorrelated data.

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
