# Neural PID

Neural amortisation of Partial Information Decomposition (PID) for rapid estimation of redundant, unique, and synergistic information between pairs of sources about a target variable.

## Overview

This repository provides a neural estimator of PID based on Gaussian Copula Mutual Information trained on synthetic data of varying signal and noise correlations (r = -0.99 - 0.99) and sample sizes (N = 10 - 1000000) (see Fig.2 of Ref.3) that predicts:

- Redundancy (R)
- Unique Information 1 (UY)
- Unique Information 2 (UZ)
- Synergy (S)

In addition to PID atoms, the model estimates the null distribution parameters for each atom:

- Null mean (μ)
- Null standard deviation (σ)

These parameters allow statistical significance testing without explicit permutation testing by approximating the null distribution as Gaussian:

\[
z = \frac{\text{atom} - \mu_{\text{null}}}{\sigma_{\text{null}}}
\]

\[
p = 2\left(1-\Phi(|z|)\right)
\]

where \(\Phi\) denotes the cumulative distribution function of the standard normal distribution.

---

## Test Performance

### PID Atom Prediction Accuracy

| Atom | MAE | RMSE |
|--------|--------|--------|
| Redundancy | 4.81 × 10⁻³ | 7.49 × 10⁻³ |
| Unique 1 | 5.01 × 10⁻³ | 8.45 × 10⁻³ |
| Unique 2 | 5.38 × 10⁻³ | 9.63 × 10⁻³ |
| Synergy | 6.95 × 10⁻³ | 1.32 × 10⁻² |

### Null Distribution Estimation Accuracy

| Atom | μ MAE | σ MAE |
|--------|--------|--------|
| Redundancy | 2.92 × 10⁻³ | 2.17 × 10⁻³ |
| Unique 1 | 6.37 × 10⁻³ | 2.93 × 10⁻³ |
| Unique 2 | 7.51 × 10⁻³ | 2.97 × 10⁻³ |
| Synergy | 2.83 × 10⁻³ | 2.30 × 10⁻³ |

---

## Example Usage

```python
from gcmi import copnorm
import numpy as np

x1 = np.squeeze(copnorm(X[:, 0][None, :]))
x2 = np.squeeze(copnorm(X[:, 1][None, :]))
t  = np.squeeze(copnorm(Y[None, :]))

xyz = np.column_stack([x1, x2, t])

Cxyz = (xyz.T @ xyz) / (xyz.shape[0] - 1)
np.fill_diagonal(Cxyz, 1.0)

result = pid_estimator.predict_from_covariance(
    Cxyz,
    n_samples=xyz.shape[0]
)

atoms = result["atoms"]
null_mu = result["null_mu"]
null_sigma = result["null_sigma"]
p_values = result["p_values"]
significant_atoms = result["significant_atoms"]
```

---

## Dependencies

- [GCMI](https://github.com/robince/gcmi)
- NumPy
- SciPy
- torch

Install with:

```bash
pip install numpy scipy torch
```

and install GCMI from:

```bash
git clone https://github.com/robince/gcmi.git
```

---

## References

1. Ince, R. A. A. (2017). *Measuring multivariate redundant information with pointwise common change in surprisal*. Entropy, 19(7), 318.

2. Ince, R. A. A., Giordano, B. L., Kayser, C., Rousselet, G. A., Gross, J., & Schyns, P. G. (2017). *A statistical framework for neuroimaging data analysis based on mutual information estimated via a Gaussian copula*. Human Brain Mapping, 38(3), 1541-1573.

3. O’Reilly, D., Shaw, W., Hilt, P., de Castro Aguiar, R., Astill, SL., Delis, I. *Quantifying the diverse contributions of hierarchical muscle interactions to motor function*. Iscience. 2025 Jan 17;28(1).
