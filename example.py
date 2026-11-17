"""Neural PID examples: binary target, continuous target, and many edges at once."""

import numpy as np

from neural_pid import NeuralPIDEstimator, correlation_from_data

rng = np.random.default_rng(0)
pid_estimator = NeuralPIDEstimator.load("neural_pid_output", n_null=1000)
names = ["Redundancy", "Unique X1", "Unique X2", "Synergy"]


def show(title, res):
    print(f"\n{title}")
    print(f"  I(X1;T) = {res['mi_source1']:.5f}  I(X2;T) = {res['mi_source2']:.5f}  "
          f"I(X1,X2;T) = {res['jmi']:.5f} bits (p = {res['p_jmi']:.4f})")
    for name, atom, p in zip(names, res["atoms"], res["p_values"]):
        print(f"  {name:11s} {atom:.5f} bits   p = {p:.4f}")


# ------------------------------------------------------------
# 1. Binary target (e.g. one condition vs the rest)
# ------------------------------------------------------------
N = 2000
y = (rng.random(N) < 0.3).astype(int)                  # 30% prevalence
x1 = 0.4 * y + rng.standard_normal(N)                  # both sources carry the label
x2 = 0.3 * y + 0.5 * x1 + rng.standard_normal(N)
C = correlation_from_data(x1, x2, y, target="binary")  # sources copnormed, target 0/1
show("Binary target", pid_estimator.predict_from_covariance(C, n_samples=N))

# ------------------------------------------------------------
# 2. Continuous target
# ------------------------------------------------------------
x1 = rng.standard_normal(N)
x2 = rng.standard_normal(N)
t = x1 + x2 + rng.standard_normal(N)                   # mostly synergistic about t
C = correlation_from_data(x1, x2, t, target="continuous")
show("Continuous target", pid_estimator.predict_from_covariance(C, n_samples=N))

# ------------------------------------------------------------
# 3. Many source pairs at once (use predict_batch: ~5,000x faster than exact PID)
# ------------------------------------------------------------
n_channels, N = 10, 1500
X = rng.standard_normal((N, n_channels))
y = (rng.random(N) < 0.5).astype(int)
X[:, :3] += 0.3 * y[:, None]                           # first 3 channels relate to the target
pairs = [(i, j) for i in range(n_channels) for j in range(i + 1, n_channels)]
Cs = np.stack([correlation_from_data(X[:, i], X[:, j], y, "binary") for i, j in pairs])
res = pid_estimator.predict_batch(Cs, n_samples=N)     # n_null defaults to 1000
sig = res["p_values"] < 0.05
print(f"\n{len(pairs)} channel pairs: significant redundancy in {sig[:, 0].sum()}, "
      f"synergy in {sig[:, 3].sum()}")
best = int(np.argmax(res["jmi"]))
print(f"Most informative pair {pairs[best]}: atoms = {np.round(res['atoms'][best], 5)}")
