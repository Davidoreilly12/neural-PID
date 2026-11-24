"""Neural PID [2,2,1] examples: two 2-channel sources and one target."""

import numpy as np

from neural_pid_221 import NeuralPID221Estimator, correlation_from_data

rng = np.random.default_rng(0)
est = NeuralPID221Estimator.load("neural_pid_221_output", n_null=1000)
names = ["Redundancy", "Unique X1", "Unique X2", "Synergy"]


def show(title, res):
    print(f"\n{title}")
    print(f"  I(X1;T) = {res['mi_source1']:.5f}  I(X2;T) = {res['mi_source2']:.5f}  "
          f"I(X1,X2;T) = {res['jmi']:.5f} bits (p = {res['p_jmi']:.4f})")
    for name, atom, p in zip(names, res["atoms"], res["p_values"]):
        print(f"  {name:11s} {atom:.5f} bits ({atom / max(res['jmi'], 1e-300):6.1%} of JMI)   p = {p:.4f}")


# 1. Binary target: both 2-channel sources carry the label (mostly redundant)
N = 2000
y = (rng.random(N) < 0.3).astype(int)
shared = 0.5 * y + rng.standard_normal(N)
X1 = np.column_stack([shared + 0.5 * rng.standard_normal(N), rng.standard_normal(N)])
X2 = np.column_stack([rng.standard_normal(N), shared + 0.5 * rng.standard_normal(N)])
show("Binary target", est.predict_from_covariance(correlation_from_data(X1, X2, y, "binary"), n_samples=N))

# 2. Continuous target that is the sum of one channel from each source (synergy)
X1 = rng.standard_normal((N, 2)); X2 = rng.standard_normal((N, 2))
t = X1[:, 0] + X2[:, 1] + rng.standard_normal(N)
show("Continuous target", est.predict_from_covariance(correlation_from_data(X1, X2, t, "continuous"), n_samples=N))

# 3. Many pairs of 2-channel sources at once
n_groups, N = 8, 1500
X = rng.standard_normal((N, 2 * n_groups))
y = (rng.random(N) < 0.5).astype(int)
X[:, :4] += 0.3 * y[:, None]                         # groups 0 and 1 relate to the target
groups = [X[:, 2 * g:2 * g + 2] for g in range(n_groups)]
pairs = [(i, j) for i in range(n_groups) for j in range(i + 1, n_groups)]
Cs = np.stack([correlation_from_data(groups[i], groups[j], y, "binary") for i, j in pairs])
res = est.predict_batch(Cs, n_samples=N)
sig = res["p_values"] < 0.05
print(f"\n{len(pairs)} source pairs: significant redundancy in {sig[:, 0].sum()}, synergy in {sig[:, 3].sum()}")
best = int(np.argmax(res["jmi"]))
print(f"Most informative pair {pairs[best]}: atoms = {np.round(res['atoms'][best], 5)}")
