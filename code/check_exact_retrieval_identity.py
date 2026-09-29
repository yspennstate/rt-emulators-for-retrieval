"""Check the exact retrieval-error identity of notes/retrieval_consistent_training.tex (Lemma 1) and its first-order form."""
import numpy as np

rng = np.random.default_rng(1)
worst_exact, worst_first = 0.0, 0.0
for _ in range(100000):
    a, t, s, rho = rng.uniform(0, 0.3), rng.uniform(1e-3, 1.0), rng.uniform(0, 0.9), rng.uniform(0.05, 0.9)
    q = 1 - rho * s
    L = a + rho * t / q
    ea, et, es = rng.normal(0, 0.3, 3) * np.array([0.1, t, 0.2])
    den = (t + et) + (s + es) * (L - a - ea)
    if abs(den) < 1e-6:
        continue
    exact = (L - a - ea) / den - rho
    ell = -(q / t) * (q * ea + rho * et) - rho ** 2 * es
    delta = (q / t) * (et - s * ea) + rho * es - (q / t) * ea * es
    worst_exact = max(worst_exact, abs(exact - (ell + (rho * q / t) * ea * es) / (1 + delta)) / max(1.0, abs(exact)))
    small = 1e-7
    den_s = (t + small * et) + (s + small * es) * (L - a - small * ea)
    ex_s = (L - a - small * ea) / den_s - rho
    lin_s = small * ell
    worst_first = max(worst_first, abs(ex_s - lin_s) / max(abs(lin_s), 1e-18))
print(f"exact identity, largest discrepancy: {worst_exact:.2e}")
print(f"first-order form at errors scaled by 1e-7, largest relative gap: {worst_first:.2e}")
