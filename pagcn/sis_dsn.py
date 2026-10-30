"""SIS spread with the dynamics of DeepSN's simulator (diffusion_models.sis_model): beta = lambda = 0.001,
synchronous updates, at most 200 iterations, a run stops once its infected count is unchanged for 20 consecutive
iterations; the statistic is the number of infected nodes at the end, in % of n. Vectorised over runs."""
import numpy as np, scipy.sparse as sp


def sis_dsn(A, S, runs, seed=99, beta=0.001, lam=0.001, steps=200, stable=20, batch=250):
    A = sp.csr_matrix(A, dtype=np.float32); n = A.shape[0]; rng = np.random.default_rng(seed); tot = 0.0
    lq = np.log1p(-beta)
    for b0 in range(0, runs, batch):
        R = min(batch, runs - b0)
        I = np.zeros((n, R), bool); I[list(S)] = True
        cnt = I.sum(0); still = np.zeros(R, int); alive = np.ones(R, bool)
        for _ in range(steps):
            if not alive.any(): break
            c = A @ I.astype(np.float32); u = rng.random((n, R))
            nxt = np.where(I, ~(u < lam), (c > 0) & (u < 1 - np.exp(c * lq)))
            nxt[:, ~alive] = I[:, ~alive]                      # stopped runs keep their final state
            new = nxt.sum(0)
            still = np.where(alive & (new == cnt), still + 1, np.where(alive, 0, still))
            alive &= still < stable
            I, cnt = nxt, new
        tot += I.sum()
    return 100.0 * tot / runs / n
