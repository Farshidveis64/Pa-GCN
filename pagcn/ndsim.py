"""Exact equivalents of the evaluation protocol behind the published baseline tables:
DeepIM's diffusion_evaluation (main/utils.py, github.com/triplej0079/DeepIM), which DeepSN reuses
("baseline results are sourced from Ling et al."). It runs ndlib with
  IC  : IndependentCascadesModel, edge threshold set once per undirected edge e=(a,b) of G.edges() to
        1/deg(b); ndlib looks the key up in either orientation, so both directions use 1/deg(b),
        where b is the second endpoint as networkx lists the edge (the larger index here)
  LT  : ThresholdModel, threshold 0.5 for every node: a node activates when at least half of its
        neighbours were active in the previous iteration (deterministic, synchronous)
  SIS : SISModel, beta = lambda = 0.001, tp_rate 1: infection w.p. 1-(1-beta)^(#infected nbrs),
        recovery w.p. lambda, one shared uniform per node per iteration, synchronous
  iteration_bunch(100): iteration 0 returns the initial state, so 99 update steps; the final
  status is counted (IC 'removed' counts as infected).
IC is evaluated exactly by an independent RR sample under these edge probabilities.
"""
import numpy as np, numba, scipy.sparse as sp, networkx as nx

STEPS = 99


def ic_edge_probs(A):
    """P[u, v] = probability that active u activates v, exactly as ndlib resolves the edge keys."""
    A = sp.csr_matrix(A); G = nx.from_scipy_sparse_array(A); deg = dict(G.degree())
    rows, cols, vals = [], [], []
    for a, b in G.edges():
        if a == b: continue
        p = 1.0 / deg[b]
        rows += [a, b]; cols += [b, a]; vals += [p, p]
    return sp.csr_matrix((vals, (rows, cols)), shape=A.shape)


@numba.njit(cache=True)
def _gen(indptr, indices, probs, n, R, seed):
    np.random.seed(seed)
    starts = np.zeros(R + 1, np.int64); cap = R * 16; buf = np.empty(cap, np.int32)
    mark = np.full(n, -1, np.int64); queue = np.empty(n, np.int32); pos = 0
    for r in range(R):
        root = np.random.randint(n); mark[root] = r; queue[0] = root; qh = 0; qt = 1
        while qh < qt:
            v = queue[qh]; qh += 1
            for e in range(indptr[v], indptr[v + 1]):          # row v of P^T: in-neighbours u, prob P[u, v]
                u = indices[e]
                if mark[u] != r and np.random.random() < probs[e]:
                    mark[u] = r; queue[qt] = u; qt += 1
        if pos + qt > cap:
            nc = max(cap * 2, pos + qt + 1); nb = np.empty(nc, np.int32); nb[:pos] = buf[:pos]; buf = nb; cap = nc
        buf[pos:pos + qt] = queue[:qt]; pos += qt; starts[r + 1] = pos
    return starts, buf[:pos].copy()


def rr_sets_ic(A, R, seed, P=None):
    P = ic_edge_probs(A) if P is None else P
    PT = sp.csr_matrix(P.T); PT.sort_indices()
    st, nodes = _gen(PT.indptr.astype(np.int64), PT.indices.astype(np.int32), PT.data.astype(np.float64),
                     PT.shape[0], R, seed)
    return sp.csr_matrix((np.ones(len(nodes), np.float32), nodes, st), shape=(R, A.shape[0]))


def lt_spread(A, S, steps=STEPS):
    A = sp.csr_matrix(A, dtype=np.float64); deg = np.asarray(A.sum(1)).ravel()
    x = np.zeros(A.shape[0], bool); x[list(S)] = True
    for _ in range(steps):
        frac = np.where(deg > 0, (A @ x.astype(np.float64)) / np.maximum(deg, 1), -1.0)
        new = x | (frac >= 0.5)
        if (new == x).all(): break
        x = new
    return 100.0 * x.sum() / A.shape[0]


def sis_spread(A, S, runs=1000, seed=0, beta=0.001, lam=0.001, steps=STEPS, batch=500):
    A = sp.csr_matrix(A, dtype=np.float32); n = A.shape[0]; rng = np.random.default_rng(seed); tot = 0.0
    lq = np.log1p(-beta)
    for b0 in range(0, runs, batch):
        R = min(batch, runs - b0)
        I = np.zeros((n, R), bool); I[list(S)] = True
        for _ in range(steps):
            c = A @ I.astype(np.float32); u = rng.random((n, R))
            inf = (~I) & (c > 0) & (u < 1 - np.exp(c * lq))
            rec = I & (u < lam)
            I = (I | inf) & ~rec
        tot += I.sum()
    return 100.0 * tot / runs / n
