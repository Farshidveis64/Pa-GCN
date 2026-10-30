"""Scale suite helpers: identical model (pagcn.model), but graph preprocessing that never forms A^2.

GraphLarge reproduces every quantity of pagcn.model.Graph:
  d2 = diag(At^2), d3 = diag(At^3), Phi = (r2, r3, r4) with r_t = (P^t)_ii,
computed in row chunks so that only (chunk x n) blocks of At^2 / P^2 exist at any time; total work is
O(sum_v d_v^2), memory O(m + chunk block).  r4 uses reversibility: (P^2)_ji = d_i (P^2)_ij / d_j.
Also: weighted-cascade IC (p_uv = 1/d_v) with vectorised RR generation, and random-walk node sampling
for the training subgraph.
"""
import numpy as np, scipy.sparse as sp, torch, numba
from . import model as PF
from .model import ETA, _t


class GraphLarge(PF.Graph):
    def __init__(self, A, budget=4_000_000):
        A = sp.csr_matrix(A, dtype=np.float64); A.setdiag(0); A.eliminate_zeros(); A.sort_indices()
        self.A = A; n = self.n = A.shape[0]
        deg = self.deg = np.asarray(A.sum(1)).ravel(); ds = np.maximum(deg, 1)
        self.nonisol = np.nonzero(deg > 0)[0]
        d12 = sp.diags(np.where(deg > 0, ds ** -0.5, 0.0))
        At = (d12 @ A @ d12).tocsr(); At.sort_indices(); self.At = At
        self.tA = _t(At)
        P = (sp.diags(1 / ds) @ A).tocsr(); P.sort_indices(); PT = P.T.tocsr(); PT.sort_indices()
        d2 = np.asarray(At.multiply(At).sum(1)).ravel()                 # At symmetric
        r2 = np.asarray(P.multiply(PT).sum(1)).ravel()
        d3 = np.zeros(n); r3 = np.zeros(n); r4 = np.zeros(n)
        # chunk rows so that the 2-hop block has at most ~budget structural nonzeros
        work = np.asarray(A @ deg).ravel()                                # sum of neighbour degrees
        start = 0
        while start < n:
            end = start + 1; acc = work[start]
            while end < n and acc + work[end] <= budget: acc += work[end]; end += 1
            R = slice(start, end)
            X = (At[R] @ At).tocsr(); d3[R] = np.asarray(X.multiply(At[R]).sum(1)).ravel()
            Y = (P[R] @ P).tocsr(); r3[R] = np.asarray(Y.multiply(PT[R]).sum(1)).ravel()
            Yc = Y.tocoo(); rows = Yc.row + start
            r4[R] = np.bincount(Yc.row, weights=ds[rows] * Yc.data ** 2 / ds[Yc.col], minlength=end - start)
            start = end
        self.d2, self.d3 = d2, d3
        self.dR = ETA ** 2 * d2 + ETA ** 3 * d3
        one = np.ones(n); v1 = At @ one; v2 = At @ v1; v3 = At @ v2
        self.a_row = torch.tensor(v1, dtype=torch.float32)
        self.r_row = torch.tensor(ETA * v1 + ETA ** 2 * v2 + ETA ** 3 * v3 - self.dR, dtype=torch.float32)
        self.tdR = torch.tensor(self.dR, dtype=torch.float32).unsqueeze(1)
        Phi = np.stack([r2, r3, r4], 1)
        self.Phi = torch.tensor((Phi - Phi.mean(0)) / (Phi.std(0) + 1e-12), dtype=torch.float32)
        self.Phi_raw = Phi
        self.Ain = P
        self.no_R = False


def wc_probs(A):
    """weighted cascade: P[u, v] = 1/d_v for every edge (u, v)."""
    A = sp.csr_matrix(A, dtype=np.float64); deg = np.asarray(A.sum(1)).ravel()
    return (A @ sp.diags(1 / np.maximum(deg, 1))).tocsr()


@numba.njit(cache=True)
def _gen(indptr, indices, probs, n, R, seed):
    np.random.seed(seed)
    st = np.zeros(R + 1, np.int64); out = np.empty(max(16 * R, 1024), np.int32); m = 0
    mark = np.zeros(n, np.int64); q = np.empty(n, np.int32)
    for r in range(R):
        root = np.random.randint(n); h = 0; t = 1; q[0] = root; mark[root] = r + 1
        while h < t:
            v = q[h]; h += 1
            for e in range(indptr[v], indptr[v + 1]):
                u = indices[e]
                if mark[u] != r + 1 and np.random.random() < probs[e]:
                    mark[u] = r + 1; q[t] = u; t += 1
        if m + t > out.shape[0]:
            new = np.empty(2 * (m + t), np.int32); new[:m] = out[:m]; out = new
        out[m:m + t] = q[:t]; m += t; st[r + 1] = m
    return st, out[:m]


def rr_sets(A, R, seed, P=None):
    """R reverse-reachable sets under IC with edge probabilities P (rows u -> cols v)."""
    P = wc_probs(A) if P is None else P
    PT = sp.csr_matrix(P.T); PT.sort_indices()
    st, nodes = _gen(PT.indptr.astype(np.int64), PT.indices.astype(np.int32), PT.data.astype(np.float64),
                     PT.shape[0], R, seed)
    return sp.csr_matrix((np.ones(len(nodes), np.float32), nodes, st), shape=(R, A.shape[0]))


def rw_sample(A, size, seed=0, restart=0.15):
    """random walk with restart (re-seeded at a fresh random node when stuck); induced subgraph."""
    A = sp.csr_matrix(A); rng = np.random.default_rng(seed); n = A.shape[0]
    seen = set(); v = int(rng.integers(n)); root = v; steps = 0
    while len(seen) < size:
        seen.add(v); steps += 1
        nb = A.indices[A.indptr[v]:A.indptr[v + 1]]
        if len(nb) == 0 or rng.random() < restart: v = root
        else: v = int(nb[rng.integers(len(nb))])
        if steps % (50 * size) == 0: root = v = int(rng.integers(n))
    idx = np.array(sorted(seen))
    return A[idx][:, idx].tocsr(), idx
