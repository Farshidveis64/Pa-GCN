"""Reverse-reachable sets under weighted-cascade IC (p_uv = 1/d_v) on an undirected graph."""
import numpy as np, numba, scipy.sparse as sp


@numba.njit(cache=True)
def _gen(indptr, indices, deg, n, R, seed):
    np.random.seed(seed)
    starts = np.zeros(R + 1, np.int64)
    cap = R * 16
    buf = np.empty(cap, np.int32)
    mark = np.full(n, -1, np.int64)
    queue = np.empty(n, np.int32)
    pos = 0
    for r in range(R):
        root = np.random.randint(n)
        mark[root] = r; queue[0] = root; qh = 0; qt = 1
        while qh < qt:
            v = queue[qh]; qh += 1
            p = 1.0 / deg[v] if deg[v] > 0 else 0.0
            for e in range(indptr[v], indptr[v + 1]):
                u = indices[e]
                if mark[u] != r and np.random.random() < p:
                    mark[u] = r; queue[qt] = u; qt += 1
        if pos + qt > cap:
            nc = max(cap * 2, pos + qt + 1)
            nb = np.empty(nc, np.int32); nb[:pos] = buf[:pos]; buf = nb; cap = nc
        buf[pos:pos + qt] = queue[:qt]
        pos += qt
        starts[r + 1] = pos
    return starts, buf[:pos].copy()


def rr_sets(A, R, seed):
    A = sp.csr_matrix(A)
    deg = np.asarray(A.sum(1)).ravel().astype(np.float64)
    st, nodes = _gen(A.indptr.astype(np.int64), A.indices.astype(np.int32), deg, A.shape[0], R, seed)
    # incidence matrix M: R x n, M[r, v] = 1 iff v in RR set r
    M = sp.csr_matrix((np.ones(len(nodes), np.float32), nodes, st), shape=(R, A.shape[0]))
    return M


def greedy_rr(M, k, n):
    """Standard greedy max-coverage on RR sets (the RR-greedy reference)."""
    Mc = M.tocsc(); Mr = M.tocsr()
    cnt = np.asarray(Mr.sum(0)).ravel().astype(np.float64)
    covered = np.zeros(M.shape[0], bool); S = []
    for _ in range(k):
        cnt_masked = cnt.copy(); cnt_masked[S] = -1
        v = int(np.argmax(cnt_masked)); S.append(v)
        rows = Mc.indices[Mc.indptr[v]:Mc.indptr[v + 1]]
        new = rows[~covered[rows]]; covered[new] = True
        for r in new:
            cnt[Mr.indices[Mr.indptr[r]:Mr.indptr[r + 1]]] -= 1
    return S


def spread(Mcsc, S, n):
    """sigma(S) as % of n, estimated by RR coverage."""
    rows = np.unique(np.concatenate([Mcsc.indices[Mcsc.indptr[v]:Mcsc.indptr[v + 1]] for v in S])) if len(S) else []
    return 100.0 * len(rows) / Mcsc.shape[0]
