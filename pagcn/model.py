"""PA-GCN core model (matrix-free).

  SDS        Phi_i = (r^(2), r^(3), r^(4)),  r^(t)_i = (P^t)_ii,  P = D^-1 A,  standardized (structural input)
  seed state x^S (indicator), a^S (T rounds of the IC activation update), r/n (picks left this round)
  kernel     adaptive-horizon kernel (bottom of this file):  (K H)_v = sum_{s=1..S} w_s(v) (At^s H)_v,
             w(v) = softmax(MLP(Phi_v)), At = D^-1/2 A D^-1/2; the fixed-horizon kernel K(alpha) of the base
             class is kept for the ablation ("fixedK")
  encoder    H0 = MLP(inputs);  GCNII layers  H^{l} = ReLU(Z((1-b_l) I + b_l W_l)),
             Z = 0.9 K H^{l-1} + 0.1 H0,  b_l = log(0.5/l + 1);  jumping-knowledge MLP head
             (the virtual-node extension used for every result of the paper is in pagcn/vn.py)
  objective  IC: expected new RR coverage per relaxed pick (label-free, Coverage);
             LT: exact LT gain as reward, score-function estimator (LTReward); + gamma * L_smooth
  states     own seed-aware rollouts (every 20 epochs) and random sets; r = picks per round
  inference  seed-aware greedy, B = min(k, 100) rounds (B = min(k, 20) on the million-edge graphs)

Threads: torch uses TORCH_THREADS (default 1) threads.
"""
import math, numpy as np, scipy.sparse as sp, torch

T, ETA, L, D = 3, 0.5, 3, 64
BETA, GAMMA, DELTA, EPS, MPATH = 0.01, 0.05, 0.8, 1e-8, 512
import os as _os
torch.set_num_threads(int(_os.environ.get("TORCH_THREADS", "1")))
CFG = dict(sds=True, ap=True, kernel=True, ps=True, smooth=True, gcnii=True, jk=True, dagger=True,
           lr=1e-3, lr_alpha=1e-2, epochs=200, steps=2, states=16, L=L, beta=BETA, gamma=GAMMA)


def _t(M):
    M = sp.csr_matrix(M, dtype=np.float32); M.sort_indices()
    return torch.sparse_csr_tensor(torch.from_numpy(M.indptr.astype(np.int64)),
                                   torch.from_numpy(M.indices.astype(np.int64)),
                                   torch.from_numpy(M.data), M.shape)


class Graph:
    def __init__(self, A):
        A = sp.csr_matrix(A, dtype=np.float64); A.setdiag(0); A.eliminate_zeros()
        self.A = A; n = self.n = A.shape[0]
        deg = self.deg = np.asarray(A.sum(1)).ravel(); ds = np.maximum(deg, 1)
        self.nonisol = np.nonzero(deg > 0)[0]
        d12 = sp.diags(np.where(deg > 0, ds ** -0.5, 0.0))
        At = (d12 @ A @ d12).tocsr(); At.sort_indices(); self.At = At
        self.tA = _t(At)
        # diag(At^2), diag(At^3) for the off-diagonal correction of Rt
        A2 = (At @ At).tocsr(); self.d2 = A2.diagonal(); self.d3 = np.asarray(A2.multiply(At).sum(1)).ravel()
        self.dR = ETA ** 2 * self.d2 + ETA ** 3 * self.d3
        one = np.ones(n); v1 = At @ one; v2 = At @ v1; v3 = At @ v2
        self.a_row = torch.tensor(v1, dtype=torch.float32)
        self.r_row = torch.tensor(ETA * v1 + ETA ** 2 * v2 + ETA ** 3 * v3 - self.dR, dtype=torch.float32)
        self.tdR = torch.tensor(self.dR, dtype=torch.float32).unsqueeze(1)
        # SDS: exact return probabilities r^(2..4)
        P = (sp.diags(1 / ds) @ A).tocsr(); P2 = (P @ P).tocsr()
        Phi = np.stack([P2.diagonal(), np.asarray(P2.multiply(P.T).sum(1)).ravel(),
                        np.asarray(P2.multiply(P2.T).sum(1)).ravel()], 1)
        self.Phi = torch.tensor((Phi - Phi.mean(0)) / (Phi.std(0) + 1e-12), dtype=torch.float32)
        self.Phi_raw = Phi
        self.Ain = (sp.diags(1 / ds) @ A).tocsr()          # rows v, cols u, weight p_uv = 1/d_v
        self.no_R = False

    # ---- explicit Rt only where needed (effective resistance on small graphs) ----
    def Rt_explicit(self):
        n = self.n; Pw = sp.identity(n, format="csr"); R = sp.csr_matrix((n, n))
        for s in range(1, T + 1):
            Pw = (Pw @ self.At).tocsr(); R = R + (ETA ** s) * Pw
        R = R.tolil(); R.setdiag(0); R = R.tocsr(); R.eliminate_zeros(); R.sort_indices()
        return R

    def R_rows(self, U):
        """rows of Rt for the node list U (len(U) x n, csr, sorted), computed matrix-free."""
        E = sp.csr_matrix((np.ones(len(U)), (np.arange(len(U)), U)), shape=(len(U), self.n))
        X1 = (E @ self.At).tocsr(); X2 = (X1 @ self.At).tocsr(); X3 = (X2 @ self.At).tocsr()
        R = (ETA * X1 + ETA ** 2 * X2 + ETA ** 3 * X3).tocsr()
        ii = np.arange(len(U)); dv = np.asarray(R[ii, U]).ravel()
        R = (R - sp.csr_matrix((dv, (ii, U)), shape=R.shape)).tocsr()
        R.eliminate_zeros(); R.sort_indices(); return R

    def ap(self, S, rounds=T):
        n = self.n; x = np.zeros(n); x[S] = 1.0; a = x.copy(); M = self.Ain
        for _ in range(rounds):
            vals = np.log1p(-np.minimum(a[M.indices] * M.data, 1 - 1e-12))
            lg = np.zeros(n)
            nz = np.diff(M.indptr) > 0
            lg[nz] = np.add.reduceat(vals, M.indptr[:-1][nz]) if M.nnz else 0
            a = 1 - (1 - x) * np.exp(lg)
        return a

    @staticmethod
    def _row_lookup(M, rows, cols):
        out = np.zeros(len(rows))
        for t in range(len(rows)):
            s0, s1 = M.indptr[rows[t]], M.indptr[rows[t] + 1]
            j = np.searchsorted(M.indices[s0:s1], cols[t])
            if j < s1 - s0 and M.indices[s0 + j] == cols[t]: out[t] = M.data[s0 + j]
        return out

    def sample_paths(self, alpha, M, rng):
        a = self.a_row.numpy(); r = self.r_row.numpy(); At = self.At
        if len(self.nonisol) == self.n: cur = rng.integers(0, self.n, M)
        else: cur = self.nonisol[rng.integers(0, len(self.nonisol), M)]
        U, V, aUV, rUV = [], [], [], []
        for _ in range(T):
            uniq, inv = np.unique(cur, return_inverse=True)
            Rr = self.R_rows(uniq)
            pA = alpha * a[cur] / (alpha * a[cur] + (1 - alpha) * r[cur])
            useA = rng.random(M) < pA
            nxt = np.empty(M, int)
            idx = np.nonzero(useA)[0]; x = rng.random(len(idx))
            for t, i in enumerate(idx):
                s0, s1 = At.indptr[cur[i]], At.indptr[cur[i] + 1]
                cs = np.cumsum(At.data[s0:s1]) / At.data[s0:s1].sum()
                nxt[i] = At.indices[s0 + min(np.searchsorted(cs, x[t]), s1 - s0 - 1)]
            idx = np.nonzero(~useA)[0]; x = rng.random(len(idx))
            for t, i in enumerate(idx):
                row = inv[i]; s0, s1 = Rr.indptr[row], Rr.indptr[row + 1]
                cs = np.cumsum(Rr.data[s0:s1]) / Rr.data[s0:s1].sum()
                nxt[i] = Rr.indices[s0 + min(np.searchsorted(cs, x[t]), s1 - s0 - 1)]
            U.append(cur.copy()); V.append(nxt.copy())
            aUV.append(self._row_lookup(At, cur, nxt)); rUV.append(self._row_lookup(Rr, inv, nxt))
            cur = nxt
        return np.concatenate(U), np.concatenate(V), np.concatenate(aUV), np.concatenate(rUV)


class _KMul(torch.autograd.Function):
    """K(alpha) H, matrix-free: three applications of At plus a diagonal correction."""
    @staticmethod
    def forward(ctx, H, al, G):
        Hd = H.detach(); A = G.tA
        X1 = torch.sparse.mm(A, Hd)
        if G.no_R:
            ctx.G = G; ctx.a = 1.0; ctx.save_for_backward(X1, X1); return X1
        X2 = torch.sparse.mm(A, X1); X3 = torch.sparse.mm(A, X2)
        RH = ETA * X1 + ETA ** 2 * X2 + ETA ** 3 * X3 - G.tdR * Hd
        a = float(al); ctx.G = G; ctx.a = a; ctx.save_for_backward(X1, RH)
        return a * X1 + (1 - a) * RH

    @staticmethod
    def backward(ctx, g):
        X1, RH = ctx.saved_tensors; G = ctx.G; a = ctx.a; g = g.contiguous(); A = G.tA
        gH = ga = None
        if ctx.needs_input_grad[0]:
            Y1 = torch.sparse.mm(A, g)
            if a == 1.0: gH = Y1
            else:
                Y2 = torch.sparse.mm(A, Y1); Y3 = torch.sparse.mm(A, Y2)
                gH = a * Y1 + (1 - a) * (ETA * Y1 + ETA ** 2 * Y2 + ETA ** 3 * Y3 - G.tdR * g)
        if ctx.needs_input_grad[1]:
            ga = (g * (X1 - RH)).sum()
        return gH, ga, None


class PAGCN(torch.nn.Module):
    def __init__(self, G, cfg, seed):
        super().__init__()
        self.G, self.cfg = G, cfg; Lc = cfg.get("L", L)
        self.X = G.Phi if cfg["sds"] else torch.zeros(G.n, 0)
        fin_old = self.X.shape[1] + 1 + (1 if cfg["ap"] else 0)
        torch.manual_seed(seed)                              # same parameter stream as pagcn3
        self.inp = torch.nn.Sequential(torch.nn.Linear(fin_old, D), torch.nn.ReLU(), torch.nn.Linear(D, D))
        self.Ws = torch.nn.ModuleList([torch.nn.Linear(D, D, bias=False) for _ in range(Lc)])
        hd = D * (Lc + 1) if cfg["jk"] else D
        self.head = torch.nn.Sequential(torch.nn.Linear(hd, D), torch.nn.ReLU(), torch.nn.Linear(D, 1))
        self.a = torch.nn.Parameter(torch.zeros(()))
        torch.manual_seed(seed)
        self.inp[0] = torch.nn.Linear(fin_old + 1, D)        # + r/n channel

    def alpha(self):
        return torch.ones(()) if not self.cfg["kernel"] else torch.sigmoid(self.a)

    def prop(self, al, H):
        G = self.G
        if not self.cfg["kernel"]:
            G.no_R = True
            try: return _KMul.apply(H.contiguous(), torch.ones(()), G)
            finally: G.no_R = False
        return _KMul.apply(H.contiguous(), al, G)

    def state_feats(self, states):
        n = self.G.n; cols = []
        xs = torch.zeros(len(states), n, 1)
        for b, (S, _) in enumerate(states): xs[b, S, 0] = 1
        cols.append(xs)
        if self.cfg["ap"]:
            cols.append(torch.tensor(np.stack([self.G.ap(S) for S, _ in states]), dtype=torch.float32).unsqueeze(-1))
        cols.append(torch.tensor([r / n for _, r in states], dtype=torch.float32).view(-1, 1, 1).expand(-1, n, 1))
        return torch.cat(cols, -1)

    def forward(self, states, want_energy=False, return_hidden=False):
        G = self.G; Bn = len(states); n = G.n; al = self.alpha(); cfg = self.cfg
        X = torch.cat([self.X.unsqueeze(0).expand(Bn, -1, -1), self.state_feats(states)], -1)
        H0 = self.inp(X); H = H0; Hs = [H0]
        for l, W in enumerate(self.Ws, 1):
            Hf = H.permute(1, 0, 2).reshape(n, Bn * D)
            KH = self.prop(al, Hf).reshape(n, Bn, D).permute(1, 0, 2)
            if cfg["gcnii"]:
                bl = math.log(0.5 / l + 1); Z = 0.9 * KH + 0.1 * H0
                H = torch.relu((1 - bl) * Z + bl * W(Z))
            else:
                H = torch.relu(W(KH))
            Hs.append(H)
        s = self.head(torch.cat(Hs, -1) if cfg["jk"] else H).squeeze(-1)
        if return_hidden: return s, Hs
        if not want_energy: return s
        return s, self.energies(Hs, al)

    def energies(self, Hs, al):
        G = self.G; n = G.n
        dK = al * G.a_row + (1 - al) * G.r_row if self.cfg["kernel"] else G.a_row
        dm = torch.where(dK > 0, dK.clamp_min(1e-12).rsqrt(), torch.zeros_like(dK))
        sub = min(Hs[0].shape[0], 4); Eh = []
        for H in Hs:
            H = H[:sub]
            Hf = (H * dm[None, :, None]).permute(1, 0, 2).reshape(n, sub * D)
            KnH = self.prop(al, Hf).reshape(n, sub, D).permute(1, 0, 2) * dm[None, :, None]
            nrm = H.pow(2).sum((1, 2))
            Eh.append((nrm - (H * KnH).sum((1, 2))) / (nrm + EPS))
        return Eh

    def l_ps(self, rng, baseline):
        G = self.G; al = self.alpha()
        U, V, aUV, rUV = G.sample_paths(al.item(), MPATH, rng)
        num = al * torch.tensor(aUV, dtype=torch.float32) + (1 - al) * torch.tensor(rUV, dtype=torch.float32)
        den = al * G.a_row[U] + (1 - al) * G.r_row[U]
        lp = torch.log(num / den + 1e-30).reshape(T, MPATH).sum(0) - math.log(len(G.nonisol))
        f = lp.detach()
        b = f.mean().item() if baseline is None else 0.9 * baseline + 0.1 * f.mean().item()
        return ((f - b) * lp).mean() + lp.mean(), b, f.mean().item()


def relax(s, S, r, iters=40):
    mask = torch.ones_like(s, dtype=torch.bool); mask[S] = False
    sd = s.detach()
    lo, hi = float(sd[mask].min()) - 30, float(sd[mask].max()) + 30
    for _ in range(iters):
        c = 0.5 * (lo + hi)
        if torch.sigmoid(sd[mask] - c).sum() > r: lo = c
        else: hi = c
    c0 = 0.5 * (lo + hi)
    g = (torch.sigmoid(sd - c0) * (1 - torch.sigmoid(sd - c0))) * mask
    c = c0 + (g * (s - sd)).sum() / g.sum().clamp(min=1e-12)
    return torch.sigmoid(s - c) * mask


class _SpMV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, M, MT):
        ctx.MT = MT; return torch.sparse.mm(M, x.unsqueeze(1)).squeeze(1)

    @staticmethod
    def backward(ctx, g):
        return torch.sparse.mm(ctx.MT, g.contiguous().unsqueeze(1)).squeeze(1), None, None


class Coverage:
    """unsupervised objective over a fixed sample of RR sets (incidence matrix M, R x n)."""
    def __init__(self, M):
        M = sp.csr_matrix(M); self.R, self.n = M.shape; self.Mc = M.tocsc(); self.MT = self.Mc.T.tocsr()
        self.tM = _t(M); self.tMT = _t(M.T)

    def covered(self, S):
        cov = np.zeros(self.R, bool)
        for v in S: cov[self.Mc.indices[self.Mc.indptr[v]:self.Mc.indptr[v + 1]]] = True
        return cov

    def loss(self, p, S, r):
        lg = _SpMV.apply(torch.log1p(-p.clamp(max=1 - 1e-6)), self.tM, self.tMT)
        unc = torch.from_numpy(~self.covered(S))
        gain = ((1 - torch.exp(lg)) * unc).sum()
        c = self.MT @ unc.numpy().astype(np.float32)
        mask = np.ones(self.n, bool); mask[S] = False
        return -gain / (float(r) * (float(c[mask].mean()) + 1e-6))


@torch.no_grad()
def seed_aware_greedy(model, k, B=None):
    """B rounds; each takes the ceil(k/B) best-scored nodes, deferring to the next round any node
    adjacent to one already picked in the same round (picks within a round are not re-scored)."""
    n = model.G.n; B = min(k, 100) if B is None else B; per = math.ceil(k / B); S = []
    A = model.G.A; inS = np.zeros(n, bool)
    while len(S) < k:
        s = model([(S, k - len(S))])[0].clone(); s[S] = -float("inf")
        want = min(per, k - len(S)); pick = []; blocked = np.zeros(n, bool)
        for v in torch.argsort(s, descending=True).tolist():
            if len(pick) == want: break
            if inS[v] or blocked[v]: continue
            pick.append(v); blocked[A.indices[A.indptr[v]:A.indptr[v + 1]]] = True
        if not pick:                                          # every candidate blocked: fall back
            pick = [v for v in torch.argsort(s, descending=True).tolist() if not inS[v]][:want]
        S += pick; inS[pick] = True
    return S


@torch.no_grad()
def topk_unaware(model, k):
    return torch.topk(model([([], k)])[0], k).indices.tolist()


def train(G, cov, cfg, seed, kmax, log=None):
    rng = np.random.default_rng(1000 + seed); n = G.n
    model = PAGCN(G, cfg, seed)
    groups = [{"params": [p for nm, p in model.named_parameters() if nm != "a"], "lr": cfg["lr"]},
              {"params": [model.a], "lr": cfg["lr_alpha"]}]
    opt = torch.optim.Adam(groups, weight_decay=1e-5)
    beta = cfg["beta"] if (cfg["ps"] and cfg["kernel"]) else 0.0
    gamma = cfg["gamma"] if cfg["smooth"] else 0.0
    Lc = cfg.get("L", L); baseline = None; own = None; hist = []
    for ep in range(cfg["epochs"]):
        if cfg["dagger"] and ep % 20 == 0 and ep > 0:
            own = seed_aware_greedy(model, kmax, B=50)
        for st in range(cfg["steps"]):
            states = []
            for _ in range(cfg["states"]):
                m = int(rng.integers(0, kmax)); r = int(rng.integers(1, math.ceil(kmax / 100) + 1))
                S = list(own[:m]) if (own is not None and rng.random() < 0.5) else list(rng.choice(n, m, replace=False))
                states.append((S, r))
            s, Eh = model(states, want_energy=True)
            if hasattr(cov, "loss_scores"):
                lc = cov.loss_scores(s, states)
            elif hasattr(cov, "loss_batch"):
                lc = cov.loss_batch(torch.stack([relax(s[b], S, r) for b, (S, r) in enumerate(states)]), states)
            else:
                lc = sum(cov.loss(relax(s[b], S, r), S, r) for b, (S, r) in enumerate(states)) / len(states)
            lsm = sum(torch.clamp(DELTA - Eh[l + 1] / (Eh[l] + EPS), min=0) for l in range(Lc)).mean() / Lc
            loss = lc + gamma * lsm
            if beta > 0:
                lps, baseline, _ = model.l_ps(rng, baseline); loss = loss + beta * lps
            opt.zero_grad(); loss.backward(); opt.step()
        if log is not None and (ep % 25 == 0 or ep == cfg["epochs"] - 1):
            hist.append(dict(ep=ep, l_cov=float(lc), alpha=float(model.alpha())))
            log(f"  ep {ep:3d} L_cov {float(lc):.4f} L_sm {float(lsm):.3f} alpha {float(model.alpha()):.3f}")
    return model, hist


class SoftLT:
    """Unsupervised LT objective: differentiable relaxation of the LT dynamics of the protocol
    (thresholds ~ U[lo, hi], normalized in-weights 1/d_v, synchronous rounds). For a state (S, r),
    relaxed picks p raise the soft activation; the objective is the expected new activation per pick.
    Same interface as Coverage.loss(p, S, r)."""
    def __init__(self, A, samples=32, rounds=20, tau=0.05, lo=0.3, hi=0.6, seed=0):
        A = sp.csr_matrix(A, dtype=np.float64); self.n = A.shape[0]
        deg = np.asarray(A.sum(1)).ravel()
        W = (sp.diags(np.where(deg > 0, 1 / np.maximum(deg, 1), 0)) @ A).tocsr()   # row v: weights of in-nbrs
        self.tW = _t(W); self.tWT = _t(W.T)
        self.samples, self.rounds, self.tau, self.lo, self.hi = samples, rounds, tau, lo, hi
        self.rng = np.random.default_rng(seed)

    def _run(self, a0, th):
        a = a0
        for _ in range(self.rounds):
            w = _SpMMt.apply(a, self.tW, self.tWT)
            a = torch.maximum(a0, torch.sigmoid((w - th) / self.tau))
        return a.sum(0).mean()

    def loss_batch(self, P, states):
        """all states of a step in one unrolled pass: P is (B, n) relaxed picks."""
        n = self.n; B = len(states); m = self.samples
        th = torch.from_numpy(self.rng.uniform(self.lo, self.hi, (n, m)).astype(np.float32)).repeat(1, B)
        X = torch.zeros(n, B)
        for b, (S, _) in enumerate(states): X[S, b] = 1
        X = X.repeat_interleave(m, 1)
        A0 = torch.maximum(X, P.t().repeat_interleave(m, 1))
        r = torch.tensor([float(r) for _, r in states]).repeat_interleave(m)
        with torch.no_grad(): base = self._run_cols(X, th)
        gain = (self._run_cols(A0, th) - base) / r
        return -gain.mean()

    def _run_cols(self, a0, th):
        a = a0
        for _ in range(self.rounds):
            w = _SpMMt.apply(a, self.tW, self.tWT)
            a = torch.maximum(a0, torch.sigmoid((w - th) / self.tau))
        return a.sum(0)

    def loss(self, p, S, r):
        n = self.n
        th = torch.from_numpy(self.rng.uniform(self.lo, self.hi, (n, self.samples)).astype(np.float32))
        x = torch.zeros(n, 1); x[S] = 1
        with torch.no_grad(): base = self._run(x.expand(n, self.samples), th)
        a0 = torch.maximum(x, p.unsqueeze(1)).expand(n, self.samples)
        return -(self._run(a0, th) - base) / float(r)


class _SpMMt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, M, MT):
        ctx.MT = MT; return torch.sparse.mm(M, X.contiguous())

    @staticmethod
    def backward(ctx, g):
        return torch.sparse.mm(ctx.MT, g.contiguous()), None, None


class LTReward:
    """Unsupervised LT objective without relaxation: the exact (deterministic, threshold-0.5) LT spread
    gain of sampled next picks, optimized by the score-function estimator with a shared-sample
    baseline (Kool et al., ICLR'19). For a state (S, r): draw m sets of r picks from the
    Plackett-Luce distribution of the scores (Gumbel-top-r), simulate their gain, and minimize
    -mean(A * log pi(picks)), A the standardized gain."""
    def __init__(self, A, m=32, thr=0.5, steps=99, seed=0):
        A = sp.csr_matrix(A, dtype=np.float32); self.A = A; self.n = A.shape[0]
        self.deg = np.asarray(A.sum(1)).ravel(); self.m, self.thr, self.steps = m, thr, steps
        self.rng = np.random.default_rng(seed)

    def _spread(self, X):                                    # X: n x m bool seeds -> activated count per column
        deg = np.maximum(self.deg, 1)[:, None]; has = (self.deg > 0)[:, None]
        for _ in range(self.steps):
            new = X | (has & ((self.A @ X.astype(np.float32)) / deg >= self.thr))
            if (new == X).all(): break
            X = new
        return X.sum(0).astype(np.float64)

    def loss_scores(self, s, states):
        tot = 0.0; n, m = self.n, self.m
        for b, (S, r) in enumerate(states):
            lg = s[b].clone(); mask = torch.zeros(n, dtype=torch.bool); mask[S] = True
            lg = lg.masked_fill(mask, -float("inf"))
            g = torch.from_numpy(self.rng.gumbel(size=(m, n)).astype(np.float32))
            picks = torch.topk(lg.detach().unsqueeze(0) + g, r, dim=1).indices          # m x r, in sampled order
            lp = torch.zeros(m); rem = lg.unsqueeze(0).expand(m, n).clone()
            for t in range(r):
                idx = picks[:, t]
                lp = lp + rem.gather(1, idx[:, None]).squeeze(1) - torch.logsumexp(rem, 1)
                rem = rem.scatter(1, idx[:, None], -float("inf"))
            X0 = np.zeros((n, 1), bool); X0[S] = True
            base = self._spread(X0.copy())[0]
            X = np.repeat(X0, m, 1); X[picks.numpy().T, np.arange(m)[None, :]] = True
            gain = self._spread(X) - base
            adv = (gain - gain.mean()) / (gain.std() + 1e-6)
            tot = tot - (torch.from_numpy(adv.astype(np.float32)) * lp).mean()
        return tot / len(states)


# ======================================================================================================
# NAK: node-adaptive influence-horizon kernel (adaptive-horizon kernel of the paper; replaces K(alpha))
#   (K H)_v = sum_{s=1..S} w_s(v) (At^s H)_v,  w(v) = softmax(MLP(Phi_v))   (SDS only)
#   D^{-1/2} K D^{1/2} = sum_s diag(w_s) P^s is row-stochastic -> spectral radius 1 (as for K(alpha)).
#   L_PS is tied to the scalar alpha of K(alpha) and is not used.
# ======================================================================================================
S_HOPS = 4
_PAGCNBase = PAGCN


def _coo(M):
    M = sp.coo_matrix(M, dtype=np.float32)
    return torch.sparse_coo_tensor(np.vstack([M.row, M.col]), M.data, M.shape).coalesce()


class PAGCN(_PAGCNBase):
    def __init__(self, G, cfg, seed):
        super().__init__(G, cfg, seed)
        if not hasattr(G, "tA_coo"):
            G.tA_coo = _coo(G.At)
            G.zstruct = G.Phi                                   # SDS only: no degree or other centrality
        torch.manual_seed(seed + 1000)
        self.S = cfg.get("hops", S_HOPS)
        self.hop = torch.nn.Sequential(torch.nn.Linear(3, 16), torch.nn.ReLU(), torch.nn.Linear(16, self.S))
        torch.nn.init.zeros_(self.hop[2].weight); torch.nn.init.zeros_(self.hop[2].bias)

    def alpha(self):
        return torch.ones(())

    def hop_weights(self):
        return torch.softmax(self.hop(self.G.zstruct), -1)

    def prop(self, al, H):
        if not self.cfg["kernel"]: return super().prop(al, H)
        w = self.hop_weights(); A = self.G.tA_coo; X = H; out = 0
        for s in range(self.S):
            X = torch.sparse.mm(A, X); out = out + w[:, s:s + 1] * X
        return out


CFG["ps"] = False
