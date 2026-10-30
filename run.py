"""Run PA-GCN: select k seed nodes on a network and evaluate their spread (IC, LT, SIS).

  python run.py --net jazz --k 10 --train IC                     # train on the network, then select seeds
  python run.py --net jazz --k 10 --model checkpoints/synthetic/synth_s0.pt   # use a model trained by train.py
  python run.py --edges my_graph.txt --k 50 --train IC           # your own graph ("u v" per line, undirected)
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, scipy.sparse as sp, torch
from pagcn import use_vn
from pagcn.ndsim import rr_sets_ic, ic_edge_probs, lt_spread
from pagcn.sis_dsn import sis_dsn
from pagcn.rr import spread, greedy_rr
from pagcn.paths import load_graph


def read_edges(path):
    e = np.loadtxt(path, dtype=np.int64, comments="#")[:, :2]
    ids, e = np.unique(e, return_inverse=True); e = e.reshape(-1, 2); n = len(ids)
    A = sp.csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float64).tocsr(); A.setdiag(0); A.eliminate_zeros(); return A


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--net"); g.add_argument("--edges")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--model", default=None, help="checkpoint (state_dict) of a PA-GCN; default checkpoints/synthetic/synth_s0.pt")
    ap.add_argument("--train", choices=["IC", "LT"], default=None, help="train a model on the target instead of loading one")
    ap.add_argument("--epochs", type=int, default=200); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None, help="where to save a trained model")
    ap.add_argument("--no-refs", action="store_true")
    a = ap.parse_args(); PF = use_vn()
    A = load_graph(a.net) if a.net else read_edges(a.edges); n = A.shape[0]
    print(f"graph: n={n} m={A.nnz // 2}", flush=True)
    t = time.time(); G = PF.Graph(A); print(f"structural preprocessing {time.time() - t:.2f}s", flush=True)
    if a.train:
        obj = PF.Coverage(rr_sets_ic(A, 200000, 1)) if a.train == "IC" else PF.LTReward(A)
        t = time.time(); m, _ = PF.train(G, obj, dict(PF.CFG, epochs=a.epochs), a.seed, max(a.k, int(round(0.2 * n))), log=print)
        print(f"trained in {time.time() - t:.0f}s", flush=True)
        if a.save: torch.save(m.state_dict(), a.save)
    else:
        ck = a.model or os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "synthetic", "synth_s0.pt")
        m = PF.PAGCN(G, PF.CFG, a.seed); m.load_state_dict(torch.load(ck, map_location="cpu")); print("loaded", os.path.relpath(ck))
    m.eval()
    t = time.time()
    with torch.no_grad(): S = PF.seed_aware_greedy(m, a.k)
    print(f"selected k={a.k} seeds in {time.time() - t:.2f}s: {sorted(S)}", flush=True)
    E = rr_sets_ic(A, 200000, 999, ic_edge_probs(A)).tocsc()
    def score(name, X):
        print(f"{name:12s} IC {spread(E, X, n):6.2f}%   LT {lt_spread(A, X):6.2f}%   SIS(DeepSN sim.) {sis_dsn(A, X, 1000 if n < 20000 else 200):6.2f}%", flush=True)
    score("PA-GCN", S)
    if not a.no_refs:
        deg = np.asarray(A.sum(1)).ravel(); score("Degree", [int(x) for x in np.argsort(-deg, kind="stable")[:a.k]])
        score("RR greedy", greedy_rr(rr_sets_ic(A, 200000, 1, ic_edge_probs(A)), a.k, n))
