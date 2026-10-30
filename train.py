"""Train PA-GCN once on synthetic graphs; the model can then be applied to any network with run.py --model.

  python train.py --diff IC --seed 0        # -> checkpoints/synthetic/synth_s0.pt
  python train.py --diff LT --seed 0        # -> checkpoints/synthetic/synth_LT_s0.pt
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from pagcn import use_vn, scale as PS
from pagcn.synthetic import synthetic_union
from pagcn.paths import ckpt_dir

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diff", choices=["IC", "LT"], required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt", default=None); ap.add_argument("--epochs", type=int, default=None, help="override (testing only)")
    a = ap.parse_args(); PF = use_vn()
    d = a.ckpt or ckpt_dir("synthetic"); os.makedirs(d, exist_ok=True)
    ck = os.path.join(d, f"synth_s{a.seed}" if a.diff == "IC" else f"synth_LT_s{a.seed}")
    cfg = dict(PF.CFG, **({"epochs": a.epochs} if a.epochs else {}))
    A, ngraphs = synthetic_union(a.seed); G = PF.Graph(A); n = G.n
    if a.diff == "IC":
        t = time.time(); obj = PF.Coverage(PS.rr_sets(A, 200000, 1)); t_cov = time.time() - t
    else:
        obj = PF.LTReward(A); t_cov = 0.0
    t = time.time(); model, _ = PF.train(G, obj, cfg, a.seed, int(round(0.2 * n))); t_tr = time.time() - t
    torch.save(model.state_dict(), ck + ".pt")
    meta = dict(n=n, m=int(A.nnz // 2), graphs=ngraphs, rr_s=t_cov, train_s=t_tr, alpha=float(model.alpha()))
    json.dump(meta, open(ck + ".json", "w")); print("trained", ck + ".pt", meta, flush=True)
