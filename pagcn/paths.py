"""Repository-relative locations and graph loading. Every script resolves paths from the repository root, so the
code runs from any working directory."""
import os, numpy as np, scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")                    # <net>.npz: symmetric 0/1 adjacency (scipy.sparse)
RUNS = os.environ.get("PAGCN_RUNS", os.path.join(ROOT, "results", "runs"))   # new runs write here
CKPT = os.environ.get("PAGCN_CKPT", os.path.join(ROOT, "checkpoints"))      # models trained by the experiment scripts
MODELS = CKPT                                        # trained models are read from here


GENERATED = os.path.join(DATA, "generated")         # optional extra graphs


def graph_path(net):
    p = os.path.join(DATA, f"{net}.npz")
    return p if os.path.exists(p) else os.path.join(GENERATED, f"{net}.npz")


def load_graph(net):
    """Adjacency of a dataset in data/ (float64 csr)."""
    return sp.load_npz(graph_path(net)).astype(np.float64).tocsr()


def run_file(name):
    """Output file for a new run: $OUT if set, else results/runs/<name>."""
    p = os.environ.get("OUT") or os.path.join(RUNS, name)
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    return p


def ckpt_dir(sub):
    d = os.path.join(CKPT, sub); os.makedirs(d, exist_ok=True); return d
