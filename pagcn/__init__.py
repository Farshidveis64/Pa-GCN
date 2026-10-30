"""PA-GCN: label-free learned influence maximization.

Modules
  model    core PA-GCN (structural input, adaptive-horizon kernel, GCNII encoder, objectives, seed-aware greedy)
  vn       virtual-node extension (VNGCN) -- the model used for every result of the paper
  scale    GraphLarge (preprocessing without forming A^2) and weighted-cascade RR sampling for million-edge graphs
  rr       RR sets under weighted cascade, RR greedy reference, RR-coverage spread estimator
  ndsim    evaluation protocol of the benchmark tables (IC edge probabilities, LT, SIS)
  sis_dsn  SIS with DeepSN's simulator dynamics
  synthetic  the synthetic training union used by the train-once models
  paths    repository-relative paths, graph loading

Typical use (see README):  from pagcn import use_vn; PF = use_vn()
"""
__version__ = "1.0.0"


def use_vn():
    """Make pagcn.model.train / PAGCN build the virtual-node model (as in every experiment of the paper) and return
    the pagcn.model module."""
    from . import model as PF, vn as PV
    PF.PAGCN = PV.VNGCN
    return PF
