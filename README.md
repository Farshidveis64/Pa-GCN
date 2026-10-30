# PA-GCN

Official implementation of **PA-GCN**, a label-free graph neural network for influence maximization.

PA-GCN selects seed nodes that maximize the expected spread of influence under the Independent Cascade (IC),
Linear Threshold (LT) and SIS diffusion models. It uses only structural node information, propagates messages with
an adaptive-horizon kernel, and is trained without labels. A model trained once on small synthetic graphs can be
applied unchanged to unseen networks with millions of edges.

## Requirements

- Python >= 3.10
- PyTorch, NumPy, SciPy, NetworkX (see `requirements.txt`)

```bash
pip install -r requirements.txt
```

## Repository structure

```
pagcn/          core library (model, kernel, training, seed selection, diffusion evaluation)
run.py          select seeds on a network and evaluate their spread
train.py        train PA-GCN once on synthetic graphs
data/           benchmark networks (.npz) and their sources (data/README.md)
```

## Usage

Train a model on a network and select seeds:

```bash
python run.py --net jazz --k 10 --train IC
```

Train once on synthetic graphs, then apply the model to another network:

```bash
python train.py --diff IC --seed 0
python run.py --net jazz --k 10 --model checkpoints/synthetic/synth_s0.pt
```

Use your own graph (one undirected edge `u v` per line):

```bash
python run.py --edges my_graph.txt --k 50 --train IC
```

Run `python run.py --help` or `python train.py --help` for all options.

## Data

The networks are publicly available; sources and preprocessing are listed in `data/README.md`.

## Citation

If you use this code, please cite:

```bibtex
@article{pagcn2026,
  title   = {PA-GCN},
  author  = {Anonymous},
  journal = {IEEE Transactions on Network Science and Engineering},
  year    = {2026}
}
```

## License

This project is released under the MIT License (see `LICENSE`).
