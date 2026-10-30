# Datasets

All graphs are stored as `scipy.sparse` `.npz` files: symmetric 0/1 adjacency, no self-loops, nodes `0..n-1`.
Load with `pagcn.paths.load_graph(name)` or `scipy.sparse.load_npz("data/<name>.npz")`. Total size ~22 MB.
`n`, `m` = nodes and undirected edges; LCC = fraction of nodes in the largest connected component; D90 = effective
diameter (90th percentile of shortest-path lengths in the LCC; sampled for the three largest graphs).

| file | type | n | m | comp. | LCC | D90 | original source |
|---|---|---:|---:|---:|---:|---:|---|
| `jazz.npz` | real | 198 | 2,742 | 1 | 1.00 | 3 | Jazz musicians (Gleiser & Danon 2003); DeepIM release |
| `netscience.npz` | real | 1,589 | 2,742 | 396 | 0.24 | 9 | Network Science co-authorship (Newman 2006); DeepIM release |
| `cora_ml.npz` | real | 2,810 | 7,981 | 1 | 1.00 | 7 | Cora-ML citations (Bojchevski & Günnemann 2018); DeepIM release |
| `power_grid.npz` | real | 4,941 | 6,594 | 1 | 1.00 | 27 | US power grid (Watts & Strogatz 1998); DeepIM release |
| `random5.npz` | synthetic | 50,000 | 250,000 | 3 | 1.00 | 6 | Erdős–Rényi graph of the DeepIM release |
| `er5k.npz` | synthetic | 5,000 | 25,173 | 1 | 1.00 | – | `nx.fast_gnp_random_graph(5000, 10/4999, seed=11)` |
| `celegans.npz` | real | 297 | 2,148 | 1 | 1.00 | 3 | C. elegans neural network (Watts & Strogatz 1998), Network Repository |
| `email_eu.npz` | real | 986 | 16,064 | 1 | 1.00 | 3 | SNAP `email-Eu-core` |
| `euroroad.npz` | real | 1,174 | 1,417 | 26 | 0.89 | 32 | Network Repository `inf-euroroad` (KONECT `subelj_euroroad`) |
| `yeast.npz` | real | 1,458 | 1,948 | 1 | 1.00 | 9 | Network Repository `bio-yeast` |
| `facebook.npz` | real | 4,039 | 88,234 | 1 | 1.00 | 5 | SNAP `ego-Facebook` (`facebook_combined.txt`) |
| `grqc.npz` | real | 5,242 | 14,484 | 355 | 0.79 | 8 | SNAP `ca-GrQc` |
| `dblp.npz` | real | 317,080 | 1,049,866 | 1 | 1.00 | 9 | SNAP `com-DBLP` (`com-dblp.ungraph.txt`) |
| `youtube.npz` | real | 1,134,890 | 2,987,624 | 1 | 1.00 | 7 | SNAP `com-Youtube` (`com-youtube.ungraph.txt`) |

Sources: SNAP <https://snap.stanford.edu/data/>, Network Repository <https://networkrepository.com/>,
KONECT <http://konect.cc/>, DeepIM release <https://github.com/triplej0079/DeepIM> (`data/*.SG`, key `adj`).

Preprocessing: edge lists are read as undirected simple graphs (self-loops and duplicate edges
removed, directions ignored), nodes relabeled consecutively; nothing else is filtered (disconnected components are kept).
The graphs are redistributed here for reproducibility under the terms of their sources; please cite the original
sources when using them.
