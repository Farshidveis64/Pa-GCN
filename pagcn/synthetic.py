"""Synthetic training graphs of the train-once models: six generator families (stochastic block model with a
preferential-attachment periphery, dense modules joined by low-degree brokers, random geometric graphs,
Barabasi-Albert, Watts-Strogatz small worlds, and degree-preserving rewirings of these), three graphs per family,
200-800 nodes each, combined into one disjoint union. The RNG seed 7000 + SEED fixes the union of model seed SEED."""
import numpy as np, scipy.sparse as sp, networkx as nx


def families(rng):
    def sbm_pa():
        nb = int(rng.integers(3, 6)); sizes = [int(rng.integers(40, 120)) for _ in range(nb)]
        p = np.full((nb, nb), 0.004); np.fill_diagonal(p, rng.uniform(0.06, 0.15))
        G = nx.stochastic_block_model(sizes, p.tolist(), seed=int(rng.integers(1e9)))
        n0 = G.number_of_nodes(); per = int(rng.integers(100, 300)); deg = np.array([d for _, d in G.degree()], float) + 1
        for i in range(per):                                    # preferential-attachment periphery
            v = n0 + i; tg = rng.choice(len(deg), size=int(rng.integers(1, 3)), replace=False, p=deg / deg.sum())
            for u in tg: G.add_edge(v, int(u))
            deg = np.append(deg, len(tg)); deg[tg] += 1
        return G
    def modules():
        m = int(rng.integers(4, 9)); s = int(rng.integers(25, 60)); G = nx.Graph()
        for b in range(m):
            H = nx.gnp_random_graph(s, rng.uniform(0.25, 0.5), seed=int(rng.integers(1e9)))
            G = nx.disjoint_union(G, H)
        n0 = G.number_of_nodes()
        for b in range(m):                                      # low-degree brokers between modules
            for _ in range(int(rng.integers(1, 3))):
                c = int(rng.integers(m)); v = G.number_of_nodes(); G.add_node(v)
                G.add_edge(v, int(b * s + rng.integers(s))); G.add_edge(v, int(c * s + rng.integers(s)))
        return G
    def geometric():
        n = int(rng.integers(200, 800)); return nx.random_geometric_graph(n, np.sqrt(rng.uniform(6, 12) / (np.pi * n)), seed=int(rng.integers(1e9)))
    def ba():
        n = int(rng.integers(200, 800)); return nx.barabasi_albert_graph(n, int(rng.integers(1, 5)), seed=int(rng.integers(1e9)))
    def ws():
        n = int(rng.integers(200, 800)); return nx.watts_strogatz_graph(n, int(rng.choice([4, 6, 8])), rng.uniform(0.01, 0.2), seed=int(rng.integers(1e9)))
    base = [sbm_pa, modules, geometric, ba, ws]
    def rewired():
        G = base[int(rng.integers(len(base)))]()
        nx.double_edge_swap(G, nswap=G.number_of_edges() // 2, max_tries=10 * G.number_of_edges(), seed=int(rng.integers(1e9)))
        return G
    return base + [rewired]


def synthetic_union(seed):
    """Adjacency (csr, float64) of the training union of model seed `seed` (isolated nodes removed)."""
    rng = np.random.default_rng(7000 + seed); parts = []
    for fam in families(rng):
        for _ in range(3): parts.append(nx.convert_node_labels_to_integers(fam()))
    U = nx.disjoint_union_all(parts); U.remove_nodes_from([v for v in list(U) if U.degree(v) == 0])
    U = nx.convert_node_labels_to_integers(U)
    return sp.csr_matrix(nx.to_scipy_sparse_array(U, format="csr").astype(np.float64)), len(parts)
