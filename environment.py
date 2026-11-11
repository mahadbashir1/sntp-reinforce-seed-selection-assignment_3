import numpy as np
import pandas as pd
import networkx as nx


def load_graph(data_dir):
    nodes = pd.read_csv(f"{data_dir}/nodes.csv")
    edges = pd.read_csv(f"{data_dir}/edges.csv")
    departments = pd.read_csv(f"{data_dir}/departments.csv")

    graph = nx.Graph()
    graph.add_nodes_from(nodes["node_id"].tolist())  # add all 92 nodes first
    for u, v in zip(edges["source"], edges["target"]):
        if u != v:
            graph.add_edge(int(u), int(v))

    dept = dict(zip(departments["node_id"], departments["department"]))
    return graph, nodes, dept


def make_arcs(graph):
    # every undirected edge becomes two directed arcs, sorted by (source, target)
    arcs = []
    for u, v in graph.edges():
        arcs.append((u, v))
        arcs.append((v, u))
    arcs.sort()
    return np.array(arcs, dtype=int)


def independent_cascade(arcs, num_nodes, seeds, diffusion_seed, p=0.10):
    seeds = set(int(s) for s in seeds)
    if len(seeds) == 0:
        return 0

    # one uniform draw per arc in the fixed arc order; arc is live if draw < p
    rng = np.random.default_rng(diffusion_seed)
    draws = rng.random(len(arcs))
    live_arcs = arcs[draws < p]

    live_neighbors = [[] for _ in range(num_nodes)]
    for u, v in live_arcs:
        live_neighbors[u].append(v)

    # spread = all nodes reachable from the seeds using live arcs (BFS)
    active = set(seeds)
    frontier = list(seeds)
    while frontier:
        new_nodes = []
        for u in frontier:
            for v in live_neighbors[u]:
                if v not in active:
                    active.add(v)
                    new_nodes.append(v)
        frontier = new_nodes
    return len(active)


class SeedSelectionEnv:
    def __init__(self, graph, budget=5, p=0.10):
        self.n = graph.number_of_nodes()
        self.budget = budget
        self.p = p
        self.arcs = make_arcs(graph)

        # static features (computed once)
        self.adj = nx.to_numpy_array(graph, nodelist=range(self.n))
        self.degree = self.adj.sum(axis=1)
        self.degree_centrality = self.degree / (self.n - 1)
        clustering = nx.clustering(graph)
        self.clustering = np.array([clustering[i] for i in range(self.n)])

        self.selected = []
        self.done = True

    def reset(self, diffusion_seed):
        self.diffusion_seed = diffusion_seed
        self.selected = []
        self.done = False
        return self.get_observation()

    def get_observation(self):
        selected = np.zeros(self.n)
        selected[self.selected] = 1.0
        selected_neighbors = self.adj @ selected
        neighbor_fraction = np.zeros(self.n)
        has_edges = self.degree > 0
        neighbor_fraction[has_edges] = selected_neighbors[has_edges] / self.degree[has_edges]
        remaining = np.full(self.n, (self.budget - len(self.selected)) / self.budget)

        features = np.stack([self.degree_centrality, self.clustering, selected,
                             neighbor_fraction, remaining], axis=1).astype(np.float32)
        mask = selected == 0  # True = can still be selected
        return features, mask

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is finished, call reset() first")
        if not isinstance(action, (int, np.integer)) or action < 0 or action >= self.n:
            raise ValueError(f"Invalid action {action}")
        if action in self.selected:
            raise ValueError(f"Node {action} is already selected")

        self.selected.append(int(action))
        features, mask = self.get_observation()

        reward = 0.0
        spread = 0
        if len(self.selected) == self.budget:
            # diffusion only runs after the fifth selection
            spread = independent_cascade(self.arcs, self.n, self.selected, self.diffusion_seed, self.p)
            reward = spread / self.n
            self.done = True
        return features, mask, reward, self.done, {"spread": spread}
