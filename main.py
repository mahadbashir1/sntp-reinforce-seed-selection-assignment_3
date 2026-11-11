import argparse
import json
import os
import time

import torch  # imported before pandas, otherwise torch DLL fails to load on my Windows setup
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from environment import load_graph, independent_cascade, SeedSelectionEnv
from models import MLPPolicy, TransformerPolicy, masked_distribution, greedy_action
from reinforce import train

TRAIN_SEEDS = [7, 21, 42]
EVAL_SEEDS = list(range(900000, 900100))
POLICIES = {"reinforce_mlp": MLPPolicy, "reinforce_transformer": TransformerPolicy}


def network_analysis(graph, dept):
    centrality = nx.degree_centrality(graph)
    clustering = nx.clustering(graph)
    components = list(nx.connected_components(graph))
    degrees = [d for _, d in graph.degree()]
    top10 = sorted(graph.nodes(), key=lambda v: (-graph.degree(v), v))[:10]
    return {
        "participants": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "connected_components": len(components),
        "largest_component_size": max(len(c) for c in components),
        "isolated_nodes": nx.number_of_isolates(graph),
        "mean_degree": float(np.mean(degrees)),
        "min_degree": int(min(degrees)),
        "max_degree": int(max(degrees)),
        "mean_degree_centrality": float(np.mean(list(centrality.values()))),
        "mean_clustering": float(np.mean(list(clustering.values()))),
        "min_clustering": float(min(clustering.values())),
        "max_clustering": float(max(clustering.values())),
        "department_sizes": pd.Series(dept).value_counts().to_dict(),
        "top10_degree": [{"node_id": v, "degree": graph.degree(v), "degree_centrality": centrality[v],
                          "clustering": clustering[v], "department": dept[v]} for v in top10],
        "degree_centrality": [centrality[v] for v in range(graph.number_of_nodes())],
        "clustering": [clustering[v] for v in range(graph.number_of_nodes())],
    }


def run_checks(graph, env):
    arcs, n = env.arcs, env.n
    checks = {}
    checks["empty_seed_set_zero_spread"] = independent_cascade(arcs, n, [], 1) == 0
    checks["probability_0_only_distinct_seeds"] = independent_cascade(arcs, n, [3, 3, 10, 50], 1, p=0.0) == 3
    reachable = set()
    for s in [3, 10, 50]:
        reachable |= nx.node_connected_component(graph, s)
    checks["probability_1_all_reachable_active"] = independent_cascade(arcs, n, [3, 10, 50], 1, p=1.0) == len(reachable)
    checks["same_seed_same_spread"] = all(
        independent_cascade(arcs, n, [0, 1, 2, 3, 4], s) == independent_cascade(arcs, n, [0, 1, 2, 3, 4], s)
        for s in range(50))

    env.reset(0)
    env.step(5)
    try:
        env.step(5)
        checks["duplicate_action_rejected"] = False
    except ValueError:
        checks["duplicate_action_rejected"] = True

    # masked (already selected) participants must get probability 0
    zero_prob = True
    for policy_class in POLICIES.values():
        policy = policy_class()
        features, mask = env.reset(0)
        for action in [5, 17, 40]:
            features, mask, _, _, _ = env.step(action)
        with torch.no_grad():
            probs = masked_distribution(policy(torch.from_numpy(features)), torch.from_numpy(mask)).probs
        if not (probs[~torch.from_numpy(mask)] == 0).all() or abs(float(probs.sum()) - 1) > 1e-5:
            zero_prob = False
    checks["masked_actions_zero_probability"] = zero_prob
    return checks


def evaluate_baseline(env, method, diffusion_seed):
    start = time.perf_counter()
    if method == "random":
        rng = np.random.default_rng(diffusion_seed + 10000000)
        nodes = [int(v) for v in rng.choice(env.n, size=5, replace=False)]
    else:  # degree, ties broken by smaller node ID
        nodes = sorted(range(env.n), key=lambda v: (-env.degree[v], v))[:5]
    seconds = time.perf_counter() - start

    env.reset(diffusion_seed)
    for v in nodes:
        _, _, reward, _, info = env.step(v)
    return nodes, info["spread"], reward, seconds


def evaluate_policy(env, policy, diffusion_seed):
    features, mask = env.reset(diffusion_seed)
    seconds = 0.0
    done = False
    while not done:
        start = time.perf_counter()
        action = greedy_action(policy, features, mask)
        seconds += time.perf_counter() - start  # only selection time, diffusion not included
        features, mask, reward, done, info = env.step(action)
    return list(env.selected), info["spread"], reward, seconds


def draw_network(graph, dept, pos, path, title, selected=None):
    departments = sorted(set(dept.values()))
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(9, 7))
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.12, width=0.6)
    for i, d in enumerate(departments):
        nodes = [v for v in graph.nodes() if dept[v] == d]
        sizes = [25 + 8 * graph.degree(v) for v in nodes]
        nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_size=sizes, node_color=[colors[i]],
                               alpha=0.9 if selected is None else 0.45, label=d, ax=ax)
    if selected is None:
        nx.draw_networkx_labels(graph, pos, font_size=6, ax=ax)
    else:
        nx.draw_networkx_nodes(graph, pos, nodelist=selected, node_size=420, node_color="none",
                               edgecolors="red", linewidths=2.5, ax=ax)
        labels = {v: f"{v} (#{i + 1})" for i, v in enumerate(selected)}
        label_pos = {v: (pos[v][0], pos[v][1] + 0.07) for v in selected}  # text just above the circle
        nx.draw_networkx_labels(graph, label_pos, labels=labels, font_size=9, font_weight="bold", ax=ax)
    ax.legend(title="Department", loc="upper left", fontsize=8)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results")
    parser.add_argument("--data", default="data")
    args = parser.parse_args()
    out = args.output
    os.makedirs(f"{out}/checkpoints", exist_ok=True)
    os.makedirs(f"{out}/figures", exist_ok=True)

    # 1. dataset and network analysis
    graph, nodes, dept = load_graph(args.data)
    env = SeedSelectionEnv(graph, budget=5, p=0.10)
    n = env.n
    edge_list = sorted((min(u, v), max(u, v)) for u, v in graph.edges())
    pd.DataFrame(edge_list, columns=["source", "target"]).to_csv(f"{out}/graph_edges.csv", index=False)
    stats = network_analysis(graph, dept)
    print(f"Nodes: {stats['participants']}, Edges: {stats['edges']}, Density: {stats['density']:.4f}, "
          f"Components: {stats['connected_components']}, Mean clustering: {stats['mean_clustering']:.4f}")
    print("Top-10 degree centrality:")
    for row in stats["top10_degree"]:
        print(f"  node {row['node_id']:2d}  degree {row['degree']:2d}  centrality {row['degree_centrality']:.4f}"
              f"  clustering {row['clustering']:.4f}  {row['department']}")

    pos = nx.spring_layout(graph, seed=7, k=0.35, iterations=200)
    draw_network(graph, dept, pos, f"{out}/figures/network.png",
                 "Workplace contact network (node size = degree)")

    # 2. simulator and environment checks
    checks = run_checks(graph, env)

    # 3. training (2 architectures x 3 seeds)
    training_rows = []
    training_seconds = {}
    all_finite = True
    for method, policy_class in POLICIES.items():
        for seed in TRAIN_SEEDS:
            policy, history, seconds, finite = train(policy_class, env, seed)
            all_finite = all_finite and finite
            training_seconds[(method, seed)] = seconds
            for episode, spread, reward in history:
                training_rows.append({"method": method, "train_seed": seed, "episode": episode,
                                      "final_spread": spread, "total_reward": reward})
            torch.save({"method": method, "train_seed": seed, "episodes": 1000, "config": policy.config,
                        "state_dict": policy.state_dict()}, f"{out}/checkpoints/{method}_seed{seed}.pt")
            last = np.mean([h[1] for h in history[-100:]])
            print(f"Trained {method} seed {seed} in {seconds:.1f}s (mean spread last 100 episodes: {last:.2f})")
    training_df = pd.DataFrame(training_rows)
    training_df.to_csv(f"{out}/training.csv", index=False)

    # 4. evaluation on the same 100 diffusion seeds
    eval_rows = []
    for method in ["random", "degree"]:
        for ds in EVAL_SEEDS:
            selected, spread, reward, seconds = evaluate_baseline(env, method, ds)
            eval_rows.append({"method": method, "train_seed": -1, "diffusion_seed": ds,
                              "selected_nodes": json.dumps(selected), "final_spread": spread,
                              "total_reward": reward, "selection_seconds": seconds})
    for method, policy_class in POLICIES.items():
        for seed in TRAIN_SEEDS:
            checkpoint = torch.load(f"{out}/checkpoints/{method}_seed{seed}.pt")
            policy = policy_class()
            policy.load_state_dict(checkpoint["state_dict"])
            policy.eval()
            for ds in EVAL_SEEDS:
                selected, spread, reward, seconds = evaluate_policy(env, policy, ds)
                eval_rows.append({"method": method, "train_seed": seed, "diffusion_seed": ds,
                                  "selected_nodes": json.dumps(selected), "final_spread": spread,
                                  "total_reward": reward, "selection_seconds": seconds})
    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(f"{out}/evaluation.csv", index=False)

    summary_rows = []
    for (method, seed), group in eval_df.groupby(["method", "train_seed"], sort=False):
        summary_rows.append({"method": method, "train_seed": seed,
                             "spread_mean": group["final_spread"].mean(),
                             "spread_std": group["final_spread"].std(ddof=0),
                             "reward_mean": group["total_reward"].mean(),
                             "selection_seconds_mean": group["selection_seconds"].mean(),
                             "training_seconds": training_seconds.get((method, seed), 0.0)})
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{out}/summary.csv", index=False)
    print(summary_df.to_string(index=False))

    # remaining checks that use the evaluation and training results
    selections = [json.loads(s) for s in eval_df["selected_nodes"]]
    checks["five_distinct_selected_users"] = all(len(set(s)) == 5 for s in selections)
    checks["reward_equals_spread_over_n"] = bool(
        np.allclose(eval_df["total_reward"], eval_df["final_spread"] / n)
        and np.allclose(training_df["total_reward"], training_df["final_spread"] / n))
    checks["finite_training_losses_and_gradients"] = all_finite
    checks = {k: "pass" if v else "fail" for k, v in checks.items()}
    with open(f"{out}/checks.json", "w") as f:
        json.dump(checks, f, indent=2)
    print(json.dumps(checks, indent=2))

    # 5. architecture level results, selected users and department coverage
    architecture = {}
    for method in POLICIES:
        run_means = summary_df[summary_df["method"] == method]["spread_mean"].values
        architecture[method] = {"run_mean_spreads": run_means.tolist(),
                                "mean": float(np.mean(run_means)), "std": float(np.std(run_means))}
    selected_users = []
    for (method, seed), group in eval_df.groupby(["method", "train_seed"], sort=False):
        sets = [json.loads(s) for s in group["selected_nodes"]]
        coverage = [len(set(dept[v] for v in s)) for s in sets]
        row = {"method": method, "train_seed": int(seed), "mean_department_coverage": float(np.mean(coverage))}
        if method != "random":  # degree and greedy policies pick the same set in every episode
            row["selected_nodes"] = sets[0]
            row["departments"] = [dept[v] for v in sets[0]]
            row["degrees"] = [graph.degree(v) for v in sets[0]]
            row["same_set_all_episodes"] = all(s == sets[0] for s in sets)
            # overlap of audiences: seeds + their direct neighbors, and edges between the seeds
            reach = set(sets[0])
            for v in sets[0]:
                reach |= set(graph[v])
            row["one_hop_reach"] = len(reach)
            row["edges_among_selected"] = graph.subgraph(sets[0]).number_of_edges()
        selected_users.append(row)
        print(row)

    with open(f"{out}/analysis.json", "w") as f:
        json.dump({"network": stats, "architecture_summary": architecture,
                   "selected_users": selected_users}, f, indent=2)

    # 6. figures
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, method in zip(axes, POLICIES):
        for seed in TRAIN_SEEDS:
            run = training_df[(training_df["method"] == method) & (training_df["train_seed"] == seed)]
            block_means = run.groupby(run["episode"] // 20)["final_spread"].mean()
            ax.plot((block_means.index + 1) * 20, block_means.values, marker=".", label=f"seed {seed}")
        ax.set_title(method)
        ax.set_xlabel("Training episode (blocks of 20)")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Mean final spread per block")
    fig.tight_layout()
    fig.savefig(f"{out}/figures/learning_curves.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    groups = list(eval_df.groupby(["method", "train_seed"], sort=False))
    labels = ["random", "degree"] + [f"{m.split('_')[1]}\nseed {s}" for (m, s), _ in groups[2:]]
    ax.boxplot([g["final_spread"].values for _, g in groups], showmeans=True)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Final spread (100 evaluation seeds)")
    ax.set_title("Final spread comparison")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{out}/figures/spread_comparison.png", dpi=200)
    plt.close(fig)

    transformer_seed7 = [r for r in selected_users if r["method"] == "reinforce_transformer" and r["train_seed"] == 7][0]
    draw_network(graph, dept, pos, f"{out}/figures/transformer_seed7_selection.png",
                 "Transformer (seed 7) selected users, red circles, #k = selection order",
                 selected=transformer_seed7["selected_nodes"])
    print("Done. Results saved in", out)


if __name__ == "__main__":
    main()
