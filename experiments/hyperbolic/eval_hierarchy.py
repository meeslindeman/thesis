"""
Hierarchical evaluation of ViT models.

Metrics follow the evaluation protocol of:

  HyCoCLIP / Wanyan et al. — which uses the framework of
  Kosmopoulos et al. (2015) "Evaluation measures for hierarchical
  classification: a unified view and novel approaches", DMKD.

For a pair (ŷ, y) of predicted and true class:

  TIE  ↓  Tree-Induced Error (Dekel et al., 2004): path distance between
           ŷ and y in the class graph.  With unit edge weights this equals
           depth(ŷ) + depth(y) - 2·depth(LCA(ŷ,y)).  Zero for a correct
           prediction.

  LCA  ↓  LCA error: height of the LCA node (length of the longest path
           downward from LCA to any leaf in its subtree).  Zero for a
           correct prediction (LCA = the leaf itself).

  J    ↑  Jaccard similarity of ancestor sets (ancestors include the node
           itself): |Ŷanc ∩ Yanc| / |Ŷanc ∪ Yanc|.

  P_H  ↑  Hierarchical precision: |Ŷanc ∩ Yanc| / |Ŷanc|.

  R_H  ↑  Hierarchical recall:    |Ŷanc ∩ Yanc| / |Yanc|.

All five are averaged over all predictions (correct predictions contribute
TIE=0, LCA=0, J=P_H=R_H=1).

CIFAR-100    : uses hierarchy_cfiar100.json (node ids 0-99 = class indices).
Tiny ImageNet: uses hierarchy_tiny_imagenet.json (pre-built from WordNet via
               NLTK using the primary hypernym path → proper tree).

Usage (run from repo root with the hvit conda env):
  python experiments/hyperbolic/eval_hierarchy.py --dataset cifar100
  python experiments/hyperbolic/eval_hierarchy.py --dataset tiny_imagenet
"""

import argparse
import sys
from dataclasses import fields
from pathlib import Path

import json
import numpy as np
import networkx as nx
import torch

# ── repo path setup ───────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "main"))

from config_base import Config
from data_utils import get_dataset, get_dataloader

# ── model builder (mirrors train.py) ─────────────────────────────────────────
def build_model(model_name: str, cfg):
    from dataclasses import asdict
    if model_name == "euclidean":
        from vit.model import ViT
        return ViT(asdict(cfg))
    elif model_name == "hyperbolic":
        from vit_hyperbolic.lorentz import Lorentz
        from vit_hyperbolic.model import LorentzViT
        model_cfg = asdict(cfg)
        model_cfg["manifold"] = Lorentz(cfg.curvature)
        return LorentzViT(model_cfg)
    elif model_name == "hypercore":
        from vit_hypercore.model import LViT
        from vit_hypercore.manifold.lorentz import Lorentz
        manifold = Lorentz(cfg.curvature)
        return LViT(
            manifold_in=manifold, manifold_hidden=manifold, manifold_out=manifold,
            image_size=cfg.image_size, patch_size=cfg.patch_size,
            num_layers=cfg.num_hidden_layers, in_channel=cfg.num_channels,
            hidden_channel=cfg.hidden_size // cfg.num_attention_heads,
            out_channel=cfg.num_classes, mlp_hidden_size=cfg.mlp_dim + 1,
            num_heads=cfg.num_attention_heads, dropout=cfg.dropout_rate,
        )
    elif model_name == "hexformer":
        from vit_hexformer.model import build_hexformer
        return build_hexformer(cfg)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def cfg_from_checkpoint(path: str) -> Config:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "config" not in state:
        raise RuntimeError(f"No 'config' key in {path}.")
    init_names = {f.name for f in fields(Config) if f.init}
    return Config(**{k: v for k, v in state["config"].items() if k in init_names})


# ── hierarchy loading ─────────────────────────────────────────────────────────
def load_tree(json_path: Path, num_classes: int) -> tuple[nx.DiGraph, list, int]:
    data = json.loads(json_path.read_text())
    tree = nx.DiGraph()
    for node in data["nodes"]:
        tree.add_node(node["id"], label=node["label"])
    for link in data["links"]:
        tree.add_edge(link["source"], link["target"])
    roots = [n for n in tree.nodes if tree.in_degree(n) == 0]
    assert len(roots) == 1, f"Expected single root, got {roots}"
    if "class_to_node" in data:
        c2n = {int(k): v for k, v in data["class_to_node"].items()}
    else:
        c2n = {i: i for i in range(num_classes)}
    return tree, [c2n[i] for i in range(num_classes)], roots[0]


# ── precomputed tree properties ───────────────────────────────────────────────
def compute_node_depths(tree: nx.DiGraph, root) -> dict:
    return nx.single_source_shortest_path_length(tree, root)


def compute_node_heights(tree: nx.DiGraph, root) -> dict:
    """Height = length of longest downward path to any leaf (leaves = 0)."""
    heights = {}
    for node in nx.dfs_postorder_nodes(tree, root):
        children = list(tree.successors(node))
        heights[node] = 0 if not children else 1 + max(heights[c] for c in children)
    return heights


def compute_ancestor_sets(tree: nx.DiGraph, class_nodes: list) -> list[frozenset]:
    """
    For each class, returns the set of its ancestors including itself.
    Including self ensures J=P_H=R_H=1 for correct predictions and that
    wrong-but-same-superclass predictions score strictly below 1.
    """
    return [frozenset(nx.ancestors(tree, n) | {n}) for n in class_nodes]


def get_lca(tree, u, v, depths):
    anc_u = nx.ancestors(tree, u) | {u}
    anc_v = nx.ancestors(tree, v) | {v}
    return max(anc_u & anc_v, key=lambda n: depths[n])


def precompute_pairwise(tree, root, class_nodes):
    """
    Precomputes for every class pair (i, j):
      tie_matrix[i,j]  = TIE  (path distance, 0 on diagonal)
      lca_h_matrix[i,j] = LCA height (0 on diagonal)
    Returns (tie_matrix, lca_h_matrix, ancestor_sets).
    """
    depths = compute_node_depths(tree, root)
    heights = compute_node_heights(tree, root)
    leaf_depths = np.array([depths[n] for n in class_nodes])

    n = len(class_nodes)
    tie_mat = np.zeros((n, n), dtype=np.int32)
    lca_h_mat = np.zeros((n, n), dtype=np.int32)

    for i in range(n):
        for j in range(i + 1, n):
            lca = get_lca(tree, class_nodes[i], class_nodes[j], depths)
            # TIE = path distance = depth(i) + depth(j) - 2*depth(lca)
            tie = int(leaf_depths[i] + leaf_depths[j] - 2 * depths[lca])
            h = heights[lca]
            tie_mat[i, j] = tie_mat[j, i] = tie
            lca_h_mat[i, j] = lca_h_mat[j, i] = h
    # diagonal stays 0: correct prediction → distance = 0, LCA = leaf → height = 0

    ancestor_sets = compute_ancestor_sets(tree, class_nodes)
    return tie_mat, lca_h_mat, ancestor_sets, heights[root]


# ── inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        all_preds.append(model(images).argmax(dim=1).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


# ── metrics (Kosmopoulos et al. 2015; Dekel et al. 2004) ─────────────────────
def compute_metrics(preds, labels, tie_mat, lca_h_mat, ancestor_sets):
    """
    TIE ↓, LCA ↓, J ↑, P_H ↑, R_H ↑ averaged over all predictions.
    Correct predictions contribute TIE=0, LCA=0, J=P_H=R_H=1.
    """
    top1_acc = float((preds == labels).mean())

    tie_vals = tie_mat[preds, labels]
    lca_vals = lca_h_mat[preds, labels]

    j_vals, ph_vals, rh_vals = [], [], []
    for p, y in zip(preds, labels):
        ap, ay = ancestor_sets[p], ancestor_sets[y]
        inter = len(ap & ay)
        j_vals.append(inter / len(ap | ay))
        ph_vals.append(inter / len(ap))
        rh_vals.append(inter / len(ay))

    return {
        "top1_acc":  top1_acc,
        "TIE":  float(tie_vals.mean()),
        "LCA":  float(lca_vals.mean()),
        "J":    float(np.mean(j_vals)),
        "P_H":  float(np.mean(ph_vals)),
        "R_H":  float(np.mean(rh_vals)),
    }


# ── main ──────────────────────────────────────────────────────────────────────
CHECKPOINTS = {
    "cifar100": {
        "euclidean":  "checkpoints_final/advanced/cifar100/euclidean_base/euclidean_cifar100.pt",
        "hyperbolic": "checkpoints_final/advanced/cifar100/hyperbolic_base/hyperbolic_cifar100.pt",
        "hypercore":  "checkpoints_final/advanced/cifar100/hypercore_base/hypercore_cifar100.pt",
        "hexformer":  "checkpoints_final/advanced/cifar100/hexformer_base/hexformer_cifar100.pt",
    },
    "tiny_imagenet": {
        "euclidean":  "checkpoints_final/advanced/tiny_imagenet/euclidean_base/euclidean_tiny_imagenet.pt",
        "hyperbolic": "checkpoints_final/advanced/tiny_imagenet/hyperbolic_base/hyperbolic_tiny_imagenet.pt",
        "hypercore":  "checkpoints_final/advanced/tiny_imagenet/hypercore_base/hypercore_tiny_imagenet.pt",
        "hexformer":  "checkpoints_final/advanced/tiny_imagenet/hexformer_base/hexformer_tiny_imagenet.pt",
    },
}

HIER_JSON = {
    "cifar100":      REPO_ROOT / "experiments/hyperbolic/hierarchy_cfiar100.json",
    "tiny_imagenet": REPO_ROOT / "experiments/hyperbolic/hierarchy_tiny_imagenet.json",
}

NUM_CLASSES = {"cifar100": 100, "tiny_imagenet": 200}


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical evaluation (Kosmopoulos et al. 2015)")
    parser.add_argument("--dataset", required=True, choices=["cifar100", "tiny_imagenet"])
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models", nargs="+",
                        default=["euclidean", "hyperbolic", "hypercore", "hexformer"])
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}  |  Dataset: {args.dataset}")

    # ── build hierarchy ───────────────────────────────────────────────────────
    print("\nBuilding hierarchy...")
    num_classes = NUM_CLASSES[args.dataset]
    tree, class_nodes, root = load_tree(HIER_JSON[args.dataset], num_classes)
    print(f"  Nodes: {tree.number_of_nodes()}, Edges: {tree.number_of_edges()}")

    print("  Precomputing pairwise TIE / LCA height / ancestor sets...")
    tie_mat, lca_h_mat, anc_sets, tree_height = precompute_pairwise(
        tree, root, class_nodes)
    print(f"  Tree height: {tree_height}  Done.")

    # ── load data ─────────────────────────────────────────────────────────────
    dataset, _ = get_dataset(
        args.dataset, data_dir=str(REPO_ROOT / args.data_dir),
        validation_size=5000, augmentation="simple"
    )
    loaders = get_dataloader(dataset, args.batch_size, shuffle=False, augmentation="simple")
    loader = loaders[args.split]
    print(f"\nEvaluating on {args.split} split: {len(loader.dataset)} samples")

    # ── evaluate each model ───────────────────────────────────────────────────
    results = {}
    for model_name in args.models:
        ckpt_path = REPO_ROOT / CHECKPOINTS[args.dataset][model_name]
        if not ckpt_path.exists():
            print(f"\n[SKIP] {model_name}: checkpoint not found")
            continue

        print(f"\n── {model_name} ──")
        cfg = cfg_from_checkpoint(str(ckpt_path))
        model = build_model(model_name, cfg).to(device)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        print(f"  Loaded (epoch={state.get('epoch','?')}, "
              f"saved_test_acc={state.get('test_acc', state.get('val_acc','?')):.4f})")

        preds, labels = get_predictions(model, loader, device)
        m = compute_metrics(preds, labels, tie_mat, lca_h_mat, anc_sets)
        results[model_name] = m
        print(f"  top1={m['top1_acc']:.4f}  TIE={m['TIE']:.3f}  LCA={m['LCA']:.3f}"
              f"  J={m['J']:.3f}  P_H={m['P_H']:.3f}  R_H={m['R_H']:.3f}")
        del model

    # ── summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"Hierarchical Evaluation  |  {args.dataset}  {args.split}  "
          f"tree_height={tree_height}")
    print(f"Metrics: Kosmopoulos et al. (2015); TIE: Dekel et al. (2004)")
    print(f"Hierarchy: WordNet (Miller 1994) / CIFAR-100 superclasses")
    print("=" * 80)
    print(f"{'Model':<14}  {'Top-1':>6}  {'TIE ↓':>7}  {'LCA ↓':>7}  "
          f"{'J ↑':>7}  {'P_H ↑':>7}  {'R_H ↑':>7}")
    print("-" * 80)
    for name, m in results.items():
        print(f"{name:<14}  {m['top1_acc']:>6.4f}  {m['TIE']:>7.3f}  {m['LCA']:>7.3f}"
              f"  {m['J']:>7.3f}  {m['P_H']:>7.3f}  {m['R_H']:>7.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
