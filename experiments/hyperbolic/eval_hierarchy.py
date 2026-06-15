"""
Hierarchical evaluation of ViT models using the metrics from:

  Bertinetto et al. (2020) "Making Better Mistakes: Leveraging Class
  Hierarchies with Deep Networks." NeurIPS 2020.

Metrics (Section 4.2 of the paper):
  - top1_acc              : standard top-1 accuracy
  - hier_dist_mistake  ↓  : mean LCA height over WRONG predictions
  - avg_hier_dist_at1  ↓  : mean LCA height over ALL predictions
                             (= (1 - top1_acc) * hier_dist_mistake)

LCA height of a node = length of the longest path from that node downward
to any leaf in its subtree (leaves have height 0, root has the maximum
height equal to the tree depth). A lower LCA height means the two classes
share a more specific common ancestor, i.e. the mistake is less severe.

CIFAR-100  : uses hierarchy_cfiar100.json (node ids 0-99 = class indices).
Tiny ImageNet: uses hierarchy_tiny_imagenet.json (pre-built from WordNet
               via NLTK, primary hypernym path → proper tree).

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
        raise RuntimeError(f"No 'config' key in {path}. Cannot reconstruct model.")
    init_names = {f.name for f in fields(Config) if f.init}
    return Config(**{k: v for k, v in state["config"].items() if k in init_names})


# ── hierarchy loading ─────────────────────────────────────────────────────────
def load_tree(json_path: Path, num_classes: int) -> tuple[nx.DiGraph, list, int]:
    """
    Loads a hierarchy JSON (nodes/links format) and returns
    (tree, class_nodes, root) where class_nodes[i] is the tree node id
    for class index i.
    """
    data = json.loads(json_path.read_text())
    tree = nx.DiGraph()
    for node in data["nodes"]:
        tree.add_node(node["id"], label=node["label"])
    for link in data["links"]:
        tree.add_edge(link["source"], link["target"])
    roots = [n for n in tree.nodes if tree.in_degree(n) == 0]
    assert len(roots) == 1, f"Expected single root, got {roots}"

    if "class_to_node" in data:
        class_to_node = {int(k): v for k, v in data["class_to_node"].items()}
    else:
        # CIFAR-100: node id == class index for the 100 leaf nodes
        class_to_node = {i: i for i in range(num_classes)}

    class_nodes = [class_to_node[i] for i in range(num_classes)]
    return tree, class_nodes, roots[0]


# ── LCA height (Bertinetto et al. Section 4.2) ───────────────────────────────
def compute_node_heights(tree: nx.DiGraph, root) -> dict:
    """
    Returns dict mapping node → height, where height = length of the longest
    path downward to any leaf in the node's subtree.  Leaves have height 0.
    Uses post-order DFS so children are processed before their parent.
    """
    heights = {}
    for node in nx.dfs_postorder_nodes(tree, root):
        children = list(tree.successors(node))
        heights[node] = 0 if not children else 1 + max(heights[c] for c in children)
    return heights


def compute_node_depths(tree: nx.DiGraph, root) -> dict:
    return nx.single_source_shortest_path_length(tree, root)


def get_lca(tree: nx.DiGraph, u, v, depths: dict):
    ancestors_u = nx.ancestors(tree, u) | {u}
    ancestors_v = nx.ancestors(tree, v) | {v}
    return max(ancestors_u & ancestors_v, key=lambda n: depths[n])


def precompute_lca_heights(tree: nx.DiGraph, root, class_nodes: list
                           ) -> tuple[np.ndarray, dict]:
    """
    Returns (lca_height_matrix, heights) where
      lca_height_matrix[i, j] = height of LCA(class_nodes[i], class_nodes[j])
    For i == j (correct prediction), LCA = the leaf itself, height = 0.
    """
    depths = compute_node_depths(tree, root)
    heights = compute_node_heights(tree, root)
    n = len(class_nodes)
    mat = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i + 1, n):
            lca = get_lca(tree, class_nodes[i], class_nodes[j], depths)
            h = heights[lca]
            mat[i, j] = h
            mat[j, i] = h
    # diagonal stays 0: correct prediction → LCA = leaf → height = 0
    return mat, heights


# ── inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def get_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        preds = model(images).argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


# ── metrics (Bertinetto et al. 2020, Section 4.2) ─────────────────────────────
def compute_metrics(preds: np.ndarray, labels: np.ndarray,
                    lca_height_matrix: np.ndarray) -> dict:
    """
    lca_height_matrix[i, j] = LCA height for class pair (i, j).
    Diagonal is 0 (correct prediction: LCA = the leaf itself, height = 0).

    hier_dist_mistake : mean LCA height over wrong predictions   (↓ better)
    avg_hier_dist_at1 : mean LCA height over all predictions     (↓ better)
                        = (1 - top1_acc) * hier_dist_mistake
    """
    top1_acc = float((preds == labels).mean())
    lca_heights = lca_height_matrix[preds, labels]
    wrong = preds != labels
    hier_dist_mistake = float(lca_heights[wrong].mean()) if wrong.any() else 0.0
    avg_hier_dist_at1 = float(lca_heights.mean())
    return {
        "top1_acc": top1_acc,
        "hier_dist_mistake": hier_dist_mistake,
        "avg_hier_dist_at1": avg_hier_dist_at1,
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
        description="Hierarchical evaluation (Bertinetto et al. 2020)")
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

    print("  Precomputing LCA height matrix...")
    lca_height_matrix, heights = precompute_lca_heights(tree, root, class_nodes)
    leaf_heights = np.array([heights[n] for n in class_nodes])
    print(f"  Tree height (root): {heights[root]}  |  "
          f"Leaf heights: min={leaf_heights.min()}, max={leaf_heights.max()}  Done.")

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
        m = compute_metrics(preds, labels, lca_height_matrix)
        results[model_name] = m
        print(f"  top1_acc={m['top1_acc']:.4f}  "
              f"hier_dist_mistake={m['hier_dist_mistake']:.3f}  "
              f"avg_hier_dist@1={m['avg_hier_dist_at1']:.3f}")
        del model

    # ── summary table ─────────────────────────────────────────────────────────
    w = heights[root]
    print("\n" + "=" * 68)
    print(f"Hierarchical Evaluation  |  dataset={args.dataset}  split={args.split}")
    print(f"Metrics: Bertinetto et al. (2020), NeurIPS  |  Tree height: {w}")
    print(f"LCA height: 0 = leaf (best possible), {w} = root (worst possible)")
    print("=" * 68)
    print(f"{'Model':<14}  {'Top-1 Acc':>10}  {'Hier. dist. mistake':>20}  "
          f"{'Avg. hier. dist. @1':>20}")
    print(f"{'':14}  {'':10}  {'(↓ lower = better)':>20}  {'(↓ lower = better)':>20}")
    print("-" * 68)
    for model_name, m in results.items():
        print(f"{model_name:<14}  {m['top1_acc']:>10.4f}  "
              f"{m['hier_dist_mistake']:>20.3f}  "
              f"{m['avg_hier_dist_at1']:>20.3f}")
    print("=" * 68)


if __name__ == "__main__":
    main()
