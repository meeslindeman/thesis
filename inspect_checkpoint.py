import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to .pt checkpoint")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = ckpt["model"]

    print(f"Checkpoint epoch: {ckpt.get('epoch')}  acc: {ckpt.get('acc')}\n")

    # Per-block learned scalars: res_scale_att, res_scale_mlp, delta
    blocks = {}
    for k, v in state.items():
        if "blocks." in k and ("res_scale" in k or k.endswith(".delta")):
            parts = k.split(".")
            idx = int(parts[2])
            name = parts[3]
            blocks.setdefault(idx, {})[name] = v.item()

    if blocks:
        print(f"{'Block':>6}  {'sig(res_att)':>12}  {'sig(res_mlp)':>12}  {'delta':>8}")
        print("-" * 46)
        for i in sorted(blocks):
            b = blocks[i]
            s_att = torch.sigmoid(torch.tensor(b.get("res_scale_att", 0.0))).item()
            s_mlp = torch.sigmoid(torch.tensor(b.get("res_scale_mlp", 0.0))).item()
            delta = b.get("delta", float("nan"))
            print(f"{i:>6}  {s_att:>12.4f}  {s_mlp:>12.4f}  {delta:>8.4f}")

    # BatchNorm gamma (scale) and beta_space (bias)
    print("\nBatchNorm gamma / beta_space (abs mean per block):")
    bn = {}
    for k, v in state.items():
        if "batchnorm" not in k:
            continue
        parts = k.split(".")
        idx = int(parts[2])
        name = parts[3] + "." + parts[4]
        bn.setdefault(idx, {})[name] = v.abs().mean().item()
    if bn:
        sample = next(iter(bn.values()))
        headers = sorted(sample)
        print(f"  {'Block':>5}  " + "  ".join(f"{h:>14}" for h in headers))
        for i in sorted(bn):
            vals = "  ".join(f"{bn[i].get(h, float('nan')):>14.4f}" for h in headers)
            print(f"  {i:>5}  {vals}")
    else:
        print("  (none found)")

    # `a` bias parameters per layer type (are they learning away from 0?)
    print("\nLinear `a` bias (abs mean by layer type):")
    for key in ["W_q", "W_k", "W_v", "W_o", "fc1", "fc2", "classifier"]:
        vals = [v.abs().mean().item() for k, v in state.items() if key in k and k.endswith(".a")]
        if vals:
            print(f"  {key:12s}: mean={sum(vals)/len(vals):.4f}  min={min(vals):.4f}  max={max(vals):.4f}")

    # Weight-norm magnitudes per layer type
    print("\nLinear weight-norm magnitudes (abs mean g):")
    for key in ["W_q", "W_k", "W_v", "W_o", "fc1", "fc2", "classifier"]:
        gs = [v for k, v in state.items() if key in k and k.endswith(".g")]
        if gs:
            vals = [v.abs().mean().item() for v in gs]
            print(f"  {key:12s}: mean={sum(vals)/len(vals):.4f}  min={min(vals):.4f}  max={max(vals):.4f}")


if __name__ == "__main__":
    main()
