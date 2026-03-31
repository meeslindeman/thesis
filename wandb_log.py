import ast
import re
from pathlib import Path

import wandb


ENTITY = "hyperbolic-vit-team"
PROJECT = "base"
GROUP = "base"
RUN_NAME = "extrinsic"

# Choose which config fields should become actual table columns.
# Add/remove as needed.
CONFIG_KEYS_TO_LOG = [
    "lr",
    "epochs",
    "heads",
    "mlp_dim",
    "dropout",
    "skip_connection",
    "use_layernorm",
    "normalization"
]


def parse_config(config_str: str) -> dict:
    if not config_str.startswith("Config(") or not config_str.endswith(")"):
        raise ValueError("Config string does not have expected format.")

    inner = config_str[len("Config("):-1]

    parts = []
    depth = 0
    current = []
    for ch in inner:
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        current.append(ch)
    if current:
        parts.append("".join(current).strip())

    cfg = {}
    for part in parts:
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            cfg[key] = ast.literal_eval(value)
        except Exception:
            cfg[key] = value
    return cfg


def parse_log_file(text: str):
    lines = text.splitlines()

    config = None
    init_curvature = None
    epoch_metric_dicts = []
    epoch_summaries = []
    total_training_time_seconds = None
    total_training_time_minutes = None

    config_re = re.compile(r"^Config\(.*\)$")
    curvature_re = re.compile(r"Initialized LorentzViT with curvature k=([0-9eE+\-.]+)")
    epoch_metrics_re = re.compile(r"^epoch metrics:\s*(\{.*\})\s*$")
    epoch_summary_re = re.compile(
        r"^Epoch:\s*(\d+),\s*Train loss:\s*([0-9eE+\-.]+),\s*Test loss:\s*([0-9eE+\-.]+),\s*Accuracy:\s*([0-9eE+\-.]+)\s*$"
    )
    total_time_re = re.compile(
        r"^Total training time:\s*([0-9eE+\-.]+)\s*seconds\s*\(([0-9eE+\-.]+)\s*minutes\)\s*$"
    )

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if config is None and config_re.match(line):
            config = parse_config(line)
            continue

        m = curvature_re.match(line)
        if m:
            init_curvature = float(m.group(1))
            continue

        m = epoch_metrics_re.match(line)
        if m:
            metrics = ast.literal_eval(m.group(1))
            epoch_metric_dicts.append(metrics)
            continue

        m = epoch_summary_re.match(line)
        if m:
            epoch_summaries.append(
                {
                    "epoch": int(m.group(1)),
                    "train/loss": float(m.group(2)),
                    "test/loss": float(m.group(3)),
                    "test/accuracy": float(m.group(4)),
                }
            )
            continue

        m = total_time_re.match(line)
        if m:
            total_training_time_seconds = float(m.group(1))
            total_training_time_minutes = float(m.group(2))
            continue

    if config is None:
        raise ValueError("Could not find Config(...) line in the log file.")

    if len(epoch_metric_dicts) != len(epoch_summaries):
        raise ValueError(
            f"Mismatched epoch entries: found {len(epoch_metric_dicts)} metric dicts "
            f"but {len(epoch_summaries)} epoch summaries."
        )

    return {
        "config": config,
        "init_curvature": init_curvature,
        "epoch_metric_dicts": epoch_metric_dicts,
        "epoch_summaries": epoch_summaries,
        "total_training_time_seconds": total_training_time_seconds,
        "total_training_time_minutes": total_training_time_minutes,
    }


def extract_block_rows(metrics: dict, epoch: int, run_name: str, group_name: str, config: dict):
    """
    Convert metrics like:
      block3/res_scale_mlp -> row with block_idx=3, metric=res_scale_mlp
    into long-format rows.
    """
    rows = []
    block_re = re.compile(r"^block(\d+)/(.*)$")

    config_values = [config.get(k) for k in CONFIG_KEYS_TO_LOG]

    for key, value in metrics.items():
        m = block_re.match(key)
        if not m:
            continue

        block_idx = int(m.group(1))
        metric_name = m.group(2)
        block_name = f"block{block_idx}"

        rows.append(
            [epoch, block_idx, block_name, metric_name, float(value), run_name, group_name, *config_values]
        )

    return rows


def main(log_path: str):
    text = Path(log_path).read_text(encoding="utf-8")
    parsed = parse_log_file(text)

    config = dict(parsed["config"])
    if parsed["init_curvature"] is not None and "curvature_initialized" not in config:
        config["curvature_initialized"] = parsed["init_curvature"]

    run_name = RUN_NAME if RUN_NAME is not None else config.get("name", "recovered-run")

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        name=run_name,
        config=config,
        job_type="retroactive-log",
    )

    table_columns = [
        "epoch",
        "block_idx",
        "block",
        "metric",
        "value",
        "run_name",
        "group",
        *CONFIG_KEYS_TO_LOG,
    ]
    block_table = wandb.Table(columns=table_columns)

    final_epoch = max(s["epoch"] for s in parsed["epoch_summaries"])

    for metrics, summary in zip(parsed["epoch_metric_dicts"], parsed["epoch_summaries"]):
        epoch = summary["epoch"]

        rows = extract_block_rows(
            metrics=metrics,
            epoch=epoch,
            run_name=run_name,
            group_name=GROUP,
            config=config,
        )
        for row in rows:
            block_table.add_data(*row)

        # Log normal scalar summaries
        wandb.log(summary, step=epoch)

        # Optional: also log block metrics as ordinary scalars for quick per-run plots
        # Example key: blocks/res_scale_mlp/block3
        scalar_log = {}
        for k, v in metrics.items():
            m = re.match(r"^block(\d+)/(.*)$", k)
            if m:
                block_idx = int(m.group(1))
                metric_name = m.group(2)
                scalar_log[f"blocks/{metric_name}/block{block_idx}"] = float(v)
        wandb.log(scalar_log, step=epoch)

        # Save final-epoch block metrics into run summary for easy comparison panels
        if epoch == final_epoch:
            for k, v in metrics.items():
                m = re.match(r"^block(\d+)/(.*)$", k)
                if m:
                    block_idx = int(m.group(1))
                    metric_name = m.group(2)
                    run.summary[f"final/{metric_name}/block{block_idx}"] = float(v)

    wandb.log({"block_metrics_long": block_table})

    if parsed["total_training_time_seconds"] is not None:
        run.summary["total_training_time_seconds"] = parsed["total_training_time_seconds"]
    if parsed["total_training_time_minutes"] is not None:
        run.summary["total_training_time_minutes"] = parsed["total_training_time_minutes"]

    run.finish()


if __name__ == "__main__":
    main("logs.txt")