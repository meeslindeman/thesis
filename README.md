# Hyperbolic Vision Transformers

Comparative study of Vision Transformer (ViT) architectures in Euclidean and hyperbolic space, evaluated on CIFAR-10/100.

## Models

| Model | Geometry | Key reference |
|---|---|---|
| `euclidean` | Standard ViT in R^n | Dosovitskiy et al., 2020 |
| `hyperbolic` | Lorentz model, custom manifold | — |
| `hypercore` | Lorentz model via geoopt | HyperCore |

All three share the same training pipeline for a direct, controlled comparison.

## Setup

Clone with submodules:
```bash
git clone --recurse-submodules https://github.com/meeslindeman/hyperbolic-vit.git
```

Create the environment and install the external dependency:
```bash
conda env create -f environment.yml
conda activate hvit
pip install -e external/hyperbolic-fully-connected --no-deps
```

## Training

```bash
python train.py --model [euclidean|hyperbolic|hypercore] [args]
```

Model-specific defaults (lr, optimizer, dropout) are set in each model's `config.py` and can be overridden from the CLI. Run `python train.py --model <name> --help` to see all options for a given model.

**Common arguments**

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `cifar10` | `cifar10` or `cifar100` |
| `--epochs` | `30` | Number of training epochs |
| `--batch_size` | `256` | Batch size |
| `--scheduler` | `none` | `none`, `cosine`, or `warmup_cosine` |
| `--hidden_size` | `48` | Embedding dimension |
| `--num_hidden_layers` | `4` | Transformer depth |
| `--num_attention_heads` | `4` | Number of attention heads |

**ViT size presets** (following Dosovitskiy et al.):

| Size | `hidden_size` | `num_hidden_layers` | `num_attention_heads` | `mlp_dim` |
|---|---|---|---|---|
| Tiny  | 192 | 12 | 3  | 768  |
| Small | 384 | 12 | 6  | 1536 |
| Base  | 768 | 12 | 12 | 3072 |

## SLURM

`train.job` — full training run, configure `MODEL` and `SIZE` at the top.  
`exp.job` — quick experiment, 10-epoch default with model-appropriate LR.

```bash
sbatch exp.job
sbatch train.job
```

## Repository structure

```
train.py              # unified entry point (--model selects the model)
config_base.py        # shared base config (all common hyperparameters)
utils.py              # training loop, evaluation, argument parsing
data_utils.py         # CIFAR-10/100 loading and augmentation
vit/                  # Euclidean ViT
vit_hyperbolic/       # Hyperbolic ViT (Lorentz, custom manifold)
vit_hypercore/        # Hyperbolic ViT (Lorentz, geoopt)
external/             # hyperbolic-fully-connected submodule
```

## Updating the submodule

```bash
git submodule update --remote --merge
git commit -m "Update hyperbolic-fully-connected submodule"
```
