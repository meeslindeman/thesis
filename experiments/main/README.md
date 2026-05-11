# Main Experiments

Supervised image classification comparing three ViT variants:

| Model | Geometry | Key property |
|---|---|---|
| `euclidean` | Flat R^n | Standard ViT baseline |
| `hyperbolic` | Lorentz manifold | Hyperbolic attention + classification |
| `hypercore` | Lorentz manifold | Alternative hyperbolic ViT architecture |

---

## Datasets

All download automatically on first run.

| Dataset | Classes | Train | Test | Image size |
|---|---|---|---|---|
| `cifar10` | 10 | 50,000 | 10,000 | 32×32 |
| `cifar100` | 100 | 50,000 | 10,000 | 32×32 |
| `tiny_imagenet` | 200 | 100,000 | 10,000 | 64×64 |

---

## Running on Snellius

Edit the variables at the top of `exp.job`:

```bash
MODEL=hyperbolic        # euclidean | hyperbolic | hypercore
SIZE=base               # tiny | small | base
DATASET=cifar100        # cifar10 | cifar100 | tiny_imagenet
EPOCHS=100
BATCH_SIZE=256
SCHEDULER=warmup_cosine # none | cosine | warmup_cosine
CURVATURE=1.0           # hyperbolic / hypercore only
```

Then submit from the **repo root**:

```bash
sbatch experiments/main/exp.job
```

Logs go to `out/<dataset>/slurm_<jobid>.out`. Checkpoints are saved to `checkpoints/<dataset>/<model>_<size>/`.

### Architecture sizes

Based on DeiT / ViT-S/Ti/B (Touvron et al., 2021):

| Size | Hidden | Heads | MLP dim | Patch | CIFAR tokens | TinyIN tokens |
|---|---|---|---|---|---|---|
| `tiny` | 192 | 3 | 768 | 4 | 64 | 256 |
| `small` | 384 | 6 | 1536 | 4 | 64 | 256 |
| `base` | 768 | 12 | 3072 | 4 | 64 | 256 |

Patch size is set automatically: 4 for CIFAR (32×32), 8 for Tiny ImageNet (64×64) to keep the token count consistent.

### Resuming a run

```bash
srun python -u experiments/main/train.py \
  --model hyperbolic --dataset cifar100 \
  --resume checkpoints/cifar100/hyperbolic_base/hyperbolic_cifar100.pt \
  [... same flags as original run ...]
```

---

## Hyperparameter sweep

The sweep job runs a SLURM array where each task reads one line of arguments from a params file.

```
scripts/sweep_script.txt   — path to train.py (one line)
scripts/sweep_params.txt   — one set of CLI args per line
```

Adjust `--array=0-N` in `sweep.job` to match the number of lines in `sweep_params.txt`, then:

```bash
sbatch experiments/main/sweep.job
```

Logs go to `out/sweep/slurm_<jobid>_<taskid>.out`.

---

## Key hyperparameters

| Flag | Default | Notes |
|---|---|---|
| `--lr` | 1e-4 (hyp/hypercore), 1e-3 (euc) | Per-model default in Config |
| `--weight_decay` | 0.1 | |
| `--label_smoothing` | 0.1 | |
| `--warmup_epochs` | 10 | Only used with `warmup_cosine` scheduler |
| `--patience` | 5 | Early stopping epochs without improvement; 0 to disable |
| `--curvature` | 1.0 | Hyperbolic / hypercore only |

For batch sizes other than the default (256), scale the learning rate linearly: `lr_new = lr_base * batch_new / 256`.

---

## File overview

```
experiments/main/
├── train.py        # training script
├── data_utils.py   # CIFAR-10/100 and Tiny ImageNet loading
├── utils.py        # shared utilities (scheduler, checkpointing, eval loop)
├── exp.job         # SLURM job for a single run
├── sweep.job       # SLURM array job for hyperparameter sweeps
└── README.md
```
