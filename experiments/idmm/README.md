# IDMM Experiments

Implementation of **Instance Discrimination with Multi-crop and CutMix (IDMM)** from:
> *Training Vision Transformers with Only 2040 Images* — applied to hyperbolic and Euclidean ViT.

The method trains on very small datasets (hundreds to low thousands of images) using a two-stage pipeline:
1. **Pretrain** — self-supervised instance discrimination: each training image is its own class. Multiple crops of the same image share the same instance label, giving more gradient signal per image.
2. **Finetune** — standard supervised cross-entropy, initialised from the pretrained encoder.

---

## Supported datasets

All download automatically on first run via torchvision.

| Dataset | Classes | Train size | Test size |
|---|---|---|---|
| `flowers102` | 102 | 1,020 | 6,149 |
| `pets` | 37 | 3,680 | 3,669 |
| `dtd` | 47 | 1,880 | 1,880 |
| `aircrafts` | 100 | 3,334 | 3,333 |

---

## Method details

### Multi-crop
Each training image produces **2 global crops** (scale 0.4–1.0) and **4 local crops** (scale 0.05–0.4), all resized to 224×224. The model sees 6 views per image per step, with gradients accumulated across all crops before each optimizer update.

### Instance discrimination
Each training image is assigned a unique class index. The encoder + a linear head (size = number of training images) are trained with cross-entropy to identify the source image from any of its crops.

### CutMix
Applied during pretraining. Two crops from different instances are mixed spatially; the loss is a weighted combination of the two instance targets proportional to the crop area.

### Two-stage
The pretrained encoder (without the instance-discrimination head) is loaded for fine-tuning. Only the classification head is re-initialised from random weights.

---

## Running on Snellius

Edit the variables at the top of `idmm.job`:

```bash
DATASET=flowers102     # flowers102 | pets | dtd | aircrafts
MODEL=hyperbolic       # euclidean | hyperbolic
```

Then submit from the **repo root**:

```bash
sbatch experiments/idmm/idmm.job
```

Both stages run in the same allocation. Logs go to `out/idmm/slurm_<jobid>.out`. Checkpoints are saved to `checkpoints/idmm/<dataset>/<model>/`.

### Default hyperparameters

| | Pretrain | Finetune |
|---|---|---|
| Epochs | 800 | 50 |
| Batch size | 128 | 64 |
| LR | 5e-4 | 1e-3 |
| Scheduler | warmup cosine | warmup cosine |
| Global crops | 2 | — |
| Local crops | 4 | — |
| CutMix | yes | no |

Architecture: ViT-Tiny/16 (12 layers, hidden size 192, 3 heads) — same as used in the IDMM paper.

---

## File overview

```
experiments/idmm/
├── train_idmm.py   # training script (both stages)
├── data_utils.py   # dataset loading via torchvision
├── idmm.job        # SLURM job script
└── README.md
```

`train_idmm.py` imports `utils.py` and model configs from `experiments/main/` via `PYTHONPATH`.
