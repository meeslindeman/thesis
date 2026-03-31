import wandb
api = wandb.Api()

runs = api.runs("hyperbolic-vit-team/cifar10")

for r in runs:
    r.config["variant"] = r.name
    r.update()