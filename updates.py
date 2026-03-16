import wandb
api = wandb.Api()

runs = api.runs("hyperbolic-vit-team/hyperbolic-vit")

for r in runs:
    if r.group == "tiny":
        r.config["variant"] = r.name   # copy name to config
        r.update()