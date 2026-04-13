import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

torch.manual_seed(42)
layer1 = nn.Linear(2, 3, bias=False)

x = torch.tensor([1., 2.])
z = layer1(x)
print("output z:", z)

W = layer1.weight  # shape (3, 2)

# --- verify: signed distance from x to each hyperplane w^T u = 0
# signed distance = w^T x / ||w||
# but layer output = w^T x  (unscaled, i.e. scaled by ||w||)
print("\nVerification:")
for i in range(3):
    w = W[i]
    dot = w @ x                          # = z[i], what the layer computes
    dist = (w @ x) / w.norm()            # true signed distance
    print(f"  H{i+1}: layer output = {dot:.4f}  |  ||w{i+1}|| = {w.norm():.4f}"
          f"  |  signed dist = {dist:.4f}  |  dist * ||w|| = {dist * w.norm():.4f}")

# --- visualise
fig, ax = plt.subplots(figsize=(6, 8))
ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.axhline(0, color='lightgray', lw=0.8)
ax.axvline(0, color='lightgray', lw=0.8)
ax.set_xlabel('x₁'); ax.set_ylabel('x₂')

colors = ['#7F77DD', '#D85A30', '#1D9E75']
ts = np.linspace(-3, 3, 300)

for i in range(3):
    w = W[i].detach().numpy()
    col = colors[i]
    wnorm = np.linalg.norm(w)
    
    # hyperplane w^T u = 0: direction perpendicular to w
    # parametric: u = t * [-w[1], w[0]] / wnorm
    direction = np.array([-w[1], w[0]]) / wnorm
    pts = np.outer(ts, direction)
    ax.plot(pts[:, 0], pts[:, 1], color=col, lw=1.5, label=f'H{i+1}')
    
    # normal vector (weight direction)
    ax.annotate('', xy=w/wnorm, xytext=(0,0),
                arrowprops=dict(arrowstyle='->', color=col, lw=1.5))
    ax.text(w[0]/wnorm*1.15, w[1]/wnorm*1.15, f'w{i+1}', color=col, fontsize=10)
    
    # foot of perpendicular from x to hyperplane H_i
    x_np = x.numpy()
    proj_scalar = np.dot(w, x_np) / (wnorm**2)
    foot = x_np - proj_scalar * w          # foot on hyperplane
    
    # dashed line from x to foot
    ax.plot([x_np[0], foot[0]], [x_np[1], foot[1]],
            color=col, lw=1, linestyle='--', alpha=0.7)
    
    # small right-angle marker at foot
    perp = direction * 0.12
    along = w / wnorm * 0.12
    sq = foot + perp + along
    ax.plot([foot[0]+perp[0], sq[0], foot[0]+along[0]],
            [foot[1]+perp[1], sq[1], foot[1]+along[1]],
            color=col, lw=0.8)
    
    # label the distance
    mid = (x_np + foot) / 2 + np.array([0.08, 0.08])
    signed_dist = np.dot(w, x_np) / wnorm
    ax.text(mid[0], mid[1], f'd={signed_dist:.2f}', color=col, fontsize=8.5)

# input point x
ax.scatter(*x.numpy(), color='#378ADD', s=80, zorder=5)
ax.text(x[0]+0.1, x[1]+0.1, f'x = ({x[0]:.0f}, {x[1]:.0f})', fontsize=10, color='#378ADD')

ax.legend(loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig('funny/hyperplanes.png', dpi=150, bbox_inches='tight')
plt.show()