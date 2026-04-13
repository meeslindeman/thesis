import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from geoopt.manifolds import Lorentz

torch.manual_seed(7)

kappa = 0.5
manifold = Lorentz(k=kappa)

# --- minimal layer setup (Din=1 spatial => ambient R^2, Dout=3)
# Din=2 in the layer's terms means 2 total coords (1 time + 1 space)
# so the input lives on L^1_kappa: t^2 - x^2 = 1/kappa

in_features  = 2   # total Lorentz dim (time + 1 spatial)
out_features = 2   # same (time + 1 spatial), but we produce 3 hyperplanes

# manually set up 3 hyperplane normals v^(i) via (u, b) parametrisation
# u in R^1 (one spatial direction), b in R
params = [
    (torch.tensor([1.0]),  torch.tensor( 0.6)),   # H1
    (torch.tensor([1.0]),  torch.tensor(-0.8)),   # H2
    (torch.tensor([-1.0]), torch.tensor( 0.3)),   # H3
]

def make_v(u, b, kappa):
    """Build space-like normal vector v from direction u and bias b."""
    u_norm = u.norm()
    arg = (kappa**0.5) * b * u_norm
    v_time  =  u_norm * torch.sinh(arg)
    v_space = torch.cosh(arg) * u
    return torch.cat([v_time.unsqueeze(0), v_space])

vs = [make_v(u, b, kappa) for u, b in params]

# --- input point on the hyperboloid: x = (cosh(phi), sinh(phi)) / sqrt(kappa)... 
# hyperboloid: -t^2 + x^2 = -1/kappa  (with sign convention t^2 - x^2 = 1/kappa)
# point at rapidity phi: t = cosh(phi)/sqrt(kappa), x = sinh(phi)/sqrt(kappa)
phi = 0.9
x_lorentz = torch.tensor([
    np.cosh(phi) / np.sqrt(kappa),
    np.sinh(phi) / np.sqrt(kappa)
])
print("x on hyperboloid:", x_lorentz)
print("check <x,x>_L =", -x_lorentz[0]**2 + x_lorentz[1]**2, " should be", -1/kappa)

# --- compute signed distances
# z_i = x ◦ v^(i) = -x_t * v_t + x_s * v_s
def lorentz_inner(a, b):
    return -a[0]*b[0] + a[1]*b[1]

print("\nVerification:")
for i, (v, (u, b)) in enumerate(zip(vs, params)):
    z = lorentz_inner(x_lorentz, v)
    v_norm_L = torch.sqrt(torch.clamp(-v[0]**2 + v[1]**2, min=1e-8))
    signed_dist = 1/kappa**0.5 * torch.arcsinh(kappa**0.5 * z / v_norm_L)
    
    # verification: recover x ◦ v from the signed distance
    reconstructed = v_norm_L * torch.sinh(kappa**0.5 * signed_dist) / kappa**0.5
    
    print(f"  H{i+1}: x ◦ v = {z:.4f}"
          f"  |  ||v||_L = {v_norm_L:.4f}"
          f"  |  signed dist = {signed_dist:.4f}"
          f"  |  reconstructed x ◦ v = {reconstructed:.4f}")

# --- visualise
fig, ax = plt.subplots(figsize=(6, 8))
ax.set_xlim(-3.5, 3.5); ax.set_ylim(-0.3, 4.5)
ax.set_aspect('equal')
ax.axhline(0, color='lightgray', lw=0.8)
ax.axvline(0, color='lightgray', lw=0.8)
ax.set_xlabel('x (spatial)'); ax.set_ylabel('t (time)')

# hyperboloid curve: t = sqrt(x^2 + 1/kappa)
xs_curve = np.linspace(-3.2, 3.2, 500)
ts_curve = np.sqrt(xs_curve**2 + 1/kappa)
ax.plot(xs_curve, ts_curve, color='#444441', lw=2.5, label='hyperboloid $\\mathcal{L}^1_\\kappa$')

colors = ['#7F77DD', '#1D9E75', '#ff0000']

for i, (v, col) in enumerate(zip(vs, colors)):
    v_np = v.detach().numpy()
    
    # ambient plane through origin: -t*v_t + x*v_x = 0 => t = (v_x/v_t) * x
    # draw as line in (x, t) plane
    if abs(v_np[0]) > 1e-3:
        slope = v_np[1] / v_np[0]   # dt/dx
        xs_line = np.linspace(-3.5, 3.5, 100)
        ts_line = slope * xs_line
        ax.plot(xs_line, ts_line, color=col, lw=1.2, linestyle='--', alpha=0.5,
                label=f'ambient plane {i+1}')
    
    # geodesic = intersection of ambient plane with hyperboloid
    # solve: cosh(phi)/sqrt(k) * (-v_t) + sinh(phi)/sqrt(k) * v_x = 0
    # => -v_t * cosh(phi) + v_x * sinh(phi) = 0
    # => tanh(phi) = v_t / v_x
    if abs(v_np[1]) > 1e-3:
        tanh_val = v_np[0] / v_np[1]
        if abs(tanh_val) < 0.9999:
            phi_intersect = np.arctanh(tanh_val)
            t_int = np.cosh(phi_intersect) / np.sqrt(kappa)
            x_int = np.sinh(phi_intersect) / np.sqrt(kappa)
            ax.scatter(x_int, t_int, color=col, s=70, zorder=5)
            ax.text(x_int + 0.08, t_int + 0.08, f'$H_{i+1}$', color=col, fontsize=11)
            
            # dashed line from x to geodesic intersection
            x_np = x_lorentz.numpy()
            ax.plot([x_np[1], x_int], [x_np[0], t_int],
                    color=col, lw=1, linestyle=':', alpha=0.8)
            
            # label distance
            mid_x = (x_np[1] + x_int) / 2 + 0.08
            mid_t = (x_np[0] + t_int) / 2
            z = lorentz_inner(x_lorentz, v)
            v_norm_L = torch.sqrt(torch.clamp(-v[0]**2 + v[1]**2, min=1e-8))
            dist = (1/kappa**0.5 * torch.arcsinh(kappa**0.5 * z / v_norm_L)).item()
            ax.text(mid_x, mid_t, f'd={dist:.2f}', color=col, fontsize=8.5)

# input point (note: plot as (spatial, time) = (x[1], x[0]))
x_np = x_lorentz.numpy()
ax.scatter(x_np[1], x_np[0], color='#378ADD', s=80, zorder=6)
ax.text(x_np[1]+0.1, x_np[0]+0.1,
        f'x = ({x_np[1]:.2f}, {x_np[0]:.2f})', fontsize=9, color='#378ADD')

# legend
handles = [
    mpatches.Patch(color='#7F77DD', label='H1'),
    mpatches.Patch(color='#1D9E75', label='H2'),
    mpatches.Patch(color='#ff0000', label='H3'),
    mpatches.Patch(color='#378ADD', label='input x'),
]
ax.legend(handles=handles, loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig('funny/lorentz_hyperplanes.png', dpi=150, bbox_inches='tight')
plt.show()