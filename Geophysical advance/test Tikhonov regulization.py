import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import minimize, least_squares
from numpy.linalg import svd  # [стр. 55–57] SVD: глава 3.1-3.2

# 1. Layer visualization (визуализация геоэлектрических слоёв)
layer_names = ["Sand", "Sandstone", "Granite"]
layer_colors = ["#f4e2b4", "#d4b483", "#c0c0c0"]
layer_thicknesses = [20, 30, 50]

fig, ax = plt.subplots(figsize=(6, 6))
depth = 0
for name, color, thickness in zip(layer_names, layer_colors, layer_thicknesses):
    ax.add_patch(patches.Rectangle((0, depth), 10, thickness, color=color, edgecolor='black'))
    ax.text(5, depth + thickness/2, name, ha='center', va='center')
    depth += thickness
ax.set_xlim(0, 10); ax.set_ylim(depth, 0)
ax.set_xlabel("Horizontal position"); ax.set_ylabel("Depth (m)")
ax.set_title("3-Layer Geoelectric Model")
ax.grid(True, linestyle='--')
plt.tight_layout()
plt.show()

# 2. Synthetic dB/dt (экспоненциальная модель затухания)
time = np.logspace(-6, -2, 100)  # [стр. 26–28] генерация синтетических данных
layers = [
    {"thickness": 20, "conductivity": 1e-3},
    {"thickness": 30, "conductivity": 1e-2},
    {"thickness": np.inf, "conductivity": 1e-5}
]

def synthetic_dBdt(time, layers):
    signal = np.zeros_like(time)
    for i, lyr in enumerate(layers):
        sigma = lyr["conductivity"]
        w = 1.0 if np.isinf(lyr["thickness"]) else np.exp(-time*1e3/(i+1))
        signal += sigma * w / np.sqrt(time)
    return signal * 1e-9

dBdt = synthetic_dBdt(time, layers)

plt.figure(figsize=(8,5))
plt.loglog(time, dBdt, label="Original dB/dt", color='black')
plt.xlabel("Time (s)"); plt.ylabel("dB/dt (T/s)")
plt.title("Synthetic TEM Response")
plt.grid(True, which='both', linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()

# 3. Least Squares and Tikhonov Inversion
true_thicknesses = [20, 30, np.inf]
initial_guess = np.array([5e-3, 5e-3, 5e-3])
bounds = [(1e-6,1)]*2 + [(1e-7,1e-1)]

def misfit_ls(sigmas):  # [стр. 221, eq. 9.16] least squares criterion
    model = synthetic_dBdt(time, [{"thickness": t, "conductivity": s} for s, t in zip(sigmas, true_thicknesses)])
    return np.sum((model - dBdt)**2)

def misfit_tikh(sigmas, alpha=1e-2):  # [стр. 96, eq. 4.7] Tikhonov regularization
    return misfit_ls(sigmas) + alpha*np.sum(sigmas**2)

res_ls = minimize(misfit_ls, initial_guess, bounds=bounds)
res_tikh = minimize(lambda s: misfit_tikh(s, alpha=1e-2), initial_guess, bounds=bounds)

sol_ls = res_ls.x
sol_tikh = res_tikh.x

dB_ls = synthetic_dBdt(time, [{"thickness": t, "conductivity": s} for s, t in zip(sol_ls, true_thicknesses)])
dB_tikh = synthetic_dBdt(time, [{"thickness": t, "conductivity": s} for s, t in zip(sol_tikh, true_thicknesses)])

plt.figure(figsize=(8,5))
plt.loglog(time, dBdt, label="Original", color='black')
plt.loglog(time, dB_ls, '--', label="Least Squares", color='red')
plt.loglog(time, dB_tikh, '--', label="Tikhonov", color='green')
plt.xlabel("Time (s)"); plt.ylabel("dB/dt")
plt.title("Inversion: LS vs Tikhonov")
plt.grid(True, which='both', linestyle='--')
errors_ls = np.abs(sol_ls - np.array([1e-3,1e-2,1e-5]))
errors_tikh = np.abs(sol_tikh - np.array([1e-3,1e-2,1e-5]))
txt = ("LS σ: " + ", ".join(f"{val:.1e}" for val in sol_ls) + 
       "\nErr: " + ", ".join(f"{err:.1e}" for err in errors_ls) +
       "\nTikh σ: " + ", ".join(f"{val:.1e}" for val in sol_tikh) +
       "\nErr: " + ", ".join(f"{err:.1e}" for err in errors_tikh))
plt.text(1e-5, dBdt.max()*0.5, txt, bbox=dict(facecolor='white', alpha=0.7))
plt.legend(); plt.tight_layout(); plt.show()

# 4. SVD-based Tikhonov (TGSVD)
def jacobian(sigmas, eps=1e-6):  # [стр. 221, eq. 9.20] Якобиан
    J = np.zeros((len(time), len(sigmas)))
    base = synthetic_dBdt(time, [{"thickness": t, "conductivity": s} for s,t in zip(sigmas,true_thicknesses)])
    for i in range(len(sigmas)):
        p = sigmas.copy(); p[i]+=eps
        J[:,i] = (synthetic_dBdt(time, [{"thickness": t, "conductivity": p_i} for p_i,t in zip(p,true_thicknesses)]) - base)/eps
    return J

alphas = np.logspace(-5,-1,20)
best_sol, best_mis = None, np.inf
J0 = jacobian(initial_guess)
U,s_vals,Vt = svd(J0, full_matrices=False)  # [стр. 95–97] SVD в Тихонове
for a in alphas:
    filt = s_vals/(s_vals**2 + a**2)  # [стр. 97, eq. 4.17] фильтр Тихонова
    inv = Vt.T @ np.diag(filt) @ U.T
    dm = inv @ (dBdt - synthetic_dBdt(time, [{"thickness": t, "conductivity": ig} for ig,t in zip(initial_guess, true_thicknesses)]))
    sol = initial_guess + dm
    m = misfit_ls(sol)
    if m<best_mis: best_mis,m, best_sol = m,m, sol

sol_tgsvd = best_sol
dB_tgsvd = synthetic_dBdt(time, [{"thickness": t, "conductivity": s} for s, t in zip(sol_tgsvd, true_thicknesses)])

plt.figure(figsize=(8,5))
plt.loglog(time, dBdt, label="Original", color='black')
plt.loglog(time, dB_tgsvd, '--', label="TGSVD", color='purple')
plt.xlabel("Time (s)"); plt.ylabel("dB/dt")
plt.title("TGSVD Inversion")
plt.grid(True, which='both', linestyle='--')
err_tg = np.abs(sol_tgsvd - np.array([1e-3,1e-2,1e-5]))
txt2 = "TGSVD σ: " + ", ".join(f"{v:.1e}" for v in sol_tgsvd) + \
       "\nErr: " + ", ".join(f"{e:.1e}" for e in err_tg)
plt.text(1e-5, dBdt.max()*0.5, txt2, bbox=dict(facecolor='white', alpha=0.7))
plt.legend(); plt.tight_layout(); plt.show()

# 5. Levenberg-Marquardt [стр. 223, eq. 9.30] метод

def resid(sigmas):
    return synthetic_dBdt(time, [{"thickness": t, "conductivity": s} for s,t in zip(sigmas,true_thicknesses)]) - dBdt

res_lm = least_squares(resid, sol_tgsvd, method='lm')
sol_lm = res_lm.x
dB_lm = synthetic_dBdt(time, [{"thickness": t, "conductivity": s} for s, t in zip(sol_lm, true_thicknesses)])

plt.figure(figsize=(8,5))
plt.loglog(time, dBdt, label="Original", color='black')
plt.loglog(time, dB_lm, '--', label="LM", color='orange')
plt.xlabel("Time (s)"); plt.ylabel("dB/dt")
plt.title("Levenberg-Marquardt Inversion")
plt.grid(True, which='both', linestyle='--')
err_lm = np.abs(sol_lm - np.array([1e-3,1e-2,1e-5]))
txt3 = "LM σ: " + ", ".join(f"{v:.1e}" for v in sol_lm) + \
       "\nErr: " + ", ".join(f"{e:.1e}" for e in err_lm)
plt.text(1e-5, dBdt.max()*0.5, txt3, bbox=dict(facecolor='white', alpha=0.7))
plt.legend(); plt.tight_layout(); plt.show()
