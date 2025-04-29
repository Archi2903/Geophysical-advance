import numpy as np
import matplotlib.pyplot as plt

# === Параметры модели ===
nx, nz = 100, 100  # размер сетки
x = np.linspace(-50, 50, nx)  # метров
z = np.linspace(0, 100, nz)   # метров
X, Z = np.meshgrid(x, z)

# Слои по глубине
sigma = np.ones_like(X) * 1e-6  # воздух: очень низкая проводимость
sigma[Z > 10] = 1e-2            # слабопроводящий слой: 100 Ом·м
sigma[Z > 50] = 0.1             # нижний слой: 10 Ом·м

# Проводящее тело
body_center_x = 0  # центр тела по X
body_center_z = 30  # центр тела по Z
body_radius = 5     # радиус тела в метрах

# Условие для тела
body = (X - body_center_x)**2 + (Z - body_center_z)**2 <= body_radius**2
sigma[body] = 1000.0  # тело: высокая проводимость (0.1 Ом·м)

# === Визуализация проводимости ===
plt.figure(figsize=(10, 6))
plt.pcolormesh(x, z, np.log10(sigma), shading='auto', cmap='viridis')
plt.colorbar(label='log10(Conductivity) (S/m)')
plt.title('2D-model: conductivity ore')
plt.xlabel('JUST COORDINAT X (м)')
plt.ylabel('DEEP Z (м)')
plt.gca().invert_yaxis()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# === Временная шкала ===
t = np.logspace(-5, 0, 5000)  # от 0.00001 до 1 секунды

# === Параметры для первого тела ===
mu0_1 = 5 * np.pi * 1e-7  # Magnetic permeability (H/m)
sigma_body_1 = 1000       # Conductivity of the body (S/m)
r_body_1 = 5.0            # Radius of the body (m)

tau_1 = (mu0_1 * sigma_body_1 * r_body_1**2) / (np.pi**2)
D_1 = 1e-3  # Decay constant (arbitrary scaling)
beta_1 = 1.5

# Step response for the first body
B_t1 = D_1 * t**(-beta_1) * np.exp(-t / tau_1)

# === Параметры для второго тела (например, другая проводимость и радиус) ===
mu0_2 = 1 * np.pi * 1e-7  # Slightly different magnetic permeability (H/m)
sigma_body_2 = 10        # Lower conductivity (S/m)
r_body_2 = 1.0            # Larger radius (m)

tau_2 = (mu0_2 * sigma_body_2 * r_body_2**2) / (np.pi**2)
D_2 = 3e-4  # Different decay constant
beta_2 = 1  # Different beta for the second case

# Step response for the second body
B_t2 = D_2 * t**(-beta_2) * np.exp(-t / tau_2)

# === Построение графиков ===
plt.figure(figsize=(10, 6))
plt.loglog(t, np.abs(B_t1), label='Conductivity Body : σ=1000 S/m, r=5 m', color='blue')
plt.loglog(t, np.abs(B_t2), label='ROCK: σ=10 S/m, r=10 m', color='green')
plt.xlabel('Delay time t (s)')
plt.ylabel('Amplitude')
plt.title('Decay Curves for Two Bodies')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()
