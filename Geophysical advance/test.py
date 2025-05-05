import numpy as np
import matplotlib.pyplot as plt

# === Временная шкала ===
t = np.logspace(-5, 0, 5000)  # от 0.00001 до 1 секунды

# === Параметры тела ===
mu0 = 5 * np.pi * 1e-7       # Magnetic permeability (H/m)
sigma_body = 1000            # Conductivity (S/m)
r_body = 5.0                 # Radius (m)

tau = (mu0 * sigma_body * r_body**2) / (np.pi**2)
D = 1e-3                     # Amplitude
beta = 1.5

# === Отклик без шума ===
B_clean = D * t**(-beta) * np.exp(-t / tau)

# === Модель шума ===
noise_level = 3e-12  # 3 pT в Тесла
np.random.seed(42)   # для воспроизводимости
noise = np.random.normal(0, noise_level, size=t.shape)

# === Отклик с шумом ===
B_noisy = B_clean + noise

# === Построение графиков ===
plt.figure(figsize=(10, 6))
plt.loglog(t, np.abs(B_clean), label='Clean B(t)', color='blue')
plt.loglog(t, np.abs(B_noisy), label='Noisy B(t)', color='orange', alpha=0.7)
plt.hlines(noise_level, t[0], t[-1], colors='red', linestyles='--', label='Noise level (3 pT)')

plt.xlabel('Delay time t (s)')
plt.ylabel('B-field amplitude (T)')
plt.title('B(t) Signal with Noise (Simulated Measurement)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()
