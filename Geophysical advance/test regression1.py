import numpy as np
import matplotlib.pyplot as plt

# 1. Создание матрицы чувствительности G
def build_G(n_data=30, n_model=20):
    depth = np.linspace(1, 100, n_model)
    time = np.linspace(1, 100, n_data)
    G = np.exp(-np.outer(time, 1 / depth))  # экспоненциальная чувствительность
    return G, time, depth

# 2. Истинная модель
def true_model(depth):
    return np.where(depth < 40, 1.0, 3.0)

# 3. Добавление шума к данным
def add_noise(d, noise_level=0.01):
    np.random.seed(0)
    return d + np.random.normal(0, noise_level * np.max(d), size=d.shape)

# 4. Обратное решение через псевдообратную матрицу
def invert_lstsq(G, d):
    return np.linalg.lstsq(G, d, rcond=None)[0]

# 5. Обратное решение через QR-разложение
def invert_qr(G, d):
    Q, R = np.linalg.qr(G)
    return np.linalg.solve(R, Q.T @ d)

# 6. Число обусловленности
def condition_number(G):
    return np.linalg.cond(G)

# --- Основной блок ---
G, t, z = build_G()
m_true = true_model(z)
d = G @ m_true
d_noisy = add_noise(d)
m_lstsq = invert_lstsq(G, d_noisy)
m_qr = invert_qr(G, d_noisy)
cond_G = condition_number(G)

# Визуализация чувствительности G
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(G, aspect='auto', extent=[z[0], z[-1], t[-1], t[0]])
plt.colorbar(label='Sensitivity')
plt.xlabel('Depth')
plt.ylabel('Time')
plt.title('Матрица чувствительности G (Time x Depth)')

# Визуализация восстановления модели
plt.subplot(1, 2, 2)
plt.plot(z, m_true, label='Истинная модель', linewidth=3)
plt.plot(z, m_lstsq, '--', label='Восстановление (Least Squares)', linewidth=2)
plt.plot(z, m_qr, ':', label='Восстановление (QR)', linewidth=2)
plt.xlabel('Глубина')
plt.ylabel('Электропроводность')
plt.title(f'Сравнение моделей | Cond(G) = {cond_G:.1e}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
