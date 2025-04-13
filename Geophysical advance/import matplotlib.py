import matplotlib.pyplot as plt
import numpy as np

# Настройки времени
t = np.linspace(0, 5, 500)

# RL-цепь в DC: ток нарастает
R = 1  # Ом
L = 1  # Гн
U = 5  # Вольт
I_RL = (U / R) * (1 - np.exp(-R * t / L))

# RC-цепь в DC: зарядка и разрядка конденсатора
RC = 1  # RC = R*C
V_RC_charge = U * (1 - np.exp(-t / RC))
V_RC_discharge = U * np.exp(-t / RC)

# RL-цепь в AC: синусоидальный ток
omega = 2 * np.pi * 1  # 1 Гц
I_AC = np.sin(omega * t - np.pi / 4)  # сдвиг по фазе

# Построение графиков
fig, axs = plt.subplots(3, 1, figsize=(8, 10))
fig.suptitle('Графики переходных процессов в RL и RC цепях', fontsize=14)

# 1. RL цепь (DC)
axs[0].plot(t, I_RL, label="I(t) = (U/R)(1 - e^(-Rt/L))", color="blue")
axs[0].set_title("RL-цепь (DC): нарастание тока")
axs[0].set_xlabel("Время (с)")
axs[0].set_ylabel("Ток (A)")
axs[0].legend()
axs[0].grid(True)

# 2. RC цепь (DC)
axs[1].plot(t, V_RC_charge, label="Зарядка: V(t) = U(1 - e^(-t/RC))", color="green")
axs[1].plot(t, V_RC_discharge, label="Разрядка: V(t) = U * e^(-t/RC)", color="red")
axs[1].set_title("RC-цепь (DC): зарядка и разрядка")
axs[1].set_xlabel("Время (с)")
axs[1].set_ylabel("Напряжение (В)")
axs[1].legend()
axs[1].grid(True)

# 3. RL цепь (AC)
axs[2].plot(t, I_AC, label="I(t) = I₀ * sin(ωt + φ)", color="purple")
axs[2].set_title("RL-цепь (AC): синусоидальный ток")
axs[2].set_xlabel("Время (с)")
axs[2].set_ylabel("Ток (A)")
axs[2].legend()
axs[2].grid(True)

# Сохранение графика на рабочий стол
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("C:/Users/artur/Desktop/Graphs_RL_RC_Circuits.png")
plt.show()
