import numpy as np
import matplotlib.pyplot as plt

# Создание массива значений для оси X (от 0.1 до 10)
x = np.linspace(0.1, 10, 400)

# Вычисление логарифма
y = np.log(x)

# Построение графика
plt.plot(x, y, label="log(x)")

# Настройка осей и заголовка
plt.xlabel("x")
plt.ylabel("log(x)")
plt.title("График логарифма")

# Добавление сетки и легенды
plt.grid(True)
plt.legend()

# Отображение графика
plt.show()
