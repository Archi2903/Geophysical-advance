import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Генерация искусственных данных (замените вашими реальными данными)
# t - временные отсчеты, signal - смоделированный сигнал, reference - эталонный
t = np.linspace(0.001, 0.1, 100).reshape(-1, 1)
simulated = 100 * np.exp(-5*t) + 0.5*np.random.normal(size=t.shape)  # Смоделированные данные с шумом
reference = 100 * np.exp(-5*t)  # Идеальный экспоненциальный спад (эталон)

# Преобразование временных признаков в полиномиальные (2 степень)
poly = PolynomialFeatures(degree=2, include_bias=True)
t_poly = poly.fit_transform(t)

# Обучение линейной регрессии
model = LinearRegression()
model.fit(t_poly, simulated)

# Предсказание модели
t_fit = np.linspace(0.001, 0.1, 200).reshape(-1, 1)
t_fit_poly = poly.transform(t_fit)
predicted = model.predict(t_fit_poly)

# Оценка качества модели
mse = mean_squared_error(reference, model.predict(t_poly))
print(f"Среднеквадратичная ошибка: {mse:.4f}")

# Визуализация
plt.figure(figsize=(10, 6))
plt.scatter(t, simulated, c='blue', s=10, marker='o', label='Смоделированные данные')
plt.plot(t, reference, 'g--', lw=2, label='Эталонный сигнал')
plt.plot(t_fit, predicted, 'r-', lw=2, label='Предсказание регрессии')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда поля (нТл)')
plt.title('Сравнение смоделированного и эталонного TEM-поля')
plt.legend()
plt.grid(True)
plt.show()

# Вывод коэффициентов модели
print(f"\nКоэффициенты модели:")
print(f"Intercept: {model.intercept_[0]:.4f}")
for i, coef in enumerate(model.coef_[0]):
    print(f"t^{i}: {coef:.4f}")