import numpy as np
from sklearn.linear_model import LinearRegression

# Data
t = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([109.4, 187.5, 267.5, 331.9, 386.1, 428.4, 452.2, 498.1, 512.3, 513.0])

# Forward model
# m1 + m2*t - 0.5*m3*t^2
G = np.column_stack([
    np.ones_like(t),   # m1
    t,                 # m2
    -0.50 * t**2       # m3
])

# Model 
model = LinearRegression(fit_intercept=False)
model.fit(G, y)

# Оценка коэффициентов
m_L2 = model.coef_
print("L2 model coefficients:", m_L2)

# Regression L1

max_iter= 50
tol= 0.0003
eps= 1.e-32
print(f''iteration{0}[m_1, m_2, m_3]:", m_iter)


#stopping criterian
delta_m = m_iter - m_prev
tau = np.linalg.norm(delta_m) / (1 + np.linalg.norm(m_iter))
print(f"tau: {tau:.4f}")

for i in range(max_iter)