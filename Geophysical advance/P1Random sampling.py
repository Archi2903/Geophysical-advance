import numpy as np
import matplotlib.pyplot as plt

# Define μ,σ,N
u = 175      # Expected value E[X] - "μ" 
sigma = 3   # Expected variance Var - "σ"
N = 10000  # Sample

# Generate random samples  Z ~ N(0,1) 
Z = np.random.randn(N)

# Transform the samples X ~ N(μ, σ^2) using X = μ + σ * Z for given values of μ and σ
X = u + sigma * Z

# Vizualization Plot a histogram
plt.figure(figsize=(8, 5))
plt.hist(X, bins=50, density=True, alpha=0.6, color='gray', edgecolor='black')
plt.axvline(u, color='green', linestyle='solid', linewidth=2, label=f'Theoretical values μ: {u:.2f}')
plt.axvline(np.mean(X), color='red', linestyle='dashed', linewidth=2, label=f'Sample values μ: {np.mean(X):.2f}')
plt.axvline(u - sigma**2, color='blue', linestyle='solid', linewidth=2, label=f'Theoretical σ^2: {(sigma**2):.2f}')
plt.axvline(np.mean(X) - sigma**2, color='red', linestyle='dashed', linewidth=2, label=f'Sample σ^2: {(np.var(X)):.2f}')
plt.axvline(u + sigma**2, color='blue', linestyle='solid', linewidth=2)
plt.axvline(np.mean(X) + sigma**2, color='red', linestyle='dashed', linewidth=2)
plt.xlabel("X")
plt.ylabel("Density of Normal Distribution")
plt.title("Histogram of Transformed Normal Distribution  X = μ + σ * Z")
plt.legend()
plt.show()

# Calculate the sample mean and variance X∼N(μ,σ^2)
"1. Sample mean E[X]=E[μ+σZ]=μ+σE[Z]=μ -> E[X] = μ"
EX = np.mean(X)
"2. Sample variance Var[X]=Var[μ+σZ]=σ^2Var[Z]=σ^2 -> Var[X] = σ^2"
Var = np.var(X)
print(f"Theoretical values μ: {u}, Sample values μ: {EX:.2f}")
print(f"Theoretical σ: {sigma**2}, Sample σ: {Var:.2f}")