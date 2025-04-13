import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

x = np.linspace(0, 30, 500)
for nu in [2, 5, 10, 20]:
    plt.plot(x, chi2.pdf(x, nu), label=f'ν = {nu}')

plt.xlabel('χ²'); plt.ylabel('f(χ²)'); plt.legend(); plt.show()
