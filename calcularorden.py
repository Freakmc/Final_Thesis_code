import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
import math


eeg_signal = pd.read_csv("path")

# Supongamos que df es tu DataFrame y que la señal está en la primera columna (por ejemplo, columna 1)
canal = eeg_signal.iloc[:, 3].values

# Función para calcular la varianza de los residuos para un modelo AR de orden p
def calculate_variance(residuals):
    return np.var(residuals)

# Función para calcular el criterio de Hannan-Quinn (HQ)
def hannan_quinn_criteria(V, N, p):
    return np.log(V) + (2 * p * np.log(np.log(N)) / N)

# Número de muestras en la señal
N = len(canal)

# Rango de órdenes p a evaluar
max_p = 50
hq_values = []

# Iterar sobre diferentes valores de p
for p in range(1, max_p + 1):
    model = AutoReg(canal, lags=p)
    model_fitted = model.fit()
    residuals = model_fitted.resid
    V = calculate_variance(residuals)
    hq = hannan_quinn_criteria(V, N, p)
    hq_values.append(hq)

# Encontrar el valor de p que minimiza HQ
optimal_p = np.argmin(hq_values) + 1  # +1 porque los índices de Python empiezan en 0

# Plotear los valores de HQ en función de p
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_p + 1), hq_values, marker='o')
plt.title('Criterio de Información de Hannan-Quinn (HQ) para diferentes órdenes p')
plt.xlabel('Orden evaluado (p)')
plt.ylabel('HQ')
plt.axvline(x=optimal_p, color='r', linestyle='--', label=f'Orden óptimo p = {optimal_p}')
plt.legend()
plt.grid(True)
plt.show()

print(f'El orden óptimo p es: {optimal_p}')
