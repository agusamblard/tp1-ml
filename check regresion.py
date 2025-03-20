import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, X, y):
        # Asegurar que X es una matriz bidimensional
        X = np.array(X, dtype=np.float64)  # Asegurar tipo flotante para evitar problemas numéricos
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # Convertir a 2D si es necesario
        self.X = np.c_[np.ones(X.shape[0]), X]  # Agregar columna de unos
        self.y = np.array(y, dtype=np.float64)
        self.coef = None

    def fit_pseudo_inverse(self):
        self.coef = np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.y

    def fit_gradient_descent(self, lr=0.001, epochs=1000):
        self.coef = np.zeros(self.X.shape[1])
        m = len(self.y)
        for _ in range(epochs):
            gradient = (2/m) * self.X.T @ (self.X @ self.coef - self.y)
            self.coef -= lr * gradient
            
            # Verificar si hay valores NaN o Inf
            if np.isnan(self.coef).any() or np.isinf(self.coef).any():
                print("Error: coeficientes divergen. Reduce el learning rate.")
                return

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # Asegurar forma correcta
        X = np.c_[np.ones(X.shape[0]), X]  # Agregar columna de unos
        return X @ self.coef

    def print_coefs(self):
        print("Coeficientes:")
        for i, coef in enumerate(self.coef):
            print(f"Theta_{i}: {coef:.6f}")

# Cargar dataset con una característica
data = pd.read_csv("una_columna_normalizada.csv")
X = data["area"].values  # Convertir a array de NumPy
y = data["price"].values

# Normalizar X para estabilidad numérica
X = (X - np.mean(X)) / np.std(X)

# Inicializar y entrenar modelo
model = LinearRegression(X, y)
model.fit_gradient_descent()

# Imprimir coeficientes
if model.coef is not None:
    model.print_coefs()
    
    # Realizar predicciones
    predictions = model.predict(X)

    # Calcular error MSE
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    error = mse(y, predictions)
    print(f"Error cuadrático medio: {error:.6f}")

    # Graficar resultados
    plt.scatter(X, y, label="Datos reales", color="blue")
    plt.plot(X, predictions, label="Regresión lineal", color="red")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()
else:
    print("El modelo no pudo entrenarse debido a problemas numéricos.")

# Cargar dataset con múltiples características
data = pd.read_csv("multiples_columnas.csv")
X = data[["area", "age", "rooms"]].values
y = data["price"].values


# Inicializar y entrenar modelo
model = LinearRegression(X, y)
model.fit_pseudo_inverse()

# Imprimir coeficientes
if model.coef is not None:
    model.print_coefs()
    
    # Realizar predicciones
    predictions = model.predict(X)

    # Calcular error MSE
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    error = mse(y, predictions)
    print(f"Error cuadrático medio: {error:.6f}")

    plt.scatter(y, predictions, label="Predicciones vs Reales", color="blue")
    plt.xlabel("Valores reales")
    plt.ylabel("Predicciones")
    plt.plot([min(y), max(y)], [min(y), max(y)], linestyle="--", color="red")  # Línea ideal
    plt.legend()
    plt.show()
else:
    print("El modelo no pudo entrenarse debido a problemas numéricos.")


