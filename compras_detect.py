# -*- coding: utf-8 -*-
"""
Detección de compra de clientes - Regresión Logística
Ejecutar en Visual Studio Code: python ventas_detect.py
"""

# ============================================================
# 1. Importar librerías
# ============================================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 2. Crear el dataset
# ============================================================
data = {
    "edad": [22, 25, 47, 52, 46, 56, 23, 30, 34, 40],
    "salario": [2000, 2500, 5000, 6000, 5200, 6500, 2100, 3000, 3200, 4000],
    "compra": [0, 0, 1, 1, 1, 1, 0, 0, 0, 1]
}

df = pd.DataFrame(data)
print("=" * 50)
print("DATASET")
print("=" * 50)
print(df.to_string(index=False))


# ============================================================
# 3. Separar variables
# ============================================================
X = df[["edad", "salario"]]
y = df["compra"]


# ============================================================
# 4. Dividir los datos (80% train, 20% test)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nDatos de entrenamiento: {X_train.shape[0]} filas")
print(f"Datos de prueba:        {X_test.shape[0]} filas")


# ============================================================
# 5. Escalar los datos
# ============================================================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ============================================================
# 6. Entrenar el modelo
# ============================================================
model = LogisticRegression()
model.fit(X_train, y_train)

print(f"\nCoeficientes: edad={model.coef_[0][0]:.4f}, salario={model.coef_[0][1]:.4f}")
print(f"Intercepto:   {model.intercept_[0]:.4f}")


# ============================================================
# 7. Hacer predicciones
# ============================================================
y_pred = model.predict(X_test)


# ============================================================
# 8. Evaluar el modelo
# ============================================================
print("\n" + "=" * 50)
print("EVALUACIÓN DEL MODELO")
print("=" * 50)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred, zero_division=0))


# ============================================================
# 9. Prueba con nuevos clientes
# ============================================================
nuevos_clientes = pd.DataFrame({
    "edad": [22, 45, 55],
    "salario": [5700, 3000, 7000]
})

nuevos_scaled = scaler.transform(nuevos_clientes)
predicciones = model.predict(nuevos_scaled)
probabilidades = model.predict_proba(nuevos_scaled)

print("=" * 50)
print("PREDICCIONES PARA NUEVOS CLIENTES")
print("=" * 50)
for i, row in nuevos_clientes.iterrows():
    resultado = "COMPRA" if predicciones[i] == 1 else "NO COMPRA"
    prob = probabilidades[i][1] * 100
    print(f"  Edad: {row['edad']}, Salario: {row['salario']} -> {resultado} ({prob:.1f}%)")