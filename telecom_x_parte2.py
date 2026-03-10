# Desafío Telecom X - Parte 2
# Predicción de Evasión de Clientes (Churn)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#  CARGA DE DATOS TRATADOS 
df = pd.read_csv('datos_tratados.csv')

print("Datos cargados correctamente")
print(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
print(df.head())


# ELIMINACIÓN DE COLUMNAS IRRELEVANTES
# Aqui elimino customerID porque es solo un identificador, no ayuda a predecir

df = df.drop(columns=['customerID'])

print("Columna customerID eliminada")
print(f"Columnas restantes: {df.columns.tolist()}")