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

# ENCODING 
# Aqui convierto las columnas de texto a números con get_dummies

df = pd.get_dummies(df, drop_first=True)

print("Encoding aplicado correctamente")
print(f"Columnas después del encoding: {df.shape[1]}")
print(df.head())

# VERIFICACIÓN DE PROPORCIÓN DE CHURN 

proporcion = df['Churn'].value_counts(normalize=True) * 100

print("Proporción de Churn:")
print(f"  Se quedaron (0): {proporcion[0]:.1f}%")
print(f"  Se fueron   (1): {proporcion[1]:.1f}%")

# Visualizo la proporción
plt.figure(figsize=(6, 4))
plt.bar(['Se quedaron', 'Se fueron'], proporcion.values, color=['steelblue', 'salmon'])
plt.title('Proporción de cancelación de clientes')
plt.ylabel('Porcentaje (%)')
plt.savefig('grafica_proporcion_churn.png')
plt.show()
print("Gráfica guardada")

# NORMALIZACIÓN
# Aqui se normalizan solo las columnas numéricas continuas
# porque los modelos como Regresión Logística son sensibles a la escala

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
columnas_normalizar = ['tenure', 'Cargos_Mensuales', 'Cargos_Totales']
df[columnas_normalizar] = scaler.fit_transform(df[columnas_normalizar])

print("Normalización aplicada correctamente")
print(df[columnas_normalizar].describe())

# CORRELACIÓN Y SELECCIÓN DE VARIABLES
# Aqui voy a visualizar qué variables tienen más relación con Churn

correlacion = df.corr()['Churn'].sort_values(ascending=False)

print("Correlación con Churn:")
print(correlacion)

# Gráfica de las 10 variables más correlacionadas con Churn
plt.figure(figsize=(8, 5))
correlacion[1:11].plot(kind='bar', color='steelblue')
plt.title('Top 10 variables más relacionadas con Churn')
plt.ylabel('Correlación')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('grafica_correlacion.png')
plt.show()
print("Gráfica de correlación guardada")

# SEPARACIÓN DE DATOS
# X son las variables que usa el modelo para predecir
# y es lo que queremos predecir (Churn)

X = df.drop(columns=['Churn'])
y = df['Churn']

# Divido 80% entrenamiento y 20% prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Datos de entrenamiento: {X_train.shape[0]} filas")
print(f"Datos de prueba: {X_test.shape[0]} filas")

# CREACIÓN DE MODELOS

# Modelo 1: Regresión Logística (Este requiere normalización)
modelo_rl = LogisticRegression(max_iter=1000)
modelo_rl.fit(X_train, y_train)
pred_rl = modelo_rl.predict(X_test)

print("=== Regresión Logística ===")
print(f"Exactitud:  {accuracy_score(y_test, pred_rl):.2f}")
print(f"Precisión:  {precision_score(y_test, pred_rl):.2f}")
print(f"Recall:     {recall_score(y_test, pred_rl):.2f}")
print(f"F1-score:   {f1_score(y_test, pred_rl):.2f}")

# Modelo 2: Árbol de Decisión (Este no requiere normalización)
modelo_dt = DecisionTreeClassifier(random_state=42)
modelo_dt.fit(X_train, y_train)
pred_dt = modelo_dt.predict(X_test)

print("\n=== Árbol de Decisión ===")
print(f"Exactitud:  {accuracy_score(y_test, pred_dt):.2f}")
print(f"Precisión:  {precision_score(y_test, pred_dt):.2f}")
print(f"Recall:     {recall_score(y_test, pred_dt):.2f}")
print(f"F1-score:   {f1_score(y_test, pred_dt):.2f}")

# MATRICES DE CONFUSIÓN

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Regresión Logística
cm_rl = confusion_matrix(y_test, pred_rl)
axes[0].imshow(cm_rl, cmap='Blues')
axes[0].set_title('Regresión Logística')
axes[0].set_xlabel('Predicho')
axes[0].set_ylabel('Real')
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, cm_rl[i, j], ha='center', va='center', color='black')

# Árbol de Decisión
cm_dt = confusion_matrix(y_test, pred_dt)
axes[1].imshow(cm_dt, cmap='Blues')
axes[1].set_title('Árbol de Decisión')
axes[1].set_xlabel('Predicho')
axes[1].set_ylabel('Real')
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, cm_dt[i, j], ha='center', va='center', color='black')

plt.tight_layout()
plt.savefig('grafica_matrices_confusion.png')
plt.show()
print("Matrices de confusión guardadas")

# IMPORTANCIA DE VARIABLES 
# Con el árbol de decisión podemos ver qué variables pesan más en la predicción

importancia = pd.Series(modelo_dt.feature_importances_, index=X.columns)
importancia = importancia.sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
importancia.plot(kind='bar', color='steelblue')
plt.title('Top 10 variables más importantes - Árbol de Decisión')
plt.ylabel('Importancia')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('grafica_importancia_variables.png')
plt.show()
print("Gráfica de importancia guardada")

# CONCLUSIÓN 
print("\n=== CONCLUSIÓN ===")
print("El modelo de Regresión Logística tuvo mejor desempeño con 80% de exactitud.")
print("Las variables más importantes para predecir la evasión son:")
print("- Tiempo como cliente (tenure)")
print("- Tipo de contrato (mes a mes)")
print("- Tipo de internet (fibra óptica)")
print("- Cargos mensuales altos")
print("Se recomienda enfocarse en retener clientes nuevos con contrato mes a mes.")