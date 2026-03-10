# Telecom X — Parte 2: Predicción de Evasión de Clientes

# Propósito

El objetivo de este proyecto es predecir qué clientes tienen mayor probabilidad de cancelar el servicio (Churn) usando modelos de Machine Learning.
Con esta predicción, Telecom X puede anticiparse al problema y tomar decisiones estratégicas para retener clientes.

# Estructura del proyecto

- `telecom_x_parte2.py`: código principal con todo el análisis y modelos
- `datos_tratados.csv`: datos limpios exportados desde la Parte 1
- `grafica_proporcion_churn.png`: proporción de clientes que se fueron vs se quedaron
- `grafica_correlacion.png`: variables más relacionadas con la evasión
- `grafica_matrices_confusion.png`: comparación de los dos modelos
- `grafica_importancia_variables.png`: variables más importantes según el modelo

# Proceso de preparación de datos

1. **Eliminación de columnas irrelevantes**: Aqui eliminamos `customerID` porque no aporta información para predecir
2. **Encoding**: Aqui se convirtieron las columnas de texto a números con `get_dummies`
3. **Verificación de proporción de Churn**: el 73.4% de los clientes se quedaron y 26.6% de los clientes se fueron, sin desbalance grave
4. **Normalización**: Aqui se normalizaron `tenure`, `Cargos_Mensuales` y `Cargos_Totales` para la Regresión Logística
5. **Separación de datos**: 80% entrenamiento y 20% prueba

# Modelos utilizados

- **Regresión Logística**: modelo que requiere normalización
- **Árbol de Decisión**: modelo que no requiere normalización

# Resultados

**Regresión Logística**

- Exactitud: 80%: el programa acertó 8 de cada 10 clientes
- Precisión: 64%: cuando el programa predijo que un cliente se iría, acertó el 64% de las veces
- Recall: 54%: El programa detectó el 54% de los clientes que realmente se fueron
- F1-score: 0.59: rendimiento del programa es aceptable pero con margen de mejora

**Árbol de Decisión**

- Exactitud: 72%: el programa acertó 7 de cada 10 clientes
- Precisión: 47%: cuando el profgrama predijo que un cliente se iría, acertó el 47% de las veces
- Recall: 51%: el programa detectó el 51% de los clientes que realmente se fueron
- F1-score: 0.49: fue el rendimiento más bajo que la Regresión Logística

**Conclusión:**

En base a los datos concluimos que la Regresión Logística fue el mejor modelo.

# Conclusiones

La Regresión Logística fue el modelo con mejor desempeño, alcanzando un 80% de exactitud. Esto significa que de cada 100 clientes el modelo predijo correctamente 80 casos.

Sin embargo el Recall de 0.54 indica que el modelo solo detectó el 54% de los clientes que realmente se fueron, es decir, se perdió casi la mitad de los casos de evasión. Esto es un punto importante a mejorar en futuras versiones del modelo.

Las variables que más influyen en la evasión de clientes son:

- **Tiempo como cliente (tenure)** — los clientes nuevos se van más
- **Tipo de contrato** — los contratos mes a mes tienen mayor riesgo de cancelación
- **Tipo de internet** — los clientes con fibra óptica se van más
- **Cargos mensuales altos** — a mayor precio, mayor probabilidad de irse

Otro punto importante es que el modelo tiene dificultad para detectar a los clientes que se van — solo identificó el 54% de ellos. Para una empresa como Telecom X esto es crítico, porque cada cliente que el modelo no detecta es un cliente que se pierde sin que la empresa pueda hacer nada. Por eso se recomienda mejorar el modelo en el futuro con más datos o técnicas más avanzadas.

# Recomendación estratégica

Con base en los resultados del análisis se recomienda a Telecom X:

1. **Enfocarse en clientes nuevos**: la mayoría de las cancelaciones ocurren en los primeros meses. Implementar un programa de bienvenida o descuentos para los primeros 6 meses podría reducir la evasión.

2. **Revisar los precios de fibra óptica**: los clientes con este servicio se van más, posiblemente porque sienten que el precio no justifica el servicio. Se recomienda evaluar si el precio es competitivo en el mercado.

3. **Incentivar contratos anuales**: ofrecer beneficios a los clientes que pasen de contrato mes a mes a contrato anual, ya que estos tienen mucho menor probabilidad de cancelar.

4. **Revisar el método de pago electrónico**: los clientes que pagan con cheque electrónico se van más. Podría investigarse si hay problemas con este método de pago.

# Cómo ejecutar el proyecto

# 1. Instalar las librerías necesarias

Antes de ejecutar el código, instala las siguientes librerías desde la terminal:

pip install pandas matplotlib scikit-learn

# 2. Asegurarse de tener el archivo de datos

El archivo `datos_tratados.csv` debe estar en la misma carpeta que el código.

# 3. Ejecutar el código

Desde la terminal, dentro de la carpeta del proyecto:

python telecom_x_parte2.py
