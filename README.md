## 1. Descripción del proyecto

Este proyecto implementa un modelo de **Deep Learning basado en una Convolutional Neural Network (CNN)** para clasificar imágenes de resonancia magnética cerebral en diferentes categorías relacionadas con la enfermedad de Alzheimer.

Para el desarrollo se utiliza el dataset [OASIS Alzheimer’s Detection](https://www.kaggle.com/datasets/ninadaithal/imagesoasis/data), disponible públicamente en Kaggle.

El proyecto aborda el proceso completo de construcción y evaluación del modelo, incluyendo la preparación del dataset, división de los datos, entrenamiento, evaluación y generación de predicciones.

Además del desarrollo de la CNN, se presta especial atención a la metodología de división de los datos debido a que múltiples imágenes pertenecen a un mismo paciente y existe un fuerte desbalance entre las categorías.

---

## 2. Objetivo

El objetivo principal es desarrollar y evaluar una CNN capaz de clasificar imágenes MRI en cuatro categorías:

* **Non Demented**
* **Very Mild Dementia**
* **Mild Dementia**
* **Moderate Dementia**

El proyecto también busca analizar las limitaciones que presenta el dataset y cómo estas pueden afectar la evaluación y generalización del modelo.

Para reducir el riesgo de `Data Leakage`, la división de los datos se realiza a nivel de paciente, evitando que imágenes del mismo paciente aparezcan simultáneamente en los conjuntos de entrenamiento, validación y test.

---

## 3. Dataset

### 3.1. OASIS Alzheimer’s Detection

El dataset utilizado es [OASIS Alzheimer’s Detection](https://www.kaggle.com/datasets/ninadaithal/imagesoasis/data), disponible en Kaggle y basado en imágenes de resonancia magnética cerebral.

El conjunto utilizado contiene aproximadamente **86.437 imágenes**, distribuidas entre cuatro categorías:

| Clase              |   Imágenes | Pacientes | Promedio de imágenes por paciente |
| ------------------ | ---------: | --------: | --------------------------------: |
| Non Demented       |     67.222 |       266 |                             252,7 |
| Very mild Dementia |     13.725 |        58 |                             236,6 |
| Mild Dementia      |      5.002 |        21 |                             238,2 |
| Moderate Dementia  |        488 |         2 |                             244,0 |
| **Total**          | **86.437** |   **347** |                                 — |

La distribución presenta diferencias importantes tanto en la cantidad de imágenes como en la cantidad de pacientes disponibles para cada categoría.

### 3.2. Características de las imágenes

Las imágenes originales utilizadas por el dataset tienen una resolución de **496 × 248 píxeles**.

Para el entrenamiento, las imágenes son:

1. Convertidas a RGB.
2. Redimensionadas a **128 × 128 píxeles**.
3. Convertidas a arreglos de NumPy.
4. Normalizadas al rango **0–1**.

La entrada utilizada por la CNN tiene una dimensión de:

```text
128 × 128 × 3
```

### 3.3. Distribución de pacientes

Una característica importante del dataset es que un mismo paciente puede aportar cientos de imágenes.

En el conjunto utilizado se identificaron:

* **266 pacientes** en `Non Demented`.
* **58 pacientes** en `Very mild Dementia`.
* **21 pacientes** en `Mild Dementia`.
* **2 pacientes** en `Moderate Dementia`.

Por lo tanto, la cantidad de imágenes disponibles no representa directamente la cantidad de casos independientes utilizados por el modelo.

Esta diferencia es especialmente relevante para `Moderate Dementia`, donde las 488 imágenes pertenecen únicamente a dos pacientes.

### 3.4. Limitaciones del dataset

Las principales características que afectan al proyecto son:

* Fuerte desbalance entre las clases.
* Diferencias importantes en la cantidad de pacientes por categoría.
* Únicamente **2 pacientes** disponibles para `Moderate Dementia`.
* Gran cantidad de imágenes correspondientes a un mismo paciente.
* Riesgo de `Data Leakage` si la división se realiza únicamente a nivel de imagen.

Estas características son consideradas durante la preparación y evaluación de los datos.

---

## 4. Preparación de los datos

La preparación de los datos contempla el preprocesamiento de las imágenes, la identificación de pacientes, la división de los conjuntos y el tratamiento del desbalance entre las clases.

### 4.1. Preprocesamiento de imágenes

Las imágenes son procesadas mediante:

```text
496 × 248
     ↓
RGB
     ↓
128 × 128
     ↓
Normalización 0–1
```

La entrada final utilizada por la CNN es:

```text
128 × 128 × 3
```

### 4.2. División por paciente

La división de los datos se realiza **a nivel de paciente y no a nivel de imagen**.

El identificador del paciente se obtiene a partir del nombre de archivo. Por ejemplo:

```text
OAS1_0028_MR1_mpr-1_100.jpg
```

En este caso, `0028` corresponde al identificador utilizado para agrupar las imágenes pertenecientes al mismo paciente.

Las imágenes son agrupadas por paciente antes de realizar las divisiones de `Train`, `Validation` y `Test`.

### 4.3. Train, Validation y Test

Para `Non Demented`, `Very mild Dementia` y `Mild Dementia` se utiliza:

* **15 % de los pacientes** para `Test`.
* **20 % de los pacientes restantes** para `Validation`.
* Los pacientes restantes para `Train`.

La separación se realiza manteniendo los pacientes independientes entre los diferentes conjuntos.

```text
Pacientes
    │
    ├── Train
    ├── Validation
    └── Test
```

El conjunto de `Test` se mantiene separado hasta la evaluación final.

#### Caso particular: Moderate Dementia

`Moderate Dementia` dispone únicamente de **2 pacientes**.

Debido a esta limitación, se utiliza:

* Un paciente para `Train`.
* Un paciente para `Test`.
* Sin conjunto independiente de `Validation`.

Esta limitación debe considerarse al interpretar los resultados obtenidos para esta categoría.

### 4.4. Control de Data Leakage

Después de realizar las divisiones se comprueba que:

* No existan pacientes compartidos entre `Train` y `Validation`.
* No existan pacientes compartidos entre `Train` y `Test`.
* No existan pacientes compartidos entre `Validation` y `Test`.
* Un mismo paciente no aparezca asociado a diferentes categorías.

Estas comprobaciones permiten detectar posibles solapamientos antes de utilizar los datos para el entrenamiento.

### 4.5. Desbalance de clases

La distribución original presenta un fuerte desbalance:

| Clase              | Imágenes |
| ------------------ | -------: |
| Non Demented       |   67.222 |
| Very mild Dementia |   13.725 |
| Mild Dementia      |    5.002 |
| Moderate Dementia  |      488 |

Para el entrenamiento se utiliza una selección reducida de aproximadamente **16.000 imágenes**.

La distribución utilizada es:

| Clase              | Imágenes de Train |
| ------------------ | ----------------: |
| Non Demented       |             6.353 |
| Very mild Dementia |             6.353 |
| Mild Dementia      |             3.050 |
| Moderate Dementia  |               244 |
| **Total**          |        **16.000** |

`Mild Dementia` y `Moderate Dementia` utilizan todas las imágenes disponibles después de la separación de pacientes.

Las imágenes restantes necesarias para alcanzar las 16.000 se seleccionan entre `Non Demented` y `Very mild Dementia`, procurando distribuir la selección entre los diferentes pacientes.

No se duplican imágenes para compensar la clase `Moderate Dementia`.

Para compensar el desbalance restante durante el entrenamiento se utilizan **Class Weights** calculados mediante `compute_class_weight` de `scikit-learn`.

---

## 5. Arquitectura de la CNN

El modelo está implementado utilizando **TensorFlow/Keras**.

La arquitectura utilizada es:

```text
Input: 128 × 128 × 3

Conv2D 32
BatchNormalization
MaxPooling2D

Conv2D 64
BatchNormalization
MaxPooling2D

Conv2D 128
BatchNormalization
MaxPooling2D

Conv2D 256
BatchNormalization
MaxPooling2D

GlobalAveragePooling2D

Dense 512
Dropout 0.5

Dense 4
Softmax
```

Los bloques convolucionales utilizan:

```text
32 → 64 → 128 → 256 filtros
```

La capa final contiene cuatro neuronas correspondientes a las cuatro categorías del dataset.

### 5.1. Configuración del modelo

| Parámetro          | Configuración                   |
| ------------------ | ------------------------------- |
| Framework          | TensorFlow / Keras              |
| Entrada            | `128 × 128 × 3`                 |
| Número de clases   | 4                               |
| Activación         | ReLU                            |
| Capa de salida     | Softmax                         |
| Optimizador        | Adam                            |
| Learning Rate      | `0.001`                         |
| Función de pérdida | Sparse Categorical Crossentropy |
| Dropout            | `0.5`                           |
| Métrica            | Accuracy                        |

La arquitectura utiliza `BatchNormalization`, `MaxPooling2D` y `GlobalAveragePooling2D` en los bloques convolucionales, además de `Dropout` en la etapa de clasificación.

---

## 6. Entrenamiento

El modelo se entrena utilizando el conjunto `Train` y se supervisa mediante el conjunto `Validation`.

### 6.1. Parámetros de entrenamiento

| Parámetro            | Configuración                   |
| -------------------- | ------------------------------- |
| Épocas máximas       | 50                              |
| Batch size           | 32                              |
| Optimizador          | Adam                            |
| Learning Rate        | `0.001`                         |
| Función de pérdida   | Sparse Categorical Crossentropy |
| Métrica              | Accuracy                        |
| Class Weights        | Sí                              |
| Early Stopping       | Sí                              |
| Patience             | 5                               |
| Restore best weights | Sí                              |

### 6.2. Class Weights

Los pesos utilizados en la ejecución actual son:

```text
Clase 0: 0.6296
Clase 1: 0.6296
Clase 2: 1.3115
Clase 3: 16.3934
```

Los pesos se calculan automáticamente a partir de la distribución de `y_train`.

La utilización de `Class Weights` permite mantener las imágenes disponibles de las clases minoritarias sin realizar duplicación artificial de imágenes.

### 6.3. Early Stopping

El entrenamiento utiliza:

```python
EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```

El entrenamiento se detiene cuando `val_loss` deja de mejorar durante cinco épocas consecutivas.

Con `restore_best_weights=True`, se recuperan los pesos correspondientes a la mejor época de validación.

---

## 7. Resultados

La evaluación final se realiza sobre el conjunto de `Test`, compuesto por pacientes separados previamente del entrenamiento.

Se utilizan:

* Accuracy.
* Loss.
* Precision.
* Recall.
* F1-score.
* Matriz de confusión.

### 7.1. Curvas de entrenamiento

Durante el entrenamiento se registran las métricas de `Train` y `Validation`.

Las principales métricas observadas son:

* `accuracy`
* `val_accuracy`
* `loss`
* `val_loss`

Las curvas permiten analizar la evolución del entrenamiento y detectar diferencias entre el comportamiento del modelo sobre `Train` y `Validation`.

### 7.2. Evaluación en Test

La evaluación final se realiza mediante:

```python
test_loss, test_acc = model.evaluate(
    x_test,
    y_test
)
```

Los valores obtenidos en la ejecución final son:

```text
Test Loss: [VALOR]
Test Accuracy: [VALOR]
```

### 7.3. Classification Report

El modelo genera un `Classification Report` para las cuatro categorías:

```text
Non Demented
Very Mild Dementia
Mild Dementia
Moderate Dementia
```

El reporte incluye:

* Precision.
* Recall.
* F1-score.
* Support.

Los valores obtenidos en la ejecución final serán incorporados en esta sección.

### 7.4. Matriz de Confusión

También se genera una matriz de confusión para analizar las predicciones realizadas sobre el conjunto de `Test`.

```text
                  Predicción
              0     1     2     3
Real     0
         1
         2
         3
```

La matriz permite identificar las clases que presentan mayor cantidad de errores de clasificación.

---

## 8. Análisis de resultados

La ejecución actual muestra una diferencia importante entre el rendimiento obtenido sobre `Train` y `Validation`.

En las primeras ocho épocas se observaron los siguientes valores:

| Época | Train Accuracy | Validation Accuracy | Validation Loss |
| ----: | -------------: | ------------------: | --------------: |
|     1 |        69,74 % |             76,37 % |          2,1874 |
|     2 |        90,58 % |             66,56 % |          1,6369 |
|     3 |        95,76 % |             70,11 % |          1,5076 |
|     4 |        97,94 % |             59,86 % |          2,1074 |
|     5 |        98,30 % |             30,24 % |          4,7027 |
|     6 |        98,89 % |             73,05 % |          1,7648 |
|     7 |        98,37 % |             65,80 % |          1,9765 |
|     8 |        99,11 % |             56,48 % |          3,2090 |

La mejor `Validation Loss` registrada durante esta ejecución fue **1,5076 en la época 3**.

El comportamiento observado presenta una diferencia considerable entre `Train` y `Validation`, por lo que la ejecución muestra indicios de **overfitting**.

La interpretación de este resultado debe realizarse teniendo en cuenta las características del dataset, especialmente la cantidad reducida de pacientes disponibles para algunas categorías.

---

## 9. Limitaciones

Los resultados obtenidos deben interpretarse dentro de las características del dataset utilizado.

Las principales limitaciones son:

* Fuerte desbalance entre las clases.
* Cantidad desigual de pacientes por categoría.
* Solo **2 pacientes** disponibles para `Moderate Dementia`.
* Un único paciente de `Moderate Dementia` utilizado para `Train`.
* Un único paciente de `Moderate Dementia` utilizado para `Test`.
* Gran cantidad de imágenes pertenecientes a un mismo paciente.
* Ausencia de validación externa mediante otro dataset.
* Redimensionamiento de las imágenes originales a `128 × 128`.

La utilización de `Class Weights` permite compensar parcialmente el desbalance durante el entrenamiento, pero no soluciona la escasez de pacientes independientes.

Del mismo modo, la división por paciente reduce el riesgo de `Data Leakage`, pero no elimina las limitaciones estadísticas derivadas de disponer de pocos pacientes en determinadas categorías.

### Uso previsto

Este proyecto tiene finalidad **educativa y experimental**.

El modelo no constituye una herramienta médica ni debe utilizarse para realizar diagnósticos.

---

## 10. Uso del proyecto

El proyecto puede utilizarse para entrenar el modelo o realizar predicciones utilizando un modelo previamente entrenado.

### 10.1. Instalación

Clonar el repositorio:

```bash
git clone https://github.com/ClaudioTilbe/alzheimer-detection-oasis-cnn.git
cd alzheimer-detection-oasis-cnn
```

Crear un entorno virtual:

```bash
python -m venv .venv
```

En Windows:

```bash
.venv\Scripts\activate
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Principales tecnologías utilizadas:

* Python
* TensorFlow / Keras
* NumPy
* scikit-learn
* Matplotlib
* Seaborn
* Pillow

El dataset debe descargarse desde [OASIS Alzheimer's Detection en Kaggle](https://www.kaggle.com/datasets/ninadaithal/imagesoasis/data).

El dataset no se incluye en el repositorio debido a su tamaño.

### 10.2. Entrenamiento

El entrenamiento puede realizarse mediante:

```bash
python src/train.py
```

El proceso incluye:

1. Carga del dataset.
2. Identificación de pacientes.
3. División de los datos.
4. Preprocesamiento de imágenes.
5. Selección de imágenes para `Train`.
6. Cálculo de `Class Weights`.
7. Construcción de la CNN.
8. Entrenamiento.
9. Early Stopping.
10. Evaluación.
11. Generación de resultados.
12. Guardado del modelo.

### 10.3. Predicción

Una vez generado el modelo entrenado:

```bash
python src/predict.py ruta/a/la/imagen.jpg
```

El script carga el modelo y procesa la imagen utilizando las mismas dimensiones y normalización empleadas durante el entrenamiento.

El resultado corresponde a una de las cuatro categorías:

```text
Non Demented
Very Mild Dementia
Mild Dementia
Moderate Dementia
```

---

## 11. Estructura del proyecto

```text
alzheimer-detection-oasis-cnn/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebook/
│   └── Alzheimer_CNN.ipynb
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── models/
│   └── alzheimer_mri_cnn.keras
│
└── results/
    ├── training_curves.png
    └── confusion_matrix.png
```

### `README.md`

Documentación general del proyecto.

### `requirements.txt`

Dependencias necesarias para ejecutar el proyecto.

### `notebook/`

Notebook utilizado durante el desarrollo y experimentación de la CNN.

### `src/`

Scripts principales del proyecto:

* `train.py`: preparación, entrenamiento y evaluación del modelo.
* `predict.py`: carga del modelo y generación de predicciones.

### `models/`

Contiene el modelo entrenado en formato `.keras`.

### `results/`

Contiene las visualizaciones y resultados generados durante el entrenamiento y evaluación.

---

## 12. Trabajo futuro

Posibles líneas de desarrollo:

### Data Augmentation

Evaluar el impacto de técnicas de aumento de datos sobre la generalización del modelo.

### Transfer Learning

Comparar la CNN desarrollada desde cero con arquitecturas preentrenadas.

### Evaluación con datasets independientes

Probar el modelo sobre un dataset diferente para analizar su comportamiento fuera de la distribución utilizada durante el entrenamiento.

### Optimización de hiperparámetros

Experimentar con diferentes valores de:

* Learning Rate.
* Batch Size.
* Número de filtros.
* Número de capas.
* Dropout.
* Tamaño de entrada.

### Interpretabilidad

Incorporar técnicas como `Grad-CAM` para analizar las regiones de las imágenes asociadas a las predicciones del modelo.

### Interfaz gráfica

Integrar el modelo entrenado con una aplicación de escritorio desarrollada en **WPF**, permitiendo seleccionar imágenes y visualizar las predicciones generadas por la CNN.

---

## 13. Conclusiones

El proyecto implementa un flujo completo de clasificación de imágenes MRI mediante una **Convolutional Neural Network**, utilizando el dataset OASIS.

Uno de los principales aspectos del proyecto es la división de los datos a nivel de paciente, debido a que cada paciente puede aportar una gran cantidad de imágenes.

También se incorporan `Class Weights` para tratar el desbalance entre categorías y `Early Stopping` para controlar el entrenamiento.

La evaluación utiliza métricas por clase y matriz de confusión además de Accuracy, permitiendo analizar el comportamiento del modelo de forma más completa.

Los resultados obtenidos muestran diferencias importantes entre `Train` y `Validation`, indicando problemas de generalización en la configuración actual.

Las limitaciones del dataset, especialmente la disponibilidad de únicamente dos pacientes para `Moderate Dementia`, deben considerarse al interpretar las métricas obtenidas.

El proyecto tiene como finalidad demostrar el proceso de desarrollo y evaluación de un modelo de clasificación de imágenes mediante Deep Learning, documentando tanto sus resultados como las limitaciones encontradas durante su desarrollo.

---

## 14. Referencias

### Dataset

* **OASIS Alzheimer's Detection — Kaggle**
  https://www.kaggle.com/datasets/ninadaithal/imagesoasis/data

### Tecnologías

* **TensorFlow / Keras**
* **scikit-learn**
* **NumPy**
* **Pillow**
* **Matplotlib**
* **Seaborn**

### Conceptos utilizados

* Convolutional Neural Networks (CNN)
* Image Classification
* Deep Learning
* Patient-level Data Splitting
* Data Leakage
* Class Weights
* Early Stopping
* Confusion Matrix
* Precision
* Recall
* F1-score
* Model Generalization
