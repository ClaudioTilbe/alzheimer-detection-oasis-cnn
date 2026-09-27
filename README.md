# Clasificación de imágenes MRI mediante CNN utilizando OASIS


## 1. Descripción del proyecto

Este proyecto implementa un modelo de **Deep Learning basado en una Convolutional Neural Network (CNN)** para clasificar imágenes de resonancia magnética cerebral en diferentes categorías relacionadas con la enfermedad de Alzheimer.

Para el desarrollo se utiliza el dataset [OASIS Alzheimer’s Detection](https://www.kaggle.com/datasets/ninadaithal/imagesoasis/data), disponible públicamente en Kaggle.

El proyecto aborda el proceso completo de construcción de un modelo de clasificación de imágenes, incluyendo la preparación del dataset, división de los datos, entrenamiento de la CNN y evaluación de sus resultados.

Además de buscar un modelo con buenos resultados de clasificación, se presta especial atención a la **calidad de la metodología de evaluación**, considerando las características y limitaciones propias del dataset.

---

## 2. Objetivo

El objetivo principal es desarrollar y evaluar una CNN capaz de clasificar imágenes MRI en cuatro categorías:

- **Non Demented**
- **Very Mild Dementia**
- **Mild Dementia**
- **Moderate Dementia**

El proyecto también busca analizar cómo las características del dataset pueden afectar los resultados de un modelo de Deep Learning, especialmente cuando existen múltiples imágenes correspondientes a un mismo paciente y una distribución desigual entre las clases.

Por este motivo, se utiliza una metodología de división basada en pacientes, buscando evitar que imágenes pertenecientes al mismo paciente aparezcan simultáneamente en los conjuntos de entrenamiento, validación y test.

---

## 3. Dataset

### 3.1. OASIS Alzheimer’s Detection

El dataset utilizado es [OASIS Alzheimer’s Detection](https://www.kaggle.com/datasets/ninadaithal/imagesoasis/data), disponible en Kaggle y basado en imágenes de resonancia magnética cerebral.

El conjunto utilizado contiene aproximadamente **86.437 imágenes**, distribuidas entre cuatro categorías:

| Clase | Imágenes | Pacientes | Promedio de imágenes por paciente |
|---|---:|---:|---:|
| Non Demented | 67.222 | 266 | 252,7 |
| Very mild Dementia | 13.725 | 58 | 236,6 |
| Mild Dementia | 5.002 | 21 | 238,2 |
| Moderate Dementia | 488 | 2 | 244,0 |
| **Total** | **86.437** | **347** | — |

Esta distribución muestra una diferencia considerable tanto en la cantidad de imágenes como en la cantidad de pacientes disponibles para cada categoría.

### 3.2. Características de las imágenes

Las imágenes originales utilizadas por el dataset tienen una resolución de **496 × 248 píxeles**.

Para reducir el coste computacional durante el entrenamiento, las imágenes son convertidas a RGB y redimensionadas a **128 × 128 píxeles** antes de ser utilizadas por la CNN.

Posteriormente, los valores de los píxeles son normalizados desde el rango original de **0–255** a un rango de **0–1**.

### 3.3. Distribución de pacientes e imágenes

Una característica importante del dataset es que las imágenes no representan pacientes independientes. Un mismo paciente puede tener cientos de imágenes asociadas.

En el dataset utilizado se identificaron:

- **266 pacientes** en `Non Demented`.
- **58 pacientes** en `Very mild Dementia`.
- **21 pacientes** en `Mild Dementia`.
- **2 pacientes** en `Moderate Dementia`.

Por lo tanto, la cantidad de imágenes por clase no debe interpretarse directamente como cantidad de casos independientes. La diferencia entre **86.437 imágenes y 347 pacientes** es especialmente relevante para la metodología de evaluación utilizada posteriormente.

### 3.4. Limitaciones del dataset

El dataset presenta algunas características que deben tenerse en cuenta al interpretar los resultados:

- Existe un fuerte desbalance entre las clases.
- La cantidad de pacientes varía considerablemente entre categorías.
- `Moderate Dementia` dispone únicamente de **2 pacientes** en el conjunto utilizado.
- Cada paciente puede aportar una gran cantidad de imágenes.
- Las imágenes de un mismo paciente están relacionadas entre sí, por lo que una división aleatoria basada únicamente en imágenes puede producir una evaluación poco representativa.

Estas características influyen directamente en las decisiones tomadas para la preparación, división y evaluación de los datos.

---

## 4. Preparación de los datos

La preparación de los datos es una etapa fundamental del proyecto, ya que las características del dataset hacen que una división convencional basada únicamente en imágenes pueda generar resultados poco representativos.

El procesamiento realizado contempla el **redimensionamiento y normalización de las imágenes**, la **identificación de pacientes a partir de los nombres de archivo**, la separación de los datos a nivel de paciente y el tratamiento del desbalance entre las clases.

El objetivo es que el modelo sea evaluado sobre pacientes que no hayan sido utilizados durante su entrenamiento.

### 4.1. Preprocesamiento de imágenes

Las imágenes originales del dataset tienen una resolución de **496 × 248 píxeles**.

Antes de ser utilizadas por la CNN, cada imagen pasa por las siguientes transformaciones:

1. Conversión a formato **RGB**.
2. Redimensionamiento a **128 × 128 píxeles**.
3. Conversión de la imagen a un arreglo de NumPy.
4. Normalización de los valores de los píxeles al rango **0–1**.

La normalización se realiza dividiendo los valores originales de los píxeles, que se encuentran en el rango `0–255`, entre `255`.

De esta manera, las imágenes utilizadas por el modelo tienen una forma de entrada de:

```text
128 × 128 × 3
```

El redimensionamiento permite reducir significativamente el coste computacional del entrenamiento manteniendo una representación suficiente para el objetivo experimental del proyecto.

### 4.2. División por paciente

Una de las decisiones metodológicas más importantes del proyecto es realizar la división de los datos **a nivel de paciente y no a nivel de imagen**.

El nombre de archivo permite identificar al paciente al que pertenece cada imagen. Por ejemplo:

```text
OAS1_0028_MR1_mpr-1_100.jpg
```

En este caso, `0028` corresponde al identificador utilizado para identificar al paciente.

Esto es importante porque un mismo paciente puede tener cientos de imágenes. Si las imágenes se dividieran aleatoriamente sin considerar esta relación, diferentes imágenes del mismo paciente podrían terminar en `train`, `validation` y `test`.

El modelo podría entonces encontrarse durante la evaluación con imágenes relacionadas con pacientes que ya estuvieron presentes durante el entrenamiento.

Para evitar esta situación, primero se agrupan las imágenes por paciente y posteriormente se realiza la división utilizando estos grupos.

### 4.3. Train, Validation y Test

La división de los datos se realiza manteniendo separados los pacientes entre los diferentes conjuntos.

Para las categorías `Non Demented`, `Very mild Dementia` y `Mild Dementia` se utiliza la siguiente estrategia:

* **15 % de los pacientes** se reserva para `Test`.
* Del grupo restante, **20 % de los pacientes** se reserva para `Validation`.
* Los pacientes restantes se utilizan para `Train`.

De esta forma, los conjuntos contienen pacientes independientes:

```text
Pacientes
    │
    ├── Train
    │
    ├── Validation
    │
    └── Test
```

El conjunto `Train` se utiliza para entrenar la CNN.

El conjunto `Validation` se utiliza durante el entrenamiento para supervisar el comportamiento del modelo sobre datos que no participan directamente en la actualización de sus pesos.

El conjunto `Test` se mantiene separado hasta la evaluación final y se utiliza para medir el comportamiento del modelo sobre pacientes que no fueron utilizados durante el entrenamiento.

#### Caso particular: Moderate Dementia

La categoría `Moderate Dementia` presenta una limitación importante: únicamente se identificaron **2 pacientes**.

Debido a esta cantidad extremadamente reducida, no es posible realizar una división independiente de `Train`, `Validation` y `Test` para esta categoría manteniendo una separación significativa entre los pacientes.

Por este motivo, se reserva:

* Un paciente para `Train`.
* Un paciente para `Test`.

La categoría `Moderate Dementia` no dispone de un conjunto de `Validation` independiente.

Esta limitación se tiene en cuenta posteriormente al interpretar los resultados del modelo.

### 4.4. Control de Data Leakage

La separación por paciente se complementa con controles específicos para detectar posibles casos de **Data Leakage**.

Después de realizar las divisiones se verifica que:

* Ningún paciente aparezca simultáneamente en `Train` y `Validation`.
* Ningún paciente aparezca simultáneamente en `Train` y `Test`.
* Ningún paciente aparezca simultáneamente en `Validation` y `Test`.

También se comprueba que un mismo identificador de paciente no aparezca asociado a diferentes categorías del dataset.

El objetivo de estos controles es garantizar que la evaluación del modelo se realice sobre pacientes independientes de aquellos utilizados durante el entrenamiento.

Esta metodología no elimina las limitaciones propias del dataset, pero reduce un riesgo importante de obtener métricas artificialmente elevadas debido a la presencia del mismo paciente en diferentes conjuntos.

### 4.5. Desbalance de clases

El dataset presenta un desbalance considerable entre las cuatro categorías.

La cantidad de imágenes disponibles por clase es:

| Clase              | Imágenes |
| ------------------ | -------: |
| Non Demented       |   67.222 |
| Very mild Dementia |   13.725 |
| Mild Dementia      |    5.002 |
| Moderate Dementia  |      488 |

Por lo tanto, `Non Demented` representa una cantidad de imágenes considerablemente mayor que `Moderate Dementia`.

En lugar de realizar un undersampling agresivo de las clases mayoritarias o un oversampling de las clases minoritarias, el entrenamiento utiliza **class weights**.

Los pesos se calculan automáticamente a partir de la distribución de las clases mediante `compute_class_weight` de `scikit-learn`.

Esto permite otorgar una mayor importancia a los errores cometidos sobre las clases con menor representación durante el entrenamiento, sin necesidad de duplicar físicamente las imágenes del dataset.

---

## 5. Arquitectura de la CNN

El modelo utilizado es una **Convolutional Neural Network (CNN)** implementada utilizando **TensorFlow/Keras**.

La arquitectura está orientada a la extracción progresiva de características visuales de las imágenes MRI.

A medida que la información atraviesa las capas convolucionales, la red obtiene representaciones cada vez más abstractas de las imágenes hasta llegar a la clasificación final en las cuatro categorías.

### 5.1. Arquitectura utilizada

La CNN recibe imágenes de entrada con dimensiones:

```text
128 × 128 × 3
```

La arquitectura está compuesta por cuatro bloques convolucionales, seguidos por una etapa de clasificación:

```text
Input
  │
  ├── Conv2D (32 filtros)
  ├── BatchNormalization
  └── MaxPooling2D
        │
        ├── Conv2D (64 filtros)
        ├── BatchNormalization
        └── MaxPooling2D
              │
              ├── Conv2D (128 filtros)
              ├── BatchNormalization
              └── MaxPooling2D
                    │
                    ├── Conv2D (256 filtros)
                    ├── BatchNormalization
                    └── MaxPooling2D
                          │
                          └── GlobalAveragePooling2D
                                │
                                ├── Dense (512)
                                ├── Dropout (0.5)
                                │
                                └── Dense (4)
                                      │
                                    Softmax
```

Las capas convolucionales utilizan progresivamente una mayor cantidad de filtros:

```text
32 → 64 → 128 → 256
```

Esto permite que las primeras capas trabajen con características más simples, mientras que las capas posteriores pueden representar patrones visuales de mayor complejidad.

`BatchNormalization` se utiliza después de las capas convolucionales para normalizar las activaciones y favorecer un entrenamiento más estable.

`MaxPooling2D` reduce progresivamente las dimensiones espaciales de las características extraídas.

Después del último bloque convolucional se utiliza `GlobalAveragePooling2D`, evitando la necesidad de transformar todo el mapa de características en un vector mediante `Flatten`.

Finalmente, una capa `Dense` de 512 neuronas realiza la combinación de las características extraídas antes de la clasificación final.

La última capa contiene **4 neuronas**, una por cada categoría del dataset, y utiliza `Softmax` para obtener las probabilidades asociadas a cada clase.

### 5.2. Configuración del modelo

La configuración utilizada para compilar la CNN es:

| Parámetro             | Configuración                   |
| --------------------- | ------------------------------- |
| Framework             | TensorFlow / Keras              |
| Tamaño de entrada     | `128 × 128 × 3`                 |
| Función de activación | ReLU                            |
| Capa de salida        | Softmax                         |
| Número de clases      | 4                               |
| Optimizador           | Adam                            |
| Learning Rate         | `0.001`                         |
| Función de pérdida    | Sparse Categorical Crossentropy |
| Métrica               | Accuracy                        |
| Dropout               | `0.5`                           |

La función de activación **ReLU** se utiliza en las capas convolucionales y en la capa `Dense` de 512 neuronas.

La capa de salida utiliza **Softmax**, ya que el problema consiste en clasificar cada imagen en una de cuatro categorías mutuamente excluyentes.

El optimizador **Adam** se configura con un learning rate de `0.001`.

La función de pérdida utilizada es `Sparse Categorical Crossentropy`, adecuada para un problema de clasificación multiclase donde las etiquetas se representan mediante valores enteros.

Además, durante el entrenamiento se incorporan los `class_weights` calculados previamente para compensar parcialmente el desbalance existente entre las categorías.

---

## 6. Entrenamiento

El entrenamiento de la CNN se realiza utilizando los datos separados previamente a nivel de paciente.

El modelo se entrena utilizando el conjunto `Train`, mientras que `Validation` se utiliza para supervisar su comportamiento durante el entrenamiento. El conjunto `Test` permanece separado hasta la evaluación final.

Debido al desbalance existente entre las clases, se utilizan **class weights** durante el entrenamiento. Además, se incorpora **Early Stopping** para evitar continuar entrenando cuando el rendimiento sobre `Validation` deja de mejorar.

### 6.1. Parámetros de entrenamiento

Los principales parámetros utilizados durante el entrenamiento son:

| Parámetro               | Configuración                   |
| ----------------------- | ------------------------------- |
| Épocas máximas          | 50                              |
| Batch size              | 32                              |
| Optimizador             | Adam                            |
| Learning Rate           | `0.001`                         |
| Función de pérdida      | Sparse Categorical Crossentropy |
| Métrica                 | Accuracy                        |
| Class Weights           | Sí                              |
| Early Stopping          | Sí                              |
| Paciencia (`patience`)  | 5                               |
| Restaurar mejores pesos | Sí                              |

El modelo puede entrenarse durante un máximo de **50 épocas**, aunque este límite no implica que necesariamente se ejecuten todas.

El proceso puede detenerse anticipadamente mediante `Early Stopping` cuando el rendimiento sobre el conjunto de validación deja de mejorar.

El tamaño de batch utilizado es de **32 imágenes**.

### 6.2. Class Weights

Debido al fuerte desbalance del dataset, durante el entrenamiento se utilizan pesos diferentes para cada clase.

Los pesos se calculan mediante `compute_class_weight` de `scikit-learn` utilizando la distribución de clases presente en el conjunto de entrenamiento.

Conceptualmente, las clases con menor representación reciben un peso mayor, mientras que las clases con mayor representación reciben un peso menor.

Esto modifica la contribución de cada clase al cálculo de la función de pérdida:

```text
Clase minoritaria
      ↓
Mayor peso
      ↓
Mayor impacto del error

Clase mayoritaria
      ↓
Menor peso
      ↓
Menor impacto relativo del error
```

El objetivo no es modificar la cantidad de imágenes disponibles, sino evitar que la clase mayoritaria domine el proceso de aprendizaje únicamente por tener una cantidad mucho mayor de ejemplos.

Esta estrategia permite conservar las imágenes disponibles para cada clase sin realizar un undersampling agresivo ni generar copias artificiales mediante oversampling.

### 6.3. Early Stopping

Durante el entrenamiento se utiliza `EarlyStopping` de Keras con los siguientes parámetros:

```python
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```

El criterio utilizado para determinar la evolución del entrenamiento es `val_loss`.

Si la pérdida de validación no mejora durante **5 épocas consecutivas**, el entrenamiento se detiene.

Además, mediante `restore_best_weights=True`, el modelo recupera los pesos correspondientes a la época que obtuvo el mejor resultado de validación, en lugar de conservar necesariamente los pesos de la última época ejecutada.

Esta estrategia ayuda a evitar continuar entrenando el modelo cuando el rendimiento sobre datos de validación deja de mejorar.

---

## 7. Resultados

La evaluación del modelo se realiza utilizando métricas obtenidas sobre el conjunto de `Test`.

Este conjunto contiene pacientes que fueron mantenidos separados de los utilizados para el entrenamiento y la validación.

Los resultados se analizan desde diferentes perspectivas, ya que una única métrica como `Accuracy` no permite describir completamente el comportamiento de un clasificador multiclase.

### 7.1. Curvas de entrenamiento

Durante el entrenamiento se registra la evolución de las métricas de `Train` y `Validation`.

Se analizan principalmente:

* `Accuracy` de entrenamiento.
* `Accuracy` de validación.
* `Loss` de entrenamiento.
* `Loss` de validación.

Estas curvas permiten observar cómo evoluciona el aprendizaje del modelo y detectar comportamientos como una divergencia importante entre entrenamiento y validación.

La gráfica generada durante el entrenamiento se incorpora al repositorio dentro de la carpeta `results/`.

```text
results/
└── training_curves.png
```

La interpretación de estas curvas se realiza conjuntamente con las métricas obtenidas sobre `Test`, ya que un buen comportamiento durante el entrenamiento no garantiza necesariamente una buena generalización sobre pacientes no vistos.

### 7.2. Evaluación en Test

Una vez finalizado el entrenamiento, el modelo se evalúa utilizando exclusivamente el conjunto `Test`:

```python
test_loss, test_acc = model.evaluate(
    x_test,
    y_test
)
```

La evaluación proporciona dos valores principales:

* **Test Loss**
* **Test Accuracy**

Estos valores representan el comportamiento del modelo sobre imágenes pertenecientes a pacientes que fueron separados previamente del proceso de entrenamiento.

Los valores obtenidos en la ejecución final del proyecto son:

```text
Test Loss: [VALOR]
Test Accuracy: [VALOR]
```

La `Test Accuracy` debe interpretarse junto con las métricas por clase y la matriz de confusión, especialmente debido al desbalance existente en el dataset.

### 7.3. Classification Report

Además de la Accuracy global, se genera un `Classification Report` utilizando las predicciones realizadas sobre el conjunto de test.

El reporte incluye, para cada categoría:

* **Precision**
* **Recall**
* **F1-score**
* **Support**

Estas métricas permiten analizar el comportamiento del modelo de forma individual para cada clase.

La `Precision` indica qué proporción de las predicciones realizadas como pertenecientes a una determinada clase fueron correctas.

El `Recall` indica qué proporción de los ejemplos reales de una clase fueron identificados correctamente.

El `F1-score` combina Precision y Recall en una única métrica.

El `Support` muestra la cantidad de ejemplos reales pertenecientes a cada clase dentro del conjunto evaluado.

El reporte obtenido en la ejecución final será incorporado aquí:

```text
[Classification Report]
```

El análisis por clase es especialmente relevante en este proyecto debido a la distribución desigual de los datos. Una Accuracy global puede ocultar diferencias importantes entre las distintas categorías.

### 7.4. Matriz de Confusión

La matriz de confusión permite observar con mayor detalle qué clases son identificadas correctamente y cuáles tienden a confundirse entre sí.

Las filas representan las clases reales y las columnas las clases predichas.

De esta manera, la diagonal principal representa las predicciones correctas, mientras que los valores fuera de la diagonal representan errores de clasificación.

La matriz generada durante la evaluación se incorpora al repositorio:

```text
results/
└── confusion_matrix.png
```

Su análisis permite identificar patrones de confusión entre:

* `Non Demented`
* `Very mild Dementia`
* `Mild Dementia`
* `Moderate Dementia`

Esto proporciona información que no puede obtenerse observando únicamente la Accuracy global.

---

## 8. Precisión y realismo del modelo

Uno de los objetivos del proyecto no es únicamente obtener una métrica de Accuracy elevada, sino analizar hasta qué punto los resultados representan el comportamiento del modelo sobre pacientes que no participaron en su entrenamiento.

Esta distinción es importante debido a la estructura particular del dataset, donde un mismo paciente puede aportar una gran cantidad de imágenes.

### 8.1. Resultados de otros enfoques

Durante el desarrollo del proyecto se analizaron diferentes enfoques de preparación y entrenamiento de los datos.

Entre las principales diferencias consideradas se encuentran:

* División basada únicamente en imágenes.
* División basada en pacientes.
* Diferentes estrategias para tratar el desbalance de clases.
* Diferentes configuraciones de la arquitectura de la CNN.

Los resultados de estos enfoques no se interpretan únicamente a partir de la Accuracy.

Una configuración puede producir una métrica global elevada y, al mismo tiempo, presentar diferencias importantes entre clases o beneficiarse de una división de datos que no representa adecuadamente la independencia entre pacientes.

Por este motivo, el enfoque utilizado en la versión final prioriza una evaluación metodológicamente más controlada.

> Los valores concretos de las experiencias anteriores se incorporarán en esta sección junto con sus condiciones de evaluación, para evitar comparar métricas obtenidas bajo metodologías diferentes.

### 8.2. El problema de buscar únicamente mayor Accuracy

La `Accuracy` representa la proporción total de predicciones correctas:

```text
Accuracy =
predicciones correctas / total de predicciones
```

Sin embargo, en un dataset desbalanceado esta métrica puede proporcionar una visión incompleta del comportamiento del modelo.

En este proyecto existe una diferencia considerable entre la cantidad de imágenes de las distintas categorías. Por ejemplo, `Non Demented` contiene muchas más imágenes que `Moderate Dementia`.

Por lo tanto, un modelo puede obtener una Accuracy global elevada mientras presenta un comportamiento considerablemente diferente entre las distintas clases.

Además, existe otra consideración importante: si imágenes del mismo paciente aparecen tanto en entrenamiento como en test, las métricas pueden reflejar parcialmente la capacidad del modelo para reconocer características específicas de pacientes ya conocidos, en lugar de medir exclusivamente su comportamiento sobre pacientes nuevos.

Por estas razones, la Accuracy se interpreta conjuntamente con:

* Precision.
* Recall.
* F1-score.
* Matriz de confusión.
* Distribución de las clases.
* Separación de pacientes entre los conjuntos.

### 8.3. Decisiones tomadas en este proyecto

A partir de las características observadas en el dataset, la metodología final adopta las siguientes decisiones:

1. **División de los datos a nivel de paciente.**
2. **Separación independiente de Train, Validation y Test cuando la cantidad de pacientes lo permite.**
3. **Control explícito de posibles solapamientos de pacientes.**
4. **Uso de Class Weights para tratar el desbalance.**
5. **Evaluación mediante métricas por clase además de Accuracy.**
6. **Uso de Early Stopping para seleccionar los mejores pesos según Validation Loss.**
7. **Evaluación final sobre pacientes no utilizados durante el entrenamiento.**

Estas decisiones buscan que las métricas obtenidas representen de una manera más adecuada el comportamiento del modelo frente a pacientes que no formaron parte del proceso de aprendizaje.

### 8.4. Evaluación sobre pacientes no vistos

Una característica central de la metodología utilizada es que el conjunto de `Test` está compuesto por pacientes que no fueron utilizados para entrenar la CNN.

Esto permite evaluar una situación más cercana al escenario en el que el modelo recibe imágenes pertenecientes a un paciente que no conoce.

La diferencia puede representarse de la siguiente manera:

```text
División por imagen

Paciente A ──┬── Train
             └── Test

        Riesgo de información compartida
```

Frente a:

```text
División por paciente

Paciente A ─────── Train
Paciente B ─────── Validation
Paciente C ─────── Test

        Pacientes independientes
```

Esta separación no garantiza que el modelo pueda generalizar correctamente a nuevos pacientes o a otros datasets, pero reduce una fuente importante de contaminación entre entrenamiento y evaluación.

Por esta razón, los resultados obtenidos sobre `Test` se interpretan como una evaluación sobre pacientes no vistos durante el entrenamiento, dentro de las limitaciones del dataset utilizado.

---

## 9. Limitaciones

Los resultados obtenidos deben interpretarse teniendo en cuenta tanto las características del dataset como las decisiones tomadas durante el desarrollo del modelo.

El objetivo de esta sección es documentar las principales limitaciones en lugar de presentar las métricas obtenidas como una medida definitiva del rendimiento que tendría el modelo fuera de este entorno experimental.

### 9.1. Limitaciones del dataset

La principal limitación está relacionada con la distribución de los pacientes entre las categorías.

Mientras que `Non Demented` cuenta con **266 pacientes**, `Moderate Dementia` dispone únicamente de **2 pacientes** en el conjunto utilizado.

Esta diferencia limita la capacidad de realizar una evaluación estadísticamente representativa para todas las clases.

Además, cada paciente puede aportar una gran cantidad de imágenes. Por este motivo, la cantidad total de imágenes no representa directamente la cantidad de casos independientes disponibles para el modelo.

El fuerte desbalance entre las categorías también puede afectar el aprendizaje y la interpretación de las métricas.

### 9.2. Limitaciones del modelo

La CNN utilizada es un modelo experimental desarrollado específicamente para este proyecto.

Aunque la arquitectura utiliza varias capas convolucionales, Batch Normalization, Global Average Pooling y Dropout, esto no implica que el modelo haya sido optimizado para su utilización clínica.

Entre las limitaciones del modelo se encuentran:

* La arquitectura fue entrenada exclusivamente con el dataset utilizado en este proyecto.
* No se realizó validación externa utilizando otro dataset independiente.
* El número de pacientes disponibles para algunas categorías es reducido.
* El modelo trabaja con imágenes redimensionadas a `128 × 128` píxeles.
* Las métricas obtenidas dependen de la distribución y características específicas de los datos utilizados.

Por lo tanto, los resultados deben interpretarse dentro del contexto experimental del proyecto.

### 9.3. Generalización

La capacidad de generalización del modelo no puede determinarse completamente utilizando únicamente el conjunto de test del mismo dataset.

Una evaluación más amplia requeriría, por ejemplo, probar el modelo sobre imágenes procedentes de una fuente independiente, con características diferentes a las utilizadas durante el entrenamiento.

Esto permitiría analizar si las características aprendidas por la CNN se mantienen cuando cambian factores como:

* Dataset de origen.
* Equipamiento utilizado para obtener las imágenes.
* Protocolos de adquisición.
* Características de los pacientes.
* Distribución de las categorías.

Por este motivo, los resultados obtenidos en este proyecto deben entenderse como resultados experimentales sobre el dataset OASIS utilizado y no como una demostración de rendimiento general sobre imágenes MRI de Alzheimer.

El modelo tampoco debe interpretarse como una herramienta de diagnóstico médico. Su finalidad dentro de este proyecto es **educativa, experimental y de investigación sobre clasificación de imágenes mediante Deep Learning**.

---

## 10. Uso del proyecto
### 10.1. Instalación
### 10.2. Entrenamiento
### 10.3. Predicción

---

## 11. Estructura del proyecto

---

## 12. Trabajo futuro

---

## 13. Conclusiones

---

## 14. Referencias
