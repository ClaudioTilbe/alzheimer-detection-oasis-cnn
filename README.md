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
### 4.1. Preprocesamiento de imágenes
### 4.2. División por paciente
### 4.3. Train, Validation y Test
### 4.4. Control de Data Leakage
### 4.5. Desbalance de clases

---

## 5. Arquitectura de la CNN
### 5.1. Arquitectura utilizada
### 5.2. Configuración del modelo

---

## 6. Entrenamiento
### 6.1. Parámetros de entrenamiento
### 6.2. Class Weights
### 6.3. Early Stopping

---

## 7. Resultados
### 7.1. Curvas de entrenamiento
### 7.2. Evaluación en Test
### 7.3. Classification Report
### 7.4. Matriz de Confusión

---

## 8. Precisión y realismo del modelo
### 8.1. Resultados de otros enfoques
### 8.2. El problema de buscar únicamente mayor Accuracy
### 8.3. Decisiones tomadas en este proyecto
### 8.4. Evaluación sobre pacientes no vistos

---

## 9. Limitaciones
### 9.1. Limitaciones del dataset
### 9.2. Limitaciones del modelo
### 9.3. Generalización

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
