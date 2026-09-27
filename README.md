# Clasificación de imágenes MRI mediante CNN utilizando OASIS

## 1. Descripción del proyecto

Este proyecto implementa un modelo de **Deep Learning basado en una Convolutional Neural Network (CNN)** para clasificar imágenes de resonancia magnética cerebral asociadas a diferentes categorías relacionadas con la enfermedad de Alzheimer.

El modelo fue desarrollado utilizando el dataset **OASIS Alzheimer’s Detection** y busca explorar el proceso completo de construcción y evaluación de un modelo de clasificación de imágenes, desde la preparación de los datos hasta el análisis de sus resultados.

Uno de los principales objetivos del proyecto es priorizar una metodología de evaluación que permita obtener resultados más representativos sobre pacientes no utilizados durante el entrenamiento, en lugar de centrarse únicamente en maximizar la precisión obtenida sobre el dataset.

---

## 2. Objetivo

El objetivo principal es desarrollar y evaluar una CNN capaz de clasificar imágenes MRI en cuatro categorías:

- **Non Demented**
- **Very Mild Dementia**
- **Mild Dementia**
- **Moderate Dementia**

Además de construir el modelo, el proyecto busca analizar las dificultades presentes en este tipo de datasets, especialmente aquellas relacionadas con el desbalance de clases y la presencia de múltiples imágenes correspondientes a un mismo paciente.

Por este motivo, se implementó una división de los datos basada en pacientes, evitando que imágenes pertenecientes al mismo paciente sean utilizadas simultáneamente en los conjuntos de entrenamiento, validación y test.

---

## 3. Dataset

### 3.1. OASIS Alzheimer’s Detection

El proyecto utiliza el dataset **OASIS Alzheimer’s Detection**, un conjunto de imágenes MRI utilizado para tareas de clasificación relacionadas con la enfermedad de Alzheimer.

El dataset contiene imágenes organizadas en cuatro categorías:

| Clase | Etiqueta |
|---|---:|
| Non Demented | 0 |
| Very Mild Dementia | 1 |
| Mild Dementia | 2 |
| Moderate Dementia | 3 |

Las imágenes son procesadas y redimensionadas a **128 × 128 píxeles**, manteniendo tres canales de color (RGB), antes de ser utilizadas por la CNN.

### 3.2. Origen y características

El dataset utilizado en este proyecto corresponde a una versión disponible públicamente a través de **Kaggle**, basada en datos de OASIS (*Open Access Series of Imaging Studies*).

Una característica importante del dataset es que contiene múltiples imágenes correspondientes a un mismo paciente. Esto significa que las imágenes no pueden considerarse completamente independientes entre sí.

Esta característica presenta un riesgo importante durante la división de los datos: si imágenes del mismo paciente aparecen tanto en entrenamiento como en test, el modelo puede obtener resultados artificialmente elevados al encontrarse durante la evaluación con información relacionada con pacientes que ya estuvieron presentes durante el entrenamiento.

### 3.3. Distribución de las clases

Las clases no presentan una distribución uniforme en el dataset. Algunas categorías contienen una cantidad considerablemente mayor de imágenes y pacientes que otras.

Esta diferencia representa un problema de **desbalance de clases**, especialmente en categorías como `Moderate Dementia`, que cuentan con una cantidad muy reducida de pacientes en comparación con `Non Demented`.

En lugar de descartar grandes cantidades de imágenes para igualar artificialmente las clases, este proyecto conserva los datos disponibles y utiliza **Class Weights** durante el entrenamiento para reducir el impacto del desbalance.

### 3.4. Limitaciones del dataset

El dataset presenta varias características que deben considerarse al interpretar los resultados del modelo:

- Existe un número muy diferente de pacientes entre las distintas categorías.
- Algunos pacientes poseen una gran cantidad de imágenes.
- La clase `Moderate Dementia` cuenta con solamente dos pacientes identificados en el dataset utilizado.
- La cantidad de imágenes no representa necesariamente la cantidad de pacientes.
- Los resultados obtenidos sobre este dataset no implican necesariamente que el modelo pueda generalizar a imágenes provenientes de otros datasets o de entornos clínicos reales.

Estas características influyen directamente en la metodología utilizada posteriormente para dividir los datos y evaluar el modelo.

## 4. Preparación de los datos
### 4.1. Preprocesamiento de imágenes
### 4.2. División por paciente
### 4.3. Train, Validation y Test
### 4.4. Control de Data Leakage
### 4.5. Desbalance de clases

## 5. Arquitectura de la CNN
### 5.1. Arquitectura utilizada
### 5.2. Configuración del modelo

## 6. Entrenamiento
### 6.1. Parámetros de entrenamiento
### 6.2. Class Weights
### 6.3. Early Stopping

## 7. Resultados
### 7.1. Curvas de entrenamiento
### 7.2. Evaluación en Test
### 7.3. Classification Report
### 7.4. Matriz de Confusión

## 8. Precisión y realismo del modelo
### 8.1. Resultados de otros enfoques
### 8.2. El problema de buscar únicamente mayor Accuracy
### 8.3. Decisiones tomadas en este proyecto
### 8.4. Evaluación sobre pacientes no vistos

## 9. Limitaciones
### 9.1. Limitaciones del dataset
### 9.2. Limitaciones del modelo
### 9.3. Generalización

## 10. Uso del proyecto
### 10.1. Instalación
### 10.2. Entrenamiento
### 10.3. Predicción

## 11. Estructura del proyecto

## 12. Trabajo futuro

## 13. Conclusiones

## 14. Referencias
