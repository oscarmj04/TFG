# 🍇 Sistema Inteligente de Diagnóstico Vitícola – TFG

Este repositorio contiene el código fuente para mi Trabajo de Fin de Grado (TFG), centrado en el desarrollo de un sistema de Inteligencia Artificial capaz de diagnosticar enfermedades en hojas de vid mediante visión por computador.

## 🎯 Objetivo del Proyecto

Desarrollar y desplegar un modelo de Deep Learning (Red Neuronal Convolucional) capaz de clasificar imágenes de hojas de vid en 4 categorías: **Sana, Black Rot, Esca y Spot**. 

El proyecto abarca desde el entrenamiento automatizado y monitorización del modelo, hasta la puesta en producción a través de una interfaz web interactiva que sirve como asistente de diagnóstico temprano para viticultores.

## 📂 Archivos Principales y Estructura

- `main.py` **(Pipeline de Entrenamiento)**
  - Script principal donde se entrena la arquitectura **ResNet18**.
  - Gestiona la carga de datos, el *Data Augmentation* y la evaluación del modelo.
  - Genera matrices de confusión y guarda automáticamente los pesos (`.pth`) del mejor entrenamiento.

- `app.py` **(Despliegue e Interfaz Web)**
  - Aplicación web desarrollada con **Gradio (Blocks)**.
  - Carga el modelo `.pth` en producción e implementa un pipeline de inferencia que analiza imágenes en tiempo real.
  - Incluye gestión de seguridad y control de fallos mediante el uso de umbrales lógicos.

- `src/` **(Código de Soporte)**
  - Directorio con módulos y funciones auxiliares utilizadas durante la limpieza de datos, pruebas o tareas menores del desarrollo.

- `models/` **(Pesos del Modelo)**
  - Directorio destinado a almacenar el archivo `modelo_vid.pth` con los pesos finales de la red neuronal tras el entrenamiento.

## ✅ Características Implementadas

- **Transfer Learning con PyTorch**
  - Adaptación de la arquitectura `ResNet18` (preentrenada en ImageNet) modificando su capa final para clasificar las clases específicas de enfermedades de la vid.
  
- **Monitorización Avanzada (MLflow & Hydra)**
  - Integración con **MLflow** para el registro automático de métricas (Accuracy, Loss) y control de versiones de los artefactos.
  - Uso de **Hydra** para la gestión estructurada de hiperparámetros a través de archivos de configuración (`config.yaml`).

- **Interfaz de Usuario Profesional (UI)**
  - Interfaz web estructurada en columnas usando `gr.Blocks`, con diseño en tonos verdes, ejemplos integrados y mensajes claros de diagnóstico.

- **Manejo de Imágenes Fuera de Distribución (OOD)**
  - *Work in Progress / Implementado:* Lógica de filtrado de imágenes para detectar y rechazar aquellas fotografías que no corresponden a plantas o que generan dudas en la red (ej: fotos de vehículos, teclados u objetos aleatorios), evitando falsos positivos.

## 🚀 Cómo ejecutar la aplicación localmente

1. Clona este repositorio.
2. Instala las dependencias necesarias (`torch`, `torchvision`, `gradio`, `mlflow`, `hydra-core`).
3. Asegúrate de tener el archivo de pesos `modelo_vid.pth` dentro de la carpeta `models/`.
4. Ejecuta el siguiente comando en la terminal:
   ```bash
   python app.py
## Extra
- Disponible version en la web en:
        https://huggingface.co/spaces/oscarmj04/t1


