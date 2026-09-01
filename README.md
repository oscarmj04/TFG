🍇 Sistema Inteligente de Diagnóstico Vitícola – VitiCare

Este repositorio contiene el código fuente para el Trabajo de Fin de Grado (TFG) titulado **"VitiCare: Pipeline reproducible para la clasificación de enfermedades en hojas de vid mediante aprendizaje profundo"**, desarrollado para la Escuela Superior de Enxeñaría Informática (ESEI) de la Universidad de Vigo.

🎯 Objetivo del Proyecto

Desarrollar y desplegar un pipeline modular de Deep Learning (Redes Neuronales Convolucionales) capaz de clasificar imágenes de hojas de vid en 4 categorías: Sana, Black Rot, Esca y Spot. 
El proyecto abarca todo el ciclo de vida del software (MLOps), desde el entrenamiento parametrizado y la trazabilidad de experimentos hasta la puesta en producción mediante una interfaz web interactiva con Gradio.

📂 Estructura del Repositorio

- **`config/`**: Archivos de configuración YAML gestionados mediante Hydra para el desacoplamiento de hiperparámetros.
- **`models/`**: Directorio destinado a almacenar los archivos de pesos definitivos (`.pth`) de los modelos entrenados.
- **`images/`**: Recursos visuales e imágenes de prueba integradas en la interfaz de usuario.
- **`main.py`**: Script principal (Backend/Offline) para la orquestación del entrenamiento, aumento de datos y validación.
- **`app.py`**: Aplicación web (Frontend/Online) desarrollada con Gradio que implementa un sistema de caché (*Lazy Loading*) para la inferencia interactiva.
- **`requirements.txt`**: Listado de dependencias necesarias para la replicabilidad del entorno.

✅ Características Principales

- **Gestión Dinámica de Arquitecturas**: Soporte e interoperabilidad entre múltiples redes convolucionales (ResNet18, MobileNetV2, SqueezeNet, EfficientNet).
- **Trazabilidad y MLOps**: Integración con MLflow para el registro automatizado de métricas (Accuracy, F1-Score) y generación de matrices de confusión.
- **Configuración Paramétrica**: Uso de Facebook Hydra para modificar hiperparámetros de forma externa sin alterar el código fuente.
- **Interfaz Web Interactiva**: Despliegue optimizado con Gradio, visualización de métricas de confianza mediante gráficos de barras y carga bajo demanda en memoria RAM.

🚀 Cómo ejecutar el proyecto localmente

1. Clona este repositorio en tu equipo.

    git clone https://github.com/oscarmj04/TFG-VitiCare.git
    cd TFG-VitiCare
    
2. Crea y activa un entorno virtual de Python e instala las dependencias.

   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt

4. Ejecución del Backend (Entrenamiento y Trazabilidad):
   - Lanzamiento entrenamiento.
   
       python main.py
       
   - Lanzamiento con alteración de hiperparámetros.

       python main.py modelo.arquitectura=mobilenet_v2 entrenamiento.lr=0.005

5. Ejecución del Frontend (Interfaz Web de Inferencia):
   Asegúrate de tener los archivos de pesos (`.pth`) dentro de la carpeta `models/`.

     python app.py

🌐 Enlace Externo
Versión en producción disponible en Hugging Face Spaces: [https://huggingface.co/spaces/oscarmj04/t1ç](https://huggingface.co/spaces/oscarmj04/t1ç)
