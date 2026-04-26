import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- 1. CONFIGURACIÓN DE MODELOS ---
clases_vides = ['Black Rot', 'Esca', 'Sana', 'Spot']
device = torch.device("cpu") # Hugging Face gratuito usa CPU

# A. CARGAR EL MODELO "GUARDABARRERA" (MobileNetV2)
# Este modelo viene pre-entrenado con ImageNet y conoce 1.000 categorías de objetos.
print("Cargando modelo guardabarrera (MobileNetV2)...")
portero = models.mobilenet_v2(weights='DEFAULT')
portero.eval()

# B. CARGAR TU MODELO ESPECIALISTA (ResNet18)
print("Cargando red neuronal especialista (ResNet18)...")
# Cargamos la arquitectura base de ResNet18
especialista = models.resnet18(weights=None)
num_ftrs = especialista.fc.in_features
especialista.fc = nn.Linear(num_ftrs, 4) # Adaptamos a nuestras 4 clases

print("Cargando los pesos entrenados del especialista...")
# Cargamos tu archivo .pth (Asegúrate de que la ruta sea correcta)
especialista.load_state_dict(torch.load("models/modelo_vid.pth", map_location=device))
especialista.eval()
print("✅ Pipeline modular listo.")

# Definimos los IDs de categorías de ImageNet relacionados con plantas/hojas
# En ImageNet, los índices del 936 al 997 suelen corresponder a vegetación.
IDS_PLANTAS = list(range(936, 998)) 


# --- 2. FUNCIÓN DEL PIPELINE DE INFERENCIA (2 PASOS) ---
def pipeline_diagnostico(img):
    # Transformaciones estándar (normalización de ImageNet)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # --- PASO 1: VALIDACIÓN (El Portero) ---
        # MobileNet evalúa qué objeto hay en la imagen
        out_portero = portero(img_t)
        _, pred_id_portero = torch.max(out_portero, 1)
        
        # Filtro de Seguridad: ¿Es una planta o algo verde?
        # Comprobamos si el ID detectado está en nuestra lista de "IDs Válidos"
        if pred_id_portero.item() not in IDS_PLANTAS:
             # Si no es una planta, detenemos el pipeline y devolvemos un mensaje de error claro
             # Adaptamos el formato para que gr.Label lo muestre como un resultado especial
             return {"⚠️ ERROR: Imagen no válida": 1.0, "Por favor, sube una foto de una hoja": 0.0}

        # --- PASO 2: DIAGNÓSTICO (El Especialista) ---
        # Si pasó el filtro, tu ResNet18 analiza la enfermedad de la vid
        out_especialista = especialista(img_t)
        probabilidades = torch.nn.functional.softmax(out_especialista[0], dim=0)
    
    # Formateamos el resultado de forma normal para el Label
    resultados = {clases_vides[i]: float(probabilidades[i]) for i in range(len(clases_vides))}
    return resultados


# --- 3. DISEÑO DE LA INTERFAZ WEB (BLOKS REVISADO) ---

# Mantenemos tu tema personalizado
tema_uva = gr.themes.Soft(
    primary_hue="green", 
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
)

with gr.Blocks(theme=tema_uva) as interfaz:
    # 1. Cabecera (HTML centrado)
    gr.Markdown(
        """
        <h1 style='text-align: center; color: #2e7d32;'>🍇 Sistema Inteligente de Diagnóstico Vitícola</h1>
        <p style='text-align: center; font-size: 16px; color: #555;'>Sube una fotografía de una hoja de vid para evaluar si está sana o presenta síntomas de Black Rot, Esca o Spot.</p>
        <p style='text-align: center; font-size: 14px; color: #777;'><i>Incluye filtro de seguridad para rechazar imágenes que no sean plantas.</i></p>
        <hr>
        """
    )
    
    # 2. Cuerpo principal en dos columnas
    with gr.Row():
        # Columna Izquierda: Entrada
        with gr.Column(scale=1):
            # Mantenemos la opción de subir desde archivos (sin webcam/portapapeles)
            imagen_entrada = gr.Image(type="pil", label="Sube tu fotografía aquí", sources=["upload"])
            boton_predecir = gr.Button("🔍 Analizar Hoja", variant="primary")
            
        # Columna Derecha: Resultado
        with gr.Column(scale=1):
            resultado_salida = gr.Label(num_top_classes=4, label="Diagnóstico del Pipeline")
            
    # 3. Ejemplos (Se mantienen y siguen funcionando igual)
    gr.Markdown("### 📂 Imágenes de prueba")
    gr.Examples(
        examples=[
                    ["images/BlackRot.JPG"], 
                    ["images/Spot.JPG"],
                    ["images/ESCA.JPG"],
                    ["images/Healthy.JPG"]
        ],
        inputs=imagen_entrada,
        cache_examples=False # Vital para Hugging Face Spaces
    )
    
    # 4. Footer
    gr.Markdown(
        """
        <hr>
        <div style='text-align: center; color: gray; font-size: 12px;'>
        Desarrollado con PyTorch y Gradio | Diseño Advanced con Blocks | Trabajo de Fin de Grado
        </div>
        """
    )
    
    # 5. La conexión (Event)
    # IMPORTANTE: Ahora conectamos el botón a la nueva función `pipeline_diagnostico`
    boton_predecir.click(
        fn=pipeline_diagnostico, # <--- Usamos el nuevo pipeline modular
        inputs=imagen_entrada, 
        outputs=resultado_salida
    )

if __name__ == "__main__":
    interfaz.launch()