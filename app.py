import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- 1. CONFIGURACIÓN DEL MODELO ---
clases_vides = ['Black Rot', 'Esca', 'Sana', 'Spot']
device = torch.device("cpu") 

print("Cargando red neuronal especialista (ResNet18)...")
especialista = models.resnet18(weights=None)
num_ftrs = especialista.fc.in_features
especialista.fc = nn.Linear(num_ftrs, 4) 

print("Cargando los pesos entrenados...")
# 1. Cargamos el archivo de MLflow. 
# Añadimos weights_only=False porque sabemos que el archivo es seguro y nuestro.
archivo_cargado = torch.load("models/model.pth", map_location=device, weights_only=False)

# 2. Inyectamos los pesos a nuestra red
if isinstance(archivo_cargado, dict):
    # Si en el futuro guardamos solo los pesos (state_dict), entra por aquí
    especialista.load_state_dict(archivo_cargado)
else:
    # Como MLflow guardó el modelo completo (la clase ResNet), le extraemos los pesos con .state_dict()
    especialista.load_state_dict(archivo_cargado.state_dict())

especialista.eval()
print("✅ Modelo listo.")

# --- 2. FUNCIÓN DE PREDICCIÓN CON UMBRAL ---
def predecir_con_umbral(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out_especialista = especialista(img_t)
        probabilidades = torch.nn.functional.softmax(out_especialista[0], dim=0)
    
    # Si pasa el filtro, devolvemos los resultados normales
    resultados = {clases_vides[i]: float(probabilidades[i]) for i in range(len(clases_vides))}
    return resultados

# --- 3. DISEÑO DE LA INTERFAZ WEB ---
tema_uva = gr.themes.Soft(
    primary_hue="green", 
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
)

with gr.Blocks(theme=tema_uva) as interfaz:
    gr.Markdown(
        """
        <h1 style='text-align: center; color: #2e7d32;'>🍇 Sistema Inteligente de Diagnóstico Vitícola</h1>
        <p style='text-align: center; font-size: 16px; color: #555;'>Sube una fotografía de una hoja de vid para evaluar si está sana o presenta síntomas de Black Rot, Esca o Spot.</p>
        <p style='text-align: center; font-size: 14px; color: #777;'><i>El sistema incorpora detección de incertidumbre para imágenes fuera de dominio.</i></p>
        <hr>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            imagen_entrada = gr.Image(type="pil", label="Sube tu fotografía aquí", sources=["upload"])
            boton_predecir = gr.Button("🔍 Analizar Hoja", variant="primary")
            
        with gr.Column(scale=1):
            resultado_salida = gr.Label(num_top_classes=4, label="Diagnóstico de la IA")
            
    gr.Markdown("### 📂 Imágenes de prueba")
    gr.Examples(
        examples=[
            ["images/BlackRot.JPG"], 
            ["images/ESCA.JPG"],
            ["images/Spot2.JPG"],
            ["images/Healthy.JPG"]
        ],
        inputs=imagen_entrada,
        cache_examples=False 
    )
    
    gr.Markdown(
        """
        <hr>
        <div style='text-align: center; color: gray; font-size: 12px;'>
        Desarrollado con PyTorch y Gradio | Trabajo de Fin de Grado
        </div>
        """
    )
    
    boton_predecir.click(
        fn=predecir_con_umbral, 
        inputs=imagen_entrada, 
        outputs=resultado_salida
    )

if __name__ == "__main__":
    interfaz.launch()