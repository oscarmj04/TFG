import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
import os

# --- 1. CONFIGURACIÓN GLOBAL ---
device = torch.device("cpu") 

def cargar_clases_dinamicas(ruta_dataset):
    """
    Lee dinámicamente los nombres de las clases basándose en las carpetas del dataset.
    Se ordenan alfabéticamente para coincidir exactamente con el mapeo que hace PyTorch.
    """
    if not os.path.exists(ruta_dataset):
        raise FileNotFoundError(f"🚨 ERROR CRÍTICO: No se encuentra la ruta '{ruta_dataset}'. El servidor no puede arrancar sin conocer las clases.")
        
    clases = sorted([d for d in os.listdir(ruta_dataset) if os.path.isdir(os.path.join(ruta_dataset, d))])
    
    if not clases:
        raise ValueError(f"🚨 ERROR CRÍTICO: La carpeta '{ruta_dataset}' está vacía. No hay clases para detectar.")
        
    return clases

# Ajusta esta ruta a donde tengas tus carpetas de entrenamiento 
RUTA_DATOS = "data/train" 

clases_vides = cargar_clases_dinamicas(RUTA_DATOS)
print(f"📂 Clases detectadas automáticamente para inferencia: {clases_vides}")

# Diccionario para guardar los modelos en la memoria RAM y no tener que recargarlos en cada clic
modelos_cacheados = {}

# --- 2. MOTOR DE CARGA DINÁMICA (FACTORY PATTERN) ---
def obtener_modelo(nombre_arq):
    # Si el modelo ya se usó antes, lo sacamos de la memoria súper rápido
    if nombre_arq in modelos_cacheados:
        return modelos_cacheados[nombre_arq]
    
    print(f"⚙️ Cargando {nombre_arq} por primera vez...")
    
    # 1. Construimos el "esqueleto" vacío según el modelo elegido
    if nombre_arq == "resnet18":
        modelo = models.resnet18(weights=None)
        modelo.fc = nn.Linear(modelo.fc.in_features, len(clases_vides))
        
    elif nombre_arq == "mobilenet_v2":
        modelo = models.mobilenet_v2(weights=None)
        modelo.classifier[1] = nn.Linear(modelo.last_channel, len(clases_vides))
        
    elif nombre_arq == "squeezenet":
        modelo = models.squeezenet1_0(weights=None)
        modelo.classifier[1] = nn.Conv2d(512, len(clases_vides), kernel_size=(1, 1), stride=(1, 1))
        modelo.num_classes = len(clases_vides)

    elif nombre_arq == "efficientnet_b0":
        modelo = models.efficientnet_b0(weights=None)
        modelo.classifier[1] = nn.Linear(modelo.classifier[1].in_features, len(clases_vides))

    else:
        raise ValueError("Arquitectura no soportada")

    # 2. Buscamos el archivo de pesos correspondiente
    ruta_pesos = f"models/modelo_{nombre_arq}.pth"
    if not os.path.exists(ruta_pesos):
        raise FileNotFoundError(f"❌ No se encontró el archivo {ruta_pesos}. ¡Asegúrate de haberlo copiado!")

    # 3. Le inyectamos los pesos (El cerebro entrenado)
    archivo_cargado = torch.load(ruta_pesos, map_location=device, weights_only=False)
    
    if isinstance(archivo_cargado, dict):
        modelo.load_state_dict(archivo_cargado)
    else:
        modelo.load_state_dict(archivo_cargado.state_dict())

    modelo.eval()
    modelo = modelo.to(device)
    
    # 4. Lo guardamos en la RAM para la próxima vez
    modelos_cacheados[nombre_arq] = modelo
    print(f"✅ {nombre_arq} listo para predecir.")
    
    return modelo

# --- 3. FUNCIÓN DE PREDICCIÓN ---
def predecir_con_umbral(img, nombre_arq):
    if img is None:
        return {"Error: Sube una imagen": 1.0}
        
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_t = transform(img).unsqueeze(0).to(device)
    
    # Pedimos el modelo al sistema de caché
    especialista = obtener_modelo(nombre_arq)

    with torch.no_grad():
        out_especialista = especialista(img_t)
        probabilidades = torch.nn.functional.softmax(out_especialista[0], dim=0)
    
    resultados = {clases_vides[i]: float(probabilidades[i]) for i in range(len(clases_vides))}
    return resultados

# --- 4. DISEÑO DE LA INTERFAZ WEB ---
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
        <hr>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            imagen_entrada = gr.Image(type="pil", label="Sube tu fotografía aquí", sources=["upload"])
            
            # NUEVO: Selector de modelos
            selector_modelo = gr.Dropdown(
                choices=["resnet18", "mobilenet_v2", "squeezenet", "efficientnet_b0"],
                value="squeezenet", # Por defecto
                label="🧠 Selecciona la Red Neuronal (Modelo)",
                info="Compara cómo piensan las distintas arquitecturas"
            )
            
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
    
    # Vinculamos el botón: Ahora le pasamos TANTO la imagen COMO el valor del desplegable
    boton_predecir.click(
        fn=predecir_con_umbral, 
        inputs=[imagen_entrada, selector_modelo], 
        outputs=resultado_salida
    )

if __name__ == "__main__":
    interfaz.launch()