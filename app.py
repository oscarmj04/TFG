import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import mlflow.pytorch

# --- 1. CONFIGURACIÓN DEL MODELO ---
# ¡IMPORTANTE! Pon aquí el RUN ID de tu entrenamiento definitivo (el del Data Augmentation)
RUN_ID = "784d9ab1df6840caae47e77a719afbe2" 
model_uri = f"runs:/{RUN_ID}/modelo_resnet18_vides"

# Nombres "bonitos" para la interfaz web (en orden alfabético de tus carpetas)
clases = ['Black Rot', 'Esca', 'Sana', 'Spot']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Cargando modelo para la web...")
model = mlflow.pytorch.load_model(model_uri)
model.to(device)
model.eval()
print("✅ Modelo listo.")

# --- 2. FUNCIÓN DE PREDICCIÓN ---
def predecir_enfermedad(img):
    # Transformaciones de validación (¡las limpias!)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Convertimos la imagen y la pasamos a la gráfica
    img_t = transform(img).unsqueeze(0).to(device)

    # Inferencia
    with torch.no_grad():
        outputs = model(img_t)
        probabilidades = torch.nn.functional.softmax(outputs[0], dim=0)
    
    # Gradio necesita un diccionario con {Clase: Probabilidad}
    resultados = {clases[i]: float(probabilidades[i]) for i in range(len(clases))}
    return resultados

# --- 3. DISEÑO DE LA INTERFAZ WEB ---
interfaz = gr.Interface(
    fn=predecir_enfermedad,
    inputs=gr.Image(type="pil", label="Sube la foto de la hoja aquí", sources=["upload"] ),
    outputs=gr.Label(num_top_classes=4, label="Diagnóstico de la IA"),
    title="🍇 Asistente Vitícola IA",
    description="Sube una fotografía de una hoja de vid. La red neuronal evaluará si la hoja está sana o si presenta síntomas de Black Rot, Esca o Spot.",
    examples=[
        ["images/blackRot.png"], 
        ["images/spots.jpg"]
    ],
    theme="default" 
)

# --- 4. LANZAR LA WEB ---
if __name__ == "__main__":
    interfaz.launch()