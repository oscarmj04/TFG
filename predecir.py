import torch
from torchvision import transforms
from PIL import Image
import mlflow.pytorch

# 1. Configuración
device = torch.device("cuda")
model_uri = "runs:/784d9ab1df6840caae47e77a719afbe2/modelo_resnet18_vides" 
clases = ['Grape_Black_rot', 'Grape_Esca', 'Grape_healthy', 'Grape_spot']

# 2. Cargar modelo desde MLflow
model = mlflow.pytorch.load_model(model_uri)
model.to(device)
model.eval()

# 3. Preparar imagen
def predecir(ruta_imagen):
    img = Image.open(ruta_imagen).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_t = transform(img).unsqueeze(0).to(device)

    # 4. Inferencia
    with torch.no_grad():
        outputs = model(img_t)
        probabilidades = torch.nn.functional.softmax(outputs[0], dim=0)
        idx = torch.argmax(probabilidades).item()
        
    print(f"Predicción: {clases[idx]} ({probabilidades[idx]:.2%})")

# Úsalo así:
predecir("spots.jpg")