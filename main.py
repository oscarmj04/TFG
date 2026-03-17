import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import hydra
from omegaconf import DictConfig
import mlflow
import mlflow.pytorch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def calcular_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def validar(model, loader, criterion, device):
    model.eval()
    val_loss, val_acc = 0.0, 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_acc += calcular_accuracy(outputs, labels)
    return val_loss / len(loader), val_acc / len(loader)

@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig):
    # --- CAMBIO CLAVE: Volvemos a CUDA ---
    device = torch.device("cuda")
    print(f"Entrenando en: {device} ({torch.cuda.get_device_name(0) })")

    mlflow.set_experiment("Deteccion_Enfermedades_Vid")

    with mlflow.start_run():
        mlflow.log_params(cfg.entrenamiento)

    # 1. Transformaciones de ENTRENAMIENTO (Con Data Augmentation)
        train_transforms = transforms.Compose([
            transforms.Resize((cfg.datos.img_size, cfg.datos.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),           # Espejo aleatorio
            transforms.RandomRotation(degrees=15),            # Giros de hasta 15 grados
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # Cambios de luz
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 2. Transformaciones de VALIDACIÓN (El examen se hace con la foto original limpia)
        val_transforms = transforms.Compose([
            transforms.Resize((cfg.datos.img_size, cfg.datos.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 3. ACTUALIZA LOS DATALOADERS para que usen la variable correcta
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(cfg.datos.train, train_transforms), #train_transforms
            batch_size=cfg.entrenamiento.batch_size, 
            shuffle=True,
            num_workers=2, 
            pin_memory=True if device.type == 'cuda' else False
        )
        
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(cfg.datos.val, val_transforms),     #val_transforms
            batch_size=cfg.entrenamiento.batch_size,
            num_workers=2,
            pin_memory=True if device.type == 'cuda' else False
        )

        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, cfg.datos.num_clases)
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=cfg.entrenamiento.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(cfg.entrenamiento.epochs):
            model.train()
            train_loss, train_acc = 0.0, 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_acc += calcular_accuracy(outputs, labels)

            v_loss, v_acc = validar(model, val_loader, criterion, device)
            
            mlflow.log_metric("train_loss", train_loss/len(train_loader), step=epoch)
            mlflow.log_metric("val_accuracy", float(v_acc), step=epoch) # Forzamos float para MLflow
            
            print(f"Época {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {v_acc:.2%}")
        
        # Definir tus clases (ajusta los nombres a los tuyos)
        nombres_clases = ['Grape_Black_rot', 'Grape_Esca', 'Grape_healthy', 'Grape_spot']
        generar_matriz_confusion(model, val_loader, device, nombres_clases)

        # Guardamos el modelo en MLflow al terminar
        mlflow.pytorch.log_model(model, "modelo_resnet18_vides")
        print("Entrenamiento finalizado y modelo guardado en MLflow.")

def generar_matriz_confusion(model, loader, device, clases):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    cm = confusion_matrix(all_labels, all_preds)
    
    # Dibujar la matriz
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clases, yticklabels=clases)
    plt.xlabel('Predicción')
    plt.ylabel('Realidad')
    plt.title('Matriz de Confusión')
    plt.savefig('matriz_confusion.png')
    print("Matriz de confusión guardada como 'matriz_confusion.png'")

if __name__ == "__main__":
    train()