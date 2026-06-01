import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import hydra
from omegaconf import DictConfig
import mlflow
import mlflow.pytorch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, balanced_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import collections
import random
import os

# --- 1. FUNCIÓN PARA FIJAR LA SEMILLA (REPRODUCIBILIDAD) ---
def set_seed(seed=42):
    """Congela todo el azar de Python, NumPy y PyTorch"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- 2. CONTROL ESTRICTO PARA DATALOADERS (TRABAJADORES SECUNDARIOS) ---
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def calcular_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def validar(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    val_loss = val_loss / len(loader)
    
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    b_acc = balanced_accuracy_score(all_labels, all_preds)
    
    return val_loss, acc, precision, recall, f1, b_acc

@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig):
    # Aplicamos la semilla estática al inicio
    set_seed(42)
    generador_estricto = torch.Generator()
    generador_estricto.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"🚀 Iniciando entrenamiento en GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ No se detectó GPU. Iniciando entrenamiento en CPU (puede ser lento).")

    mlflow.set_experiment("Deteccion_Enfermedades_Vid")

    with mlflow.start_run(run_name=str(cfg.modelo.arquitectura)):
        mlflow.log_params(cfg.entrenamiento)
        mlflow.log_param("seed", 42)
        mlflow.log_param("arquitectura", str(cfg.modelo.arquitectura)) # Registramos qué modelo usamos

        train_transforms = transforms.Compose([
            transforms.Resize((cfg.datos.img_size, cfg.datos.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        val_transforms = transforms.Compose([
            transforms.Resize((cfg.datos.img_size, cfg.datos.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_dataset = datasets.ImageFolder(cfg.datos.train, train_transforms)
        val_dataset = datasets.ImageFolder(cfg.datos.val, val_transforms)
        
        nombres_clases = train_dataset.classes
        num_clases_detectadas = len(nombres_clases)

        train_counts = collections.Counter([label for _, label in train_dataset.samples])
        val_counts = collections.Counter([label for _, label in val_dataset.samples])

        print(f"\n📂 Clases detectadas automáticamente ({num_clases_detectadas}): {nombres_clases}")
        print("\n📊 Resumen del Dataset:")
        print("Train:")
        for idx, count in train_counts.items():
            print(f"  - {nombres_clases[idx]}: {count} imágenes")
        print("Validation:")
        for idx, count in val_counts.items():
            print(f"  - {nombres_clases[idx]}: {count} imágenes")
        print("-" * 40 + "\n")

        # DataLoaders con Semilla estricta
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=cfg.entrenamiento.batch_size, 
            shuffle=True,
            num_workers=2, 
            pin_memory=True if device.type == 'cuda' else False,
            worker_init_fn=seed_worker,
            generator=generador_estricto
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.entrenamiento.batch_size,
            num_workers=2,
            pin_memory=True if device.type == 'cuda' else False,
            worker_init_fn=seed_worker,
            generator=generador_estricto
        )

        # --- 3. CONSTRUCTOR DE MODELOS (FACTORY PATTERN) ---
        nombre_arq = str(cfg.modelo.arquitectura)
        print(f"🏗️ Construyendo modelo: {nombre_arq}")
        
        if nombre_arq == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, num_clases_detectadas)
            
        elif nombre_arq == "mobilenet_v2":
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            model.classifier[1] = nn.Linear(model.last_channel, num_clases_detectadas)
            
        elif nombre_arq == "efficientnet_b0":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_clases_detectadas)
            
        elif nombre_arq == "squeezenet":
            model = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)
            # SqueezeNet clasifica con una capa Conv2d 1x1, no con una Linear
            model.classifier[1] = nn.Conv2d(512, num_clases_detectadas, kernel_size=(1, 1), stride=(1, 1))
            model.num_classes = num_clases_detectadas
            
        else:
            raise ValueError(f"❌ Arquitectura no reconocida: {nombre_arq}")

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

            v_loss, v_acc, v_prec, v_rec, v_f1, v_bacc = validar(model, val_loader, criterion, device)
            
            mlflow.log_metric("train_loss", train_loss/len(train_loader), step=epoch)
            mlflow.log_metric("val_accuracy", float(v_acc), step=epoch)
            mlflow.log_metric("val_precision", float(v_prec), step=epoch)
            mlflow.log_metric("val_recall", float(v_rec), step=epoch)
            mlflow.log_metric("val_f1", float(v_f1), step=epoch)
            mlflow.log_metric("val_balanced_acc", float(v_bacc), step=epoch)
            
            print(f"Época {epoch+1}: Loss: {train_loss/len(train_loader):.4f} | Acc: {v_acc:.2%} | Prec: {v_prec:.4f} | Rec: {v_rec:.4f} | F1: {v_f1:.4f} | B.Acc: {v_bacc:.4f}")
        
        generar_matriz_confusion(model, val_loader, device, nombres_clases)

        mlflow.pytorch.log_model(model, f"modelo_{cfg.modelo.arquitectura}_vides")
        print(f"✅ Entrenamiento finalizado y modelo ({cfg.modelo.arquitectura}) guardado en MLflow.")

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
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clases, yticklabels=clases)
    plt.xlabel('Predicción')
    plt.ylabel('Realidad')
    plt.title('Matriz de Confusión')
    plt.savefig('matriz_confusion.png')
    print("Matriz de confusión guardada como 'matriz_confusion.png'")

if __name__ == "__main__":
    train()