import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import hydra
from omegaconf import DictConfig
import mlflow
import mlflow.pytorch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, balanced_accuracy_score
import random
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import collections  # Para contar las imágenes del dataset

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
            
            # Guardamos las predicciones para calcular todas las métricas juntas al final
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    val_loss = val_loss / len(loader)
    
    # Cálculos de Scikit-Learn
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    b_acc = balanced_accuracy_score(all_labels, all_preds)
    
    return val_loss, acc, precision, recall, f1, b_acc

@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig):
    # --- 1. DETECCIÓN AUTOMÁTICA DE GPU/CPU ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"🚀 Iniciando entrenamiento en GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ No se detectó GPU. Iniciando entrenamiento en CPU (puede ser lento).")

    mlflow.set_experiment("Deteccion_Enfermedades_Vid")

    with mlflow.start_run():
        mlflow.log_params(cfg.entrenamiento)

        # 2. Transformaciones
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

        # --- 3. CARGA DE DATASETS Y CLASES DINÁMICAS ---
        train_dataset = datasets.ImageFolder(cfg.datos.train, train_transforms)
        val_dataset = datasets.ImageFolder(cfg.datos.val, val_transforms)
        
        # Obtenemos los nombres directamente de las carpetas
        nombres_clases = train_dataset.classes
        num_clases_detectadas = len(nombres_clases)

        # --- 4. GENERACIÓN DEL RESUMEN POR CONSOLA ---
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

        # 5. DataLoaders (Usando los datasets ya creados)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=cfg.entrenamiento.batch_size, 
            shuffle=True,
            num_workers=2, 
            pin_memory=True if device.type == 'cuda' else False
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.entrenamiento.batch_size,
            num_workers=2,
            pin_memory=True if device.type == 'cuda' else False
        )

        # 6. Modelo Dinámico (Usa num_clases_detectadas)
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_clases_detectadas)
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
            
        # 7. Matriz de Confusión Dinámica (Usando los nombres extraídos de las carpetas)
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
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clases, yticklabels=clases)
    plt.xlabel('Predicción')
    plt.ylabel('Realidad')
    plt.title('Matriz de Confusión')
    plt.savefig('matriz_confusion.png')
    print("Matriz de confusión guardada como 'matriz_confusion.png'")

if __name__ == "__main__":
    train()