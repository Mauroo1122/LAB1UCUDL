
import numpy as np
import random
#from dataloader import train_loader, val_loader
from dataloader import get_dataloaders
from model import SimpleCNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import csv
import os


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


csv_file = 'metricas_entrenamiento_GPU.csv'
carpeta_modelos = 'modelos'
os.makedirs(carpeta_modelos, exist_ok=True)

guardar_csv = True


if guardar_csv and not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'])


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Usando el dispositivo: {device}')

    print("Creando DataLoaders...")
    train_loader, val_loader = get_dataloaders(batch_size=128, num_workers=4)
    print("¡DataLoaders creados!")

    # Modelo

    model = SimpleCNN(nb_classes=10) # Usando la API
    model.to(device)



    # Loss y optimizador
    criterion = nn.CrossEntropyLoss()
    # El optimizador solo debe actualizar los pesos de la nueva capa (model.fc)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode = 'min',
        factor = 0.1,
        patience = 2,
        min_lr = 1e-6
    )

    #hiperparametros
    num_epochs = 10
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # ENTRENAMIENTO
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # VALIDACIÓN
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Mostrar resultados
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 50)

        if guardar_csv:
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    avg_train_loss,
                    train_acc,
                    avg_val_loss,
                    val_acc
                ])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ruta_mejor_modelo = os.path.join(carpeta_modelos, 'mejor_modelo_GPU.pth')
            torch.save(model.state_dict(), ruta_mejor_modelo)
            print(f'Mejor modelo guardado en época {epoch + 1} con val_acc = {val_acc:.2f}%')

    print('¡Entrenamiento completado!')
