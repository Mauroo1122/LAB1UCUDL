
import torch
#from model import SimpleCNN  # Asegúrate de que aquí importas la arquitectura que decidas usar
from testloader import test_loader
from dataloader import LABEL_MAP
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from torchvision import models
import torch.nn as nn

# --- CONFIGURACIÓN ---
# Carpeta para guardar los resultados
# CARPETA_RESULTADOS = 'resultados_RESNET'
# os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
#
# # Ruta al modelo entrenado
# RUTA_MODELO = 'modelos/mejor_modelo_RESNET.pth'
#
# # 1. Definir dispositivo
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Usando dispositivo: {device}")
#
# # 2. Definir y cargar modelo
# # Carga la arquitectura. Asegúrate de que coincida con la que guardaste.
# model = models.resnet50()
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, len(LABEL_MAP))
# model = model.to(device)
#
# if not os.path.exists(RUTA_MODELO):
#     print(f"Error: No se encontró el modelo en {RUTA_MODELO}")
#     print("Por favor, asegúrate de haber entrenado y guardado el modelo primero.")
#     exit()
#
# model.load_state_dict(torch.load(RUTA_MODELO))
# model.eval()
#
# # 3. Listas para almacenar resultados
# all_labels = []
# all_predictions = []
# all_scores = []  # Almacenaremos los scores (salidas del modelo) para la curva P-R
#
# # 4. Bucle de Evaluación
# with torch.no_grad():
#     for images, labels in tqdm(test_loader, desc="Evaluando en Test"):
#         images, labels = images.to(device), labels.to(device)
#
#         outputs = model(images)
#         scores = torch.softmax(outputs, dim=1)  # Aplicamos softmax para obtener probabilidades
#         _, predicted = torch.max(outputs, 1)
#
#         all_labels.extend(labels.cpu().numpy())
#         all_predictions.extend(predicted.cpu().numpy())
#         all_scores.extend(scores.cpu().numpy())
#
# all_labels = np.array(all_labels)
# all_predictions = np.array(all_predictions)
# all_scores = np.array(all_scores)
#
# # 5. Generar y guardar métricas
# print("\n--- Generando Reportes y Gráficas ---")
#
# # Nombres de las clases en el orden correcto
# class_names = [item[0] for item in sorted(LABEL_MAP.items(), key=lambda x: x[1])]
#
#
# # Reporte de clasificación
# report = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)
# report_df = pd.DataFrame(report).transpose()
# print("\n--- Reporte de Clasificación ---")
# print(report_df)
# ruta_reporte = os.path.join(CARPETA_RESULTADOS, 'test_classification_report.csv')
# report_df.to_csv(ruta_reporte)
# print(f"Reporte guardado en '{ruta_reporte}'")
#
# # Matriz de confusión
# cm = confusion_matrix(all_labels, all_predictions)
# plt.figure(figsize=(12, 10))
# sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
# plt.xlabel('Predicción')
# plt.ylabel('Etiqueta Real')
# plt.title('Matriz de Confusión')
# ruta_cm = os.path.join(CARPETA_RESULTADOS, 'test_confusion_matrix.png')
# plt.savefig(ruta_cm)
# plt.close()  # Cierra la figura para liberar memoria
# print(f"Matriz de confusión guardada en '{ruta_cm}'")
#
# # --- Curva de Precisión-Recall (Micro-Promedio y por Clase) ---
# # Binarizamos las etiquetas para el cálculo multi-clase
# lb = LabelBinarizer()
# lb.fit(all_labels)
# y_true_binarized = lb.transform(all_labels)
#
# # Diccionarios para guardar los resultados de cada clase
# precision = dict()
# recall = dict()
# average_precision = dict()
#
# # Calcular curva P-R para cada clase
# for i in range(len(class_names)):
#     precision[i], recall[i], _ = precision_recall_curve(y_true_binarized[:, i], all_scores[:, i])
#     average_precision[i] = average_precision_score(y_true_binarized[:, i], all_scores[:, i])
#
# # Calcular curva P-R micro-promedio
# precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_binarized.ravel(), all_scores.ravel())
# average_precision["micro"] = average_precision_score(y_true_binarized, all_scores, average="micro")
#
# # Graficar
# plt.figure(figsize=(10, 8))
# # Graficar la curva micro-promedio
# plt.plot(recall["micro"], precision["micro"],
#          label=f'Curva P-R micro-promedio (AP = {average_precision["micro"]:.2f})',
#          color='deeppink', linestyle=':', linewidth=4)
#
# # Graficar la curva P-R para cada clase
# for i in range(len(class_names)):
#     plt.plot(recall[i], precision[i], lw=2,
#              label=f'Curva P-R clase {class_names[i]} (AP = {average_precision[i]:.2f})')
#
# plt.xlabel('Recall (Sensibilidad)')
# plt.ylabel('Precision')
# plt.title('Curva de Precisión-Recall por Clase')
# plt.legend(loc="best")
# ruta_pr_curve = os.path.join(CARPETA_RESULTADOS, 'precision_recall_curve.png')
# plt.savefig(ruta_pr_curve)
# plt.close()  # Cierra la figura
# print(f"Curva de Precisión-Recall guardada en '{ruta_pr_curve}'")
#
# # Accuracy General
# accuracy = report_df.loc['accuracy', 'precision'] * 100
# print(f'\n🔍 Test Accuracy Final: {accuracy:.2f}%')

import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

# Importa tu modelo MLP y los loaders
from model import SimpleMLP
from testloader import test_loader
from dataloader import LABEL_MAP # <-- Se mantiene el import del LABEL_MAP

# --- CONFIGURACIÓN ---
# Ruta al modelo MLP entrenado
RUTA_MODELO = 'modelos/mejor_modelo_MLP.pth'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")


if not os.path.exists(RUTA_MODELO):
    print(f"Error: No se encontró el modelo en {RUTA_MODELO}")
    exit()


model = SimpleMLP(nb_classes=len(LABEL_MAP))
model.load_state_dict(torch.load(RUTA_MODELO, map_location=device))
model.to(device)
model.eval()
print(f"Modelo '{RUTA_MODELO}' cargado exitosamente.")


# --- BUCLE DE EVALUACIÓN ---
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluando MLP en Test"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())


# --- CÁLCULO Y REPORTE DE MÉTRICAS ---
print("\n" + "="*60)
print("--- Reporte de Clasificación Detallado ---")

class_names = [item[0] for item in sorted(LABEL_MAP.items(), key=lambda x: x[1])]

report = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)

report_df = pd.DataFrame(report).transpose()

print(report_df)

accuracy = report_df.loc['accuracy', 'support'] # Nota: la accuracy está en el campo 'support' en el dict de sklearn
print("\n" + "="*60)
print(f"🔍 Accuracy Final del MLP en el Conjunto de Test: {accuracy*100:.2f}%")
print("="*60)






