from torch import nn
import torch.nn.functional as F
import torch


class SimpleCNN(nn.Module):
    def __init__(self, nb_classes, img_size=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        self.max_pool = nn.MaxPool2d((2, 2), stride=2)

        self.dropout = nn.Dropout(p=0.5)  # Dropout al 50%

        vector_size = img_size // 2 * img_size // 2 * 64
        self.fc = nn.Linear(vector_size, nb_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(len(x), -1)
        x = self.dropout(x)
        return self.fc(x)

#model.py - Sugerencia
# class SimpleCNN(nn.Module):
#     def __init__(self, nb_classes):
#         super().__init__()
#         self.feature_extractor = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2), # -> 16x16
#
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2)  # -> 8x8
#         )
#
#         # Este 'adaptador' se asegura de que la salida sea siempre 128x1x1, sin importar el tamaño exacto
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
#
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Dropout(p=0.5),
#             nn.Linear(128, nb_classes) # El 128 viene de la última capa convolucional
#         )
#
#     def forward(self, x):
#         x = self.feature_extractor(x)
#         x = self.adaptive_pool(x)
#         x = self.classifier(x)
#         return x

# --- Añade esta clase a tu archivo model.py ---

# class SimpleMLP(nn.Module):
#     def __init__(self, nb_classes, img_size=32):
#         super().__init__()
#
#         # El input es la imagen aplanada: 32 (alto) * 32 (ancho) * 3 (canales de color)
#         input_size = img_size * img_size * 3
#
#         # Definimos una secuencia de capas
#         self.classifier = nn.Sequential(
#             # Capa 1
#             nn.Linear(input_size, 512),
#             nn.Dropout(p=0.5),  # Dropout entre la capa linear y la activación [cite: 19]
#             nn.ReLU(),  # Activación ReLu [cite: 19]
#
#             # Capa 2
#             nn.Linear(512, 256),
#             nn.Dropout(p=0.5),  # Dropout
#             nn.ReLU(),  # Activación ReLu
#
#             # Capa de Salida
#             nn.Linear(256, nb_classes)  # Capa final que clasifica en 10 clases
#         )
#
#     def forward(self, x):
#         # x llega con forma [batch_size, 3, 32, 32]
#
#         # 1. Aplanar la imagen en un vector largo
#         # Esto transforma la forma a [batch_size, 3072]
#         x = x.view(x.size(0), -1)
#
#         # 2. Pasar el vector aplanado por el clasificador
#         x = self.classifier(x)
#
#         return x