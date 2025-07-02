import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torchvision.transforms as transforms


class CIFARDataset(Dataset):
    def __init__(self, image_folder, df_path, transform=None, label_map=None):
        self.image_folder = image_folder
        self.labels_df = pd.read_csv(df_path)
        self.transform = transform if transform else transforms.ToTensor()

        if label_map:
            # Si nos dan un mapa, lo usamos
            self.label2index = label_map
        else:
            # Si no, lo creamos (solo para el dataset de entrenamiento)
            unique_labels = sorted(self.labels_df['label'].unique())  # sorted() para reproducibilidad
            self.label2index = {label: idx for idx, label in enumerate(unique_labels)}

        # Es buena práctica tener también el mapa inverso
        self.index2label = {idx: label for label, idx in self.label2index.items()}

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_name = row['image_name']
        # Usamos el mapa consistente
        label = self.label2index[row['label']]

        img_path = os.path.join(self.image_folder, img_name)

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return image, label

dataset = CIFARDataset(
        image_folder=r"C:\Users\Mauro\OneDrive\Desktop\Deep Learn\Lab 1\CIFAR-10\CIFAR-10\train",
        df_path=r"C:\Users\Mauro\OneDrive\Desktop\Deep Learn\Lab 1\CIFAR-10\CIFAR-10\training_labels.csv"
)