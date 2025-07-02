# dataloader.py (VERSIÓN FINAL Y LIMPIA)

from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from data import CIFARDataset

def get_dataloaders(batch_size, num_workers):
    """
    Esta función crea y devuelve los DataLoaders de entrenamiento y validación.
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    full_train_dataset = CIFARDataset(
        image_folder=r"C:\Users\Mauro\OneDrive\Desktop\Deep Learn\Lab 1\CIFAR-10\CIFAR-10\train",
        df_path=r"C:\Users\Mauro\OneDrive\Desktop\Deep Learn\Lab 1\CIFAR-10\CIFAR-10\training_labels.csv",
        transform=train_transform
    )

    train_size = int(0.75 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])
    val_subset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader

if __name__ == '__main__':
    print("Probando la función get_dataloaders...")
    train_dl, val_dl = get_dataloaders(batch_size=32, num_workers=0)
    print("¡Prueba completada con éxito!")