from data import CIFARDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataloader import LABEL_MAP

# Mismas transformaciones que en validación
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.247, 0.243, 0.261))
])

test_dataset = CIFARDataset(
    image_folder=r"C:\Users\Mauro\OneDrive\Desktop\Deep Learn\Lab 1\CIFAR-10\CIFAR-10\test",
    df_path=r"C:\Users\Mauro\OneDrive\Desktop\Deep Learn\Lab 1\CIFAR-10\CIFAR-10\test_labels.csv",
    transform=test_transform,
    label_map = LABEL_MAP
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)