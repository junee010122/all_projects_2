import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

class MetalDefectDataset(Dataset):
    def __init__(self, root_dir, dataset_type="train", image_size=224):
        """
        Loads images from `train`, `valid`, or `test` folders.
        """
        self.root_dir = os.path.join(root_dir, dataset_type)
        self.image_size = image_size
        self.image_paths = []
        self.labels = []

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.load_images()

    def load_images(self):
        """Loads images from dataset directory, assigning labels based on folder names."""
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Dataset folder '{self.root_dir}' does not exist!")

        class_names = sorted(os.listdir(self.root_dir))
        class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

        for cls in tqdm(class_names, desc="Loading Dataset", unit="class"):
            class_path = os.path.join(self.root_dir, cls)
            if not os.path.isdir(class_path): 
                continue

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.image_paths.append(img_path)
                self.labels.append(class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("L")  # Convert to grayscale
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

def get_dataloaders(root_dir, batch_size, image_size):
    print("Initializing data loaders...")
    train_dataset = MetalDefectDataset(root_dir, "train", image_size)
    valid_dataset = MetalDefectDataset(root_dir, "valid", image_size)
    test_dataset = MetalDefectDataset(root_dir, "test", image_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Data loaders ready!")
    return train_loader, valid_loader, test_loader

