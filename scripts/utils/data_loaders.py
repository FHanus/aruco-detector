import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import v2

class ArucoClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.samples = [] 
        self.transform = transform

        self.base_transforms = v2.Compose([
            v2.Grayscale(num_output_channels=3),  # To be compatible with AlexNet
            v2.Resize((64, 64))
        ])

        class_folders = sorted(os.listdir(self.root_dir))
        for class_str in class_folders:
            class_path = os.path.join(self.root_dir, class_str)

            class_index = int(class_str)
            image_files = os.listdir(class_path)

            for f in image_files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_img_path = os.path.join(class_path, f)
                    self.samples.append((full_img_path, class_index))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = read_image(path)

        image = self.base_transforms(image)

        if self.transform:
            image = self.transform(image)

        image = v2.ToDtype(torch.float32, scale=True)(image)
        return image, label, path

def create_dataloaders(root_dir, batch_size=8, num_workers=0, train_split=0.7, val_split=0.15, shuffle=True, transform=None):
    dataset = ArucoClassificationDataset(root_dir, transform=transform)
    n = len(dataset)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    n_test = n - n_train - n_val

    train_data, val_data, test_data = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test]
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, 
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False, 
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader