import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import v2
import pandas as pd
import ast

class ArucoClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.samples = [] 
        self.transform = transform
        
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
        
        image = read_image(path)          # [C,H,W], dtype=uint8, [0,255]
        image = image.float() / 255.0     # [C,H,W], dtype=float32, [0,1]

        if self.transform:
            image = self.transform(image)

        image = v2.Grayscale(num_output_channels=3)(image)
        image = v2.Resize((224, 224))(image)
        image = v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(image)

        return image, label, path

class ArucoDetectionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.annotations_file = self._find_annotations_file(root_dir)
        if self.annotations_file is None:
            raise FileNotFoundError(f"No annotations CSV file found in directory: {root_dir}")
        self.img_labels = pd.read_csv(os.path.join(root_dir, self.annotations_file))

        self.samples = []
        image_files = os.listdir(root_dir)

        for f in image_files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_img_path = os.path.join(root_dir, f)
                self.samples.append(full_img_path)
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        filename = os.path.basename(path)

        # Find the row in the csv that corresponds to the image
        label_row = self.img_labels[self.img_labels['fileNames'] == filename].iloc[0]
        bbox = ast.literal_eval(label_row['bBox'])  # [x_min, y_min, w, h]

        image = read_image(path).float() / 255.0  # Normalize to [0,1]

        if self.transform:
            image = self.transform(image)

        ## Might need to change this based on the network
        image = v2.Grayscale(num_output_channels=3)(image)
        image = v2.Resize((224, 224))(image)
        image = v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(image)

        return image, bbox
    
    def _find_annotations_file(self, directory):
        csv_files = [file for file in os.listdir(directory) if file.lower().endswith('.csv')]
        if len(csv_files) == 1:
            return csv_files[0]
        elif len(csv_files) > 1:
            raise ValueError(f"Multiple CSV files in: {directory}.")
        else:
            return None

def create_dataloaders(root_dir, task='classification', batch_size=64, num_workers=0, train_split=0.7, val_split=0.15, shuffle=True, transform=None):
    if task == 'classification':
        dataset = ArucoClassificationDataset(root_dir, transform=transform)
    elif task == 'detection':
        dataset = ArucoDetectionDataset(root_dir, transform=transform)
    else:
        raise ValueError("Task must be either 'classification' or 'detection'")
    
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