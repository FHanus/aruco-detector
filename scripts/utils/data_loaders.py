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
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.annotations_file = self._find_annotations_file(root_dir)
        
        if self.annotations_file is None:
            raise FileNotFoundError(f"No CSV file found: {root_dir}")
        
        # Get the image path and the box from the CSV
        self.samples = [
            (
                os.path.join(root_dir, row['fileNames']),
                [int(x) for x in row['bBox'].strip('[]').split(',')]
            )
            for _, row in pd.read_csv(self.annotations_file).iterrows()
        ]
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, bbox = self.samples[idx]

        image = read_image(path)         
        image = image.float() / 255.0    

        image = v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(image)

        # Convert bbox from [x_min, y_min, w, h] to [x_min, y_min, x_max, y_max]
        bbox_tensor = torch.tensor([
            bbox[0], 
            bbox[1], 
            bbox[0] + bbox[2], 
            bbox[1] + bbox[3]
        ], dtype=torch.float32)
        
        # Required format for the imported models
        target = {
            'boxes': bbox_tensor.unsqueeze(0),
            'labels': torch.ones(1, dtype=torch.int64)
        }
        
        return image, target
    
    def _find_annotations_file(self, directory):
        # Potentially overcomplicated way of searching for the CSV file within the subfolder of e.g. File4
        for path, _, files in os.walk(directory):
            csv_found = [f for f in files if f.lower().endswith('.csv')]
            if len(csv_found) == 1:
                return os.path.join(path,csv_found[0])
            elif len(csv_found) > 1:
                raise ValueError(f"Multiple CSV files in: {directory}.")
        return None

def create_dataloaders(root_dir, task='classification', batch_size=64, num_workers=0, train_split=0.7, val_split=0.15, shuffle=True, transform=None):
    if task == 'classification':
        dataset = ArucoClassificationDataset(root_dir, transform=transform)
    elif task == 'detection':
        dataset = ArucoDetectionDataset(root_dir)
    else:
        raise ValueError("Undefined task")
    
    n = len(dataset)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    n_test = n - n_train - n_val 

    train_data, val_data, test_data = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test]
    )

    if task == 'detection':
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, 
                                num_workers=num_workers, pin_memory=True,
                                collate_fn=lambda x: tuple(zip(*x)))
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              collate_fn=lambda x: tuple(zip(*x)))
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True,
                               collate_fn=lambda x: tuple(zip(*x)))
    else:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, 
                                num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
