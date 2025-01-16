import os
import cv2
import numpy as np
from tqdm import tqdm

# Dataset paths
DATA_DIR_RAW = "./data/File1/arucoRaw"
DATA_DIR_AUGMENTED = "./data/FileCustom1/arucoAugmented"

def create_augmentation_transforms(total_transforms = 1000):
    """Generates a list of random transformation parameters.
    
    Creates transforms with:
    - Rotation: 0-360 degrees
    - Gaussian blur: 0-14 sigma
    - Salt-and-pepper noise: 0-10% of pixels
    - Scale: 50-100% of original size
    """
    transforms = []
    
    while len(transforms) < 1000:
        transforms.append({
            'rotation': np.random.uniform(0, 360),
            'blur': np.random.choice(np.arange(0, 14, 0.2)),
            'noise': np.random.choice(np.arange(0.0, 0.1, 0.025)),
            'scale': np.random.uniform(0.5, 1.0)
        })
    
    return transforms[:total_transforms]

def apply_augmentations(image, params, padding = 20):
    """Applies a set of augmentations to an image.
    
    Args:
        image: Input image
        params: Dict of transformation parameters
        padding: Border padding to prevent clipping during rotation
    
    Returns:
        Augmented image
    """
    h, w = image.shape[:2]
    
    # Create black background for placement
    black_bg = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Scale image whilst maintaining aspect ratio
    new_size = int(h * params['scale'])
    new_size = min(new_size, h - 2*padding, w - 2*padding)
    scaled = cv2.resize(image, (new_size, new_size))
    
    # Random placement with padding
    max_x = w - new_size - (2 * padding)
    max_y = h - new_size - (2 * padding)
    x_pos = padding + int(np.random.rand() * max_x)
    y_pos = padding + int(np.random.rand() * max_y)
    
    black_bg[y_pos:y_pos + new_size, x_pos:x_pos + new_size] = scaled
    
    # Apply rotation
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, params['rotation'], 1.0)
    image = cv2.warpAffine(black_bg, rotation_matrix, (w, h))
    
    # Apply Gaussian blur
    if params['blur'] > 0:
        image = cv2.GaussianBlur(image, (0, 0), params['blur'])
    
    # Apply salt-and-pepper noise
    if params['noise'] > 0:
        noise_mask = np.random.random(image.shape[:2]) < params['noise']
        image[noise_mask] = 255 - image[noise_mask]
    
    return image

def augment_dataset():
    """Creates augmented dataset by applying random transformations to raw images."""
    os.makedirs(DATA_DIR_AUGMENTED, exist_ok=True)
    
    image_files = [f for f in os.listdir(DATA_DIR_RAW) if f.endswith(('.png', '.jpg', '.jpeg'))]
    transforms = create_augmentation_transforms()
    
    for img_file in tqdm(image_files, desc="Processing images"):
        class_name = os.path.splitext(img_file)[0]
        class_dir = os.path.join(DATA_DIR_AUGMENTED, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        img_path = os.path.join(DATA_DIR_RAW, img_file)
        image = cv2.imread(img_path)
        
        for idx, transform_params in enumerate(transforms):
            augmented = apply_augmentations(image, transform_params)
            output_path = os.path.join(class_dir, f"{idx:04d}.png")
            cv2.imwrite(output_path, augmented)

if __name__ == "__main__":
    print("Starting dataset augmentation...")
    augment_dataset()
    print("Dataset augmentation completed!")
