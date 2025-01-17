import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.config import get_script_config

# Load configuration
config = get_script_config('combine_office_tags')
ARUCO_DIR = config['paths']['ARUCO_DIR']
OFFICE_DIR = config['paths']['OFFICE_DIR']
OUTPUT_DIR = config['paths']['OUTPUT_DIR']

def process_office_image(image):
    """Process office image by first cropping then resizing to final size."""
    h, w = image.shape[:2]
    crop_size = config['image_processing']['crop_size']
    
    # Get random crop position for 800x800
    x = np.random.randint(0, w - crop_size + 1)
    y = np.random.randint(0, h - crop_size + 1)
    
    # Crop crop_size-sized square
    cropped = image[y:y+crop_size, x:x+crop_size]
    
    final = cv2.resize(cropped, tuple(config['image_processing']['final_size']))
    return final

def get_valid_position(office_img_size):
    """Get random valid position for tag that ensures it's fully visible."""
    tag_size = config['image_processing']['tag_size']
    max_x = office_img_size[1] - tag_size[0]
    max_y = office_img_size[0] - tag_size[1]
    
    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)
    
    return x, y

def combine_images():
    """Combine office images with ArUco tags and create CSV with bounding boxes."""
    csv_data = []

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    office_images = [f for f in os.listdir(OFFICE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    aruco_classes = [d for d in os.listdir(ARUCO_DIR) if os.path.isdir(os.path.join(ARUCO_DIR, d))]
    
    # Process each office image with fancy progress bar
    for office_idx, office_file in enumerate(tqdm(office_images, desc="Processing office images")):
        office_path = os.path.join(OFFICE_DIR, office_file)
        office_img = cv2.imread(office_path)
        
        office_img = process_office_image(office_img)
        
        for class_name in aruco_classes:
            class_dir = os.path.join(ARUCO_DIR, class_name)
            tag_files = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            # If dataset_augmentation wasn't run yet
            if not tag_files:
                continue
                
            # Random tag
            tag_file = np.random.choice(tag_files)
            tag_path = os.path.join(class_dir, tag_file)
            tag_img = cv2.imread(tag_path)
            
            # Just like Files 4 and 5
            tag_img = cv2.resize(tag_img, tuple(config['image_processing']['tag_size']))
            
            x, y = get_valid_position(office_img.shape)
            output_img = office_img.copy()
            tag_size = config['image_processing']['tag_size']
            output_img[y:y+tag_size[1], x:x+tag_size[0]] = tag_img
            
            output_filename = f"office{office_idx:04d}_class{class_name}_{tag_file}"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            cv2.imwrite(output_path, output_img)
            
            csv_data.append({
                'fileNames': os.path.join("combinedAugmented",output_filename),
                'bBox': [x, y, tag_size[0], tag_size[1]],  # [x, y, width, height]
                'class': class_name
            })
    
    df = pd.DataFrame(csv_data)
    df.to_csv(os.path.join(OUTPUT_DIR, 'BBData.csv'), index=False)

if __name__ == "__main__":
    print("Starting image combination process...")
    combine_images()
    print("Process completed!")
