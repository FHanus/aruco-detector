import os
import shutil

# Using the same paths as defined in train_classifier.py
DATA_DIR_FILE2 = "./data/File2/arucoBasic"
DATA_DIR_FILE3 = "./data/File3/arucoChallenging"
DATA_DIR_COMBINED = "./data/FileCustom1/arucoCombinedDif"

def combine_datasets():
    os.makedirs(DATA_DIR_COMBINED, exist_ok=True)
    
    # Basic dataset
    for class_folder in os.listdir(DATA_DIR_FILE2):
        src_class_path = os.path.join(DATA_DIR_FILE2, class_folder)
        dst_class_path = os.path.join(DATA_DIR_COMBINED, class_folder)
        
        os.makedirs(dst_class_path, exist_ok=True)
        
        # Copy and rename from File2
        for file in os.listdir(src_class_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                name, ext = os.path.splitext(file)
                new_name = f"{name}b{ext}"
                
                src_file = os.path.join(src_class_path, file)
                dst_file = os.path.join(dst_class_path, new_name)
                shutil.copy2(src_file, dst_file)
    
    # Challenging dataset
    for class_folder in os.listdir(DATA_DIR_FILE3):
        src_class_path = os.path.join(DATA_DIR_FILE3, class_folder)
        dst_class_path = os.path.join(DATA_DIR_COMBINED, class_folder)
        
        os.makedirs(dst_class_path, exist_ok=True)

        for file in os.listdir(src_class_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                name, ext = os.path.splitext(file)
                new_name = f"{name}ch{ext}"
                    
                src_file = os.path.join(src_class_path, file)
                dst_file = os.path.join(dst_class_path, new_name)
                shutil.copy2(src_file, dst_file)

if __name__ == "__main__":
    combine_datasets()