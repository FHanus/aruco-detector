import os
import torch
import subprocess

def clean_gpu_memory():
    """Clean GPU memory by emptying cache and garbage collection
    (not that it helps much in terms of being able to use larger batch sizes)"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def ensure_folders_exist():
    """Ensure all required folders exist in data/"""
    required_folders = ['File1', 'File2', 'File3', 'File4', 'File5', 'File6']
    for folder in required_folders:
        folder_path = os.path.join('data', folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created folder: {folder_path}")
        else:
            print(f"Folder exists: {folder_path}")

def run_script(script_name):
    """Run a Python script and handle any errors"""
    try:
        # Add scripts directory to Python path so it can find the utils module
        env = os.environ.copy()
        env['PYTHONPATH'] = 'scripts:' + env.get('PYTHONPATH', '')
        subprocess.run(['python', os.path.join('scripts/src', script_name)], 
                      check=True, 
                      env=env)
        print(f"Successfully completed: {script_name}")
        clean_gpu_memory()
        print("GPU memory cleaned")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        raise

def main():
    # 1. Ensure required folders exist
    print("Step 1: Checking required folders...")
    #ensure_folders_exist()

    # 2. Run dataset augmentation
    print("\nStep 2: Running dataset augmentation...")
    #run_script('dataset_augmentation.py')

    # 3. Run combine office tags
    print("\nStep 3: Running combine office tags...")
    #run_script('combine_office_tags.py')

    # 4. Run classification experiments
    print("\nStep 4: Running classification experiments...")
    run_script('run_classification_experiments.py')

    # 5. Run train classifier
    print("\nStep 5: Running train classifier...")
    run_script('train_classifier.py')

    # 6. Run detection experiments
    print("\nStep 6: Running detection experiments...")
    run_script('run_detection_experiments.py')

    # 7. Run train detector
    print("\nStep 7: Running train detector...")
    run_script('train_detector.py')

    print("\nAll steps completed successfully!")

if __name__ == "__main__":
    main()
