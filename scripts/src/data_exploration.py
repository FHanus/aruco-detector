import argparse
import os, csv, ast, random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

WINDOW_NAME = "Dataset Analysis"

def parse_bbox_string(bbox_str):
    """Clean(er) way of returning bbox coordinates as tuple of floats"""
    return tuple(map(float, bbox_str.strip('"').split(',')))

def load_detection_results(csv_path):
    """Loads detection results from a CSV file
    
    Returns:
        List with file paths and bounding boxes
    """
    # Find the header row
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith('file_path,pred_boxes,true_boxes'):
                header_row = i
                break
    
    # Read CSV starting from header row
    df = pd.read_csv(csv_path, skiprows=header_row)
    results = []
    for _, row in df.iterrows():
        results.append({
            'file_path': row['file_path'],
            'pred_boxes': parse_bbox_string(row['pred_boxes']),
            'true_boxes': parse_bbox_string(row['true_boxes'])
        })
    return results

def draw_bbox(img, bbox, color, label=None):
    """Draws a bounding box to show the predictions"""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    if label:
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, color, 2)
    return img

def update_detection_analysis(detection_results, current_idx=0):
    """Necessary for the pop-up window to work"""
    result = detection_results[current_idx]
    img_path = result['file_path']
    if not os.path.isabs(img_path):
        img_path = os.path.join(os.getcwd(), img_path)
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image {img_path}")
        return
    
    display_img = img.copy()
    
    draw_bbox(display_img, result['pred_boxes'], (255, 0, 0))
    
    info_lines = [
        "Detection Analysis",
        f"File: {img_path}",
        "",
        "Bounding Boxes:",
        f"True: {result['true_boxes']}",
        f"Pred: {result['pred_boxes']}"
    ]
    
    text_panel = create_text_panel(info_lines)
    target_height = 400
    
    # Create composite display with uniform height
    display_img_resized = cv2.resize(display_img, (400, target_height))
    hist_img = create_histogram_image(img)
    hist_img_resized = cv2.resize(hist_img, (400, target_height))
    text_panel_resized = cv2.resize(text_panel, (600, target_height))
    
    composite = np.hstack([display_img_resized, hist_img_resized, text_panel_resized])
    cv2.imshow(WINDOW_NAME, composite)
    return current_idx

def load_image_paths(dataset_path):
    """Recursively finds all image files in the path"""
    image_paths = []
    for root, _, files in os.walk(dataset_path):
        for file in sorted(files):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(root, file))
    return image_paths

def load_dataset(dataset_path):
    """Loads image paths and corresponding bounding boxes"""
    image_paths = load_image_paths(dataset_path)
    bboxes = {}

    csv_path = os.path.join(dataset_path, "BBData.csv")
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            # Parse CSV to get bounding boxes
            for row in csv.DictReader(f):
                fname = os.path.basename(row['fileNames'])
                bboxes[fname] = ast.literal_eval(row['bBox'])

    return image_paths, bboxes

def create_histogram_image(image, size=(400, 400)):
    """Creates a histogram visualisation"""
    width, height = size
    # Create a Matplotlib figure and axis with the specified size
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    
    # Plot histogram based on image type
    if image.ndim == 2:
        ax.hist(image.ravel(), bins=256, range=(0, 256),
                color='gray', alpha=0.7, label='Gray')
    else:
        for color, channel in zip(('b', 'g', 'r'), range(3)):
            ax.hist(image[:, :, channel].ravel(), bins=256, range=(0, 256),
                    color=color, alpha=0.5, label=color.upper())
    
    ax.set(xlabel='Pixel Intensity', ylabel='Frequency')
    plt.tight_layout()

    # Convert matplotlib figure to cv2 image
    fig.canvas.draw()
    rgba_buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = rgba_buffer.reshape((width, height, 4))
    plt.close(fig)
    
    hist_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.resize(hist_img, (width, height))

def create_text_panel(text_lines, panel_size=(600, 400)):
    """Creates an image containing text information for display"""
    width, height = panel_size
    text_panel = np.full((height, width, 3), 255, dtype=np.uint8)
    for i, line in enumerate(text_lines):
        text_start_position = (10, 50 + i * 25)
        cv2.putText(text_panel, line, text_start_position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return text_panel

def update_analysis(image_paths, bboxes, dataset_name, is_combined):
    """Displays image, histogram and image statistics"""
    random_index = random.randrange(len(image_paths))
    path = image_paths[random_index]
    sample_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    display_img = sample_img.copy()
    bbox = None

    # Handle combined datasets with bounding boxes
    if is_combined:
        fname = os.path.basename(path)
        bbox = bboxes.get(fname)

        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        if sample_img.ndim == 2:
            display_img = cv2.cvtColor(sample_img, cv2.COLOR_GRAY2BGR)

    # Calculate image statistics
    sample_mean = float(np.mean(sample_img))
    sample_std = float(np.std(sample_img))
    
    hist_img = create_histogram_image(sample_img)
    height, width = display_img.shape[:2]

    # Prepare info panel text
    info_lines = [
        f"Dataset: {dataset_name}",
        f"File: {path}",
        "",
        "Image Statistics:",
        f"Resolution: {width} x {height}",
        f"Mean: {sample_mean:.2f}",
        f"Std: {sample_std:.2f}"
    ]
    if is_combined and bbox:
        info_lines += [
            "Bounding Box:",
            f"Position: ({x}, {y})",
            f"Size: {w}x{h}"
        ]

    text_panel = create_text_panel(info_lines)
    target_height = 400

    # Create composite display with uniform height
    display_img_resized = cv2.resize(display_img, (400, target_height))
    hist_img_resized = cv2.resize(hist_img, (400, target_height))
    text_panel_resized = cv2.resize(text_panel, (600, target_height))

    composite = np.hstack([display_img_resized, hist_img_resized, text_panel_resized])
    cv2.imshow(WINDOW_NAME, composite)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Dataset analysis.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset",
        choices=[
            "raw", "basic", "challenging", "combinedbasic", 
            "combinedchallenging", "office",
            "FileCustom1", "FileCustom2"
        ],
        help="Specify which dataset to use for exploration."
    )
    group.add_argument(
        "--detection",
        help="Path to detection results CSV file."
    )

    args = parser.parse_args()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    if args.detection:
        # Detection results analysis mode
        detection_results = load_detection_results(args.detection)
        current_idx = 0
        total_samples = len(detection_results)
        
        current_idx = update_detection_analysis(detection_results)
        print(f"Showing result 1/{total_samples}")
        print("Press 'n' for next image, 'p' for previous image, 'q' to quit.")
        
        while True:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            key = cv2.waitKey(1)
            if key == ord('e'):
                current_idx = (current_idx + 1) % total_samples
                current_idx = update_detection_analysis(detection_results, current_idx)
                print(f"Showing result {current_idx + 1}/{total_samples}")
            elif key == ord('w'):
                current_idx = (current_idx - 1) % total_samples
                current_idx = update_detection_analysis(detection_results, current_idx)
                print(f"Showing result {current_idx + 1}/{total_samples}")
            elif key == ord('q'):
                break
    else:
        # Original dataset exploration mode
        paths = {
            "raw": "data/File1/arucoRaw",
            "basic": "data/File2/arucoBasic",
            "challenging": "data/File3/arucoChallenging",
            "combinedbasic": "data/File4/combinedPicsBasic",
            "combinedchallenging": "data/File5/combinedPicsChallenging",
            "office": "data/File6/officePics",
            "custom1": "data/FileCustom1/arucoAugmented",
            "custom2": "data/FileCustom2/combinedAugmented",
        }

        dataset_path = paths[args.dataset]
        is_combined = args.dataset.startswith("combined")
        image_paths, bboxes = load_dataset(dataset_path)

        update_analysis(image_paths, bboxes, args.dataset, is_combined)
        print("Press 'e' for a new random image, 'w' to return, 'q' to quit.")

        while True:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            key = cv2.waitKey(1)
            if key == ord('e'):
                update_analysis(image_paths, bboxes, args.dataset, is_combined)
            elif key == ord('q'):
                break
