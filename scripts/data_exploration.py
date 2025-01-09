import argparse
import os, sys, csv, ast, random

import cv2
import numpy as np
import matplotlib.pyplot as plt

WINDOW_NAME = "Dataset Analysis"

def load_image_paths(dataset_path):
    image_paths = []
    for root, _, files in os.walk(dataset_path):
        for file in sorted(files):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(root, file))
    return image_paths

def load_dataset(dataset_path):
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
    width, height = size
    # Create a Matplotlib figure and axis with the specified size.
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    
    # Plot histogram depending on whether the image is grayscale or RGB.
    if image.ndim == 2:
        ax.hist(image.ravel(), bins=256, range=(0, 256),
                color='gray', alpha=0.7, label='Gray')
    else:
        for color, channel in zip(('b', 'g', 'r'), range(3)):
            ax.hist(image[:, :, channel].ravel(), bins=256, range=(0, 256),
                    color=color, alpha=0.5, label=color.upper())
    
    ax.set(xlabel='Pixel Intensity', ylabel='Frequency')
    plt.tight_layout()

    # Render the figure and convert to cv2
    fig.canvas.draw()
    rgba_buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = rgba_buffer.reshape((width, height, 4))
    plt.close(fig)
    
    hist_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.resize(hist_img, (width, height))

def create_text_panel(text_lines, panel_size=(600, 400)):
    width, height = panel_size
    text_panel = np.full((height, width, 3), 255, dtype=np.uint8)
    for i, line in enumerate(text_lines):
        text_start_position = (10, 50 + i * 25)
        cv2.putText(text_panel, line, text_start_position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return text_panel

def update_analysis(image_paths, bboxes, dataset_name, is_combined):
    random_index = random.randrange(len(image_paths))
    path = image_paths[random_index]
    sample_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    display_img = sample_img.copy()
    bbox = None

    if is_combined:
        fname = os.path.basename(path)
        bbox = bboxes.get(fname)

        # Draw bounding box
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        if sample_img.ndim == 2:
            display_img = cv2.cvtColor(sample_img, cv2.COLOR_GRAY2BGR)

    sample_mean = float(np.mean(sample_img))
    sample_std = float(np.std(sample_img))
    
    hist_img = create_histogram_image(sample_img)
    height, width = display_img.shape[:2]

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

    # Resize to same dims and stack into one image
    display_img_resized = cv2.resize(display_img, (400, target_height))
    hist_img_resized = cv2.resize(hist_img, (400, target_height))
    text_panel_resized = cv2.resize(text_panel, (600, target_height))

    composite = np.hstack([display_img_resized, hist_img_resized, text_panel_resized])

    cv2.imshow(WINDOW_NAME, composite)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Dataset analysis.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "raw", "basic", "challenging", "combinedbasic", "combinedchallenging", "office"
        ],
        help="Specify which dataset to use."
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    dataset_name = args.dataset
    paths = {
        "raw": "data/File1/arucoRaw",
        "basic": "data/File2/arucoBasic",
        "challenging": "data/File3/arucoChallenging",
        "combinedbasic": "data/File4/combinedPicsBasic",
        "combinedchallenging": "data/File5/combinedPicsChallenging",
        "office": "data/File6/officePics"
    }

    dataset_path = paths[dataset_name]
    is_combined = dataset_name.startswith("combined")
    image_paths, bboxes = load_dataset(dataset_path)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    update_analysis(image_paths, bboxes, dataset_name, is_combined)
    print("Press 'n' for a new random image, 'q' to quit.")

    while True:
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
        key = cv2.waitKey(1)
        if key == ord('n'):
            update_analysis(image_paths, bboxes, dataset_name, is_combined)
        elif key == ord('q'):
            break
