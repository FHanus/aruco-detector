# ArUco Detector

This project is part of the **Advanced Vision for Localisation and Mapping** module. It detects and classifies ArUco markers under various conditions, including distortions and noise.

## Folder Structure

The project directory is organised as follows:

```
aruco-detector/
├── aruco-venv/          # Virtual environment (ignored by Git)
├── data/                # Provided dataset
├── models/              # Directory to save trained models
├── results/             # Directory to store results
├── scripts/             # Python scripts
├── requirements.txt     # Dependencies 
├── README.md            # Project documentation
```

## Dependencies

```bash
pip install -r requirements.txt
```

## Scripts

### Data Exploration
The data explorer and visualiser (`data_exploration.py`) allows you to inspect the dataset images, view histograms, and see bounding box annotations where applicable.

### Model Training 
The classifier training script (`train_classifier.py`) trains a simple CNN model on the ArUco marker dataset (any of the two provided). During training it will:

- Train the model for the specified number of epochs
- Save the best performing model based on validation accuracy
- Generate training progress plots showing loss and accuracy curves
- Create confusion matrices and classification reports
- Save all results to the `results/` directory

The trained model weights will be saved to the `models/` directory.