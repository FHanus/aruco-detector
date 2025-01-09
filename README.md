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

Data explorer and visualiser, **data_exploration.py**:

```bash
python3 scripts/data_exploration.py 
```