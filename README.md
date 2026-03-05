# Parking Spot Classifier

A computer vision model that classifies parking spots as **empty** or **not empty** using SVM with GridSearchCV hyperparameter tuning.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0-orange) ![License](https://img.shields.io/badge/license-MIT-green)

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/parking-spot-classifier.git
cd parking-spot-classifier
pip install -r requirements.txt
```

## Usage

**Train the model:**
```bash
cd src
python train.py
```

**Predict on a new image:**
```bash
python predict.py --image_path ../data/dataset/empty/your_image.jpg
```

## Results

| Metric | Score |
|--------|-------|
| Train Accuracy | 86.39% |
| Test Accuracy | 89.24% |

## Project Structure

```
parking-spot-classifier/
├── data/               # Dataset (empty/not_empty)
├── models/             # Saved model
├── notebooks/          # Jupyter exploration
├── src/
│   ├── config.py       # Settings and parameters
│   ├── prepare_data.py # Data loading and preprocessing
│   ├── train.py        # Training and evaluation
│   └── predict.py      # Inference on new images
└── requirements.txt
```

## Tech Stack
Python · scikit-learn · scikit-image · NumPy · Matplotlib · Jupyter
