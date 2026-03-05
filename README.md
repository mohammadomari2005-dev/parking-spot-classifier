# 🅿️ Parking Spot Classifier

A machine learning model that classifies parking spots as **empty** or **not empty** using Support Vector Machine (SVM).

## 📌 Description
This project uses computer vision and machine learning to detect whether a parking spot is empty or occupied. It uses SVM with GridSearchCV for hyperparameter tuning to find the best model.

## 🗂️ Project Structure
parking-spot-classifier/
├── data/               # Dataset (empty/not_empty images)
├── models/             # Saved trained model
├── notebooks/          # Jupyter notebook for exploration
├── src/
│   ├── config.py       # All settings and parameters
│   ├── prepare_data.py # Data loading and preprocessing
│   ├── train.py        # Model training and evaluation
│   └── predict.py      # Predict on new images
├── requirements.txt
└── README.md

## 🛠️ Installation
1. Clone the repository:
git clone https://github.com/YOUR_USERNAME/parking-spot-classifier.git
cd parking-spot-classifier

2. Create a virtual environment:
python -m venv venv
source venv/bin/activate

3. Install dependencies:
pip install -r requirements.txt

## 🚀 Usage

### Train the model:
cd src
python train.py

### Predict on a new image:
python predict.py --image_path ../data/dataset/empty/your_image.jpg

## 📊 Results
- Train Accuracy: 100%
- Test Accuracy: 100%

## 🧠 Model Details
- Algorithm: Support Vector Machine (SVC)
- Image size: 64x64
- Hyperparameter tuning: GridSearchCV
- Parameters tuned: C and gamma

## 🛠️ Tech Stack
- Python
- scikit-learn
- scikit-image
- NumPy
- Matplotlib
- Jupyter Notebook