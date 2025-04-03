# Mammogram Classification Project

## Overview
This project implements a machine learning solution for classifying mammogram images into normal (BI-RADS 1) and abnormal cases. The system uses traditional machine learning algorithms applied to extracted features from mammogram images to predict breast cancer risk, focusing on maximizing recall to minimize false negatives.

## Dataset
The project uses a dataset containing mammogram images with breast-level annotations. Each image is labeled according to the BI-RADS classification system:
- Class 0: Normal (BI-RADS 1)
- Class 1: Abnormal (Other BI-RADS categories)

## Project Structure
```
├── breast-level_annotations.csv   # Annotations for mammogram images
├── images/                        # Directory containing mammogram images
├── X_train.npy                    # Saved processed training features
├── y_train.npy                    # Saved training labels
├── X_test.npy                     # Saved processed test features
├── y_test.npy                     # Saved test labels
└── main.py                        # Main script containing data processing and model training
```

## Methodology

### Data Preprocessing
1. Loading and resizing mammogram images to 128x128 pixels
2. Converting images to grayscale
3. Normalizing pixel values
4. Feature extraction using PCA (Principal Component Analysis)
5. Handling class imbalance with SMOTE (Synthetic Minority Over-sampling Technique)
6. Splitting data into training, validation, and test sets

### Models Evaluated
The project evaluates several machine learning algorithms:
- Logistic Regression
- Decision Tree
- Random Forest
- k-Nearest Neighbors (k-NN)
- Support Vector Machine (SVM)

### Hyperparameter Tuning
Randomized search with cross-validation was used to find optimal hyperparameters for each model, optimizing for recall.

### Evaluation Metrics
Models were evaluated primarily using:
- Recall (sensitivity): to minimize false negatives
- Precision: to assess false positive rate
- F1-score: harmonic mean of precision and recall
- Confusion matrices: for visual assessment of model performance

## Results
Model performance based on recall scores:

| Model | Recall |
|-------|--------|
| Logistic Regression | [score] |
| Decision Tree | [score] |
| Random Forest | [score] |
| k-NN | [score] |
| SVM | [score] |

The best-performing model was determined to be [Best Model] with a recall of [score].

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn
- Pillow (PIL)

## Installation
```bash
git clone https://github.com/yourusername/mammogram-classification.git
cd mammogram-classification
pip install -r requirements.txt
```

## Usage
1. Place your mammogram images in the `images` directory
2. Update the paths in the script if necessary
3. Run the main script:
```bash
python main.py
```

## Future Improvements
- Implement deep learning models (CNNs) for feature extraction
- Explore ensemble methods combining multiple classifiers
- Incorporate additional clinical metadata to improve predictions
- Experiment with different dimensionality reduction techniques

## License
[Your License Here]

## Acknowledgements
This project utilizes mammography data for breast cancer screening classification.
