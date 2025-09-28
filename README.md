# BISINDO-GestureNet

## About
BISINDO-GestureNet is a deep learning project for recognizing **BISINDO alphabet gestures** (Indonesian Sign Language letters) from images. The project uses **Transfer Learning** with **MobileNetV2, EfficientNetB0, or VGG16** as backbone models, combined with **data augmentation** and preprocessing for robust performance on small datasets.

## Dataset
The dataset contains images of 26 letters (A-Z) of the BISINDO alphabet, available [here](https://www.kaggle.com/datasets/achmadnoer/alfabet-bisindo).

## Features
- Preprocessing:  
  - Resize images to 224x224  
  - Standardize pixels using `preprocess_input` (ImageNet mean/std)  

- Data augmentation (ImageDataGenerator):  
  - Rotation (±15°), width/height shift (±10%), zoom (±10%), shear (±10%)  
  - Brightness adjustment (0.8–1.2), horizontal flip (if safe for letter)  

- Models:  
  - MobileNetV2, EfficientNetB0, VGG16 (pretrained ImageNet)  
  - Custom head: GAP → Dense(256, ReLU) → Dropout(0.5) → Dense(26, softmax)  

- Evaluation:  
  - Train/Validation/Test split with stratified sampling  
  - Classification report (precision, recall, F1-score) per class  
  - Confusion matrix visualization  

## Libraries Used
- `numpy`, `pandas`, `glob`, `os` → data handling  
- `matplotlib`, `seaborn`, `cv2` → visualization & preprocessing  
- `scikit-learn` → train/test split, label encoding, evaluation metrics  
- `tensorflow` / `keras` → model building, training, callbacks, transfer learning  

## Requirements
- Python 3.8+
- TensorFlow 2.x
- OpenCV
- scikit-learn
- matplotlib, seaborn
- numpy, pandas
