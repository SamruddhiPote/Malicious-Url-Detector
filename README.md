# Malicious URL Detector

A machine learning-based system to detect malicious/phishing URLs using Random Forest classifier and URL feature extraction.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## Features

- Extracts 20+ URL features including length, special characters, entropy, etc.
- Uses TF-IDF for path analysis
- Random Forest classifier with 95%+ accuracy
- CLI interface for real-time URL checking
- Model persistence with joblib

## Installation

1. Clone the repository:

-git clone https://github.com/SamruddhiPote/Malicious-Url-Detector.git
-cd malicious-url-detector

2. Install dependencies:

- pip install -r requirements.txt

3. Command Line Interface
   
- python src/detector.py

## 📊 Dataset

The model was trained on the [Malicious URLs Dataset](https://data.mendeley.com/datasets/vfszb9b36/1) from Mendeley Data

Key features extracted:
- URL structure analysis
- Domain characteristics
- Lexical features
- Content-based features
- Binary classification labels
