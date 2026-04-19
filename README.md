# Fake Social Media Profile Detection Using Machine Learning

A machine learning project that classifies social media accounts as fake or genuine using profile-level metadata and a Random Forest model. The repository is designed as a practical applied-ML project focused on fraud detection and account authenticity analysis.

## Overview

The system uses structured account features to identify suspicious profiles that may contribute to spam, fraud, impersonation, or misinformation on social platforms.

## Objective

To automatically distinguish fake profiles from genuine user accounts through supervised machine learning.

## Dataset

The training data lives in the nested project directory:

- `Fake-Profile-Detection-Using-ML/Data/users.csv` - genuine profiles
- `Fake-Profile-Detection-Using-ML/Data/fusers.csv` - fake profiles

## Features Used

- `statuses_count`
- `followers_count`
- `friends_count`
- `favourites_count`
- `listed_count`
- `lang_code`

## Model

- Algorithm: Random Forest Classifier
- Number of trees: 40
- Train-test split: 80:20
- Cross-validation: 5-fold

## Results

- Accuracy: approximately 94%
- ROC-AUC: 0.94
- strong recall for fake-profile identification

## Evaluation Outputs

The script generates:

- console metrics and classification report
- `results/learning_curve.png`
- `results/confusion_matrix.png`
- `results/roc_curve.png`

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## How To Run

```bash
pip install -r Fake-Profile-Detection-Using-ML/requirements.txt
python Fake-Profile-Detection-Using-ML/src/main.py
```

## Project Structure

- `Fake-Profile-Detection-Using-ML/Data/` - input datasets
- `Fake-Profile-Detection-Using-ML/Notebook/` - experimentation assets
- `Fake-Profile-Detection-Using-ML/src/` - cleaned source code
- `Fake-Profile-Detection-Using-ML/results/` - generated evaluation plots

## Future Improvements

- add NLP-based profile bio analysis
- integrate profile image-based features
- include network and graph-based behavior signals
- extend with account-age and temporal activity features
