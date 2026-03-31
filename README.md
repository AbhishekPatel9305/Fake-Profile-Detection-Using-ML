# Fake Social Media Profile Detection using Machine Learning

## 📌 Overview

This project detects fake social media profiles using a Machine Learning approach. A Random Forest Classifier is trained on profile-based features to classify users as fake or genuine.

## 🎯 Objective

To automatically identify fake profiles and reduce risks like fraud, misinformation, and spam on social platforms.

## 📊 Dataset

* users.csv → Genuine profiles
* fusers.csv → Fake profiles

## ⚙️ Features Used

* statuses_count
* followers_count
* friends_count
* favourites_count
* listed_count
* lang_code

## 🤖 Model

* Algorithm: Random Forest Classifier
* Trees: 40
* Train-Test Split: 80:20
* Cross-validation: 5-fold

## 📈 Results

* Accuracy: ~94%
* ROC-AUC: 0.94
* High recall for fake profiles (≈99%)

## 📉 Evaluation Metrics

* Confusion Matrix
* Classification Report
* ROC Curve

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

## ▶️ How to Run

```bash
pip install -r requirements.txt
python src/main.py
```

## 📂 Project Structure

* `data/` → dataset
* `notebook/` → analysis & visualization
* `src/` → clean production code
* `model/` → trained model
* `results/` → graphs

## 🚧 Challenges

* Gender feature removed due to library incompatibility

## 🔮 Future Scope

* NLP for bio analysis
* Profile image analysis
* Network-based features
* Account age & location

## 👨‍💻 Author

Abhishek Kumar
Lovely Professional University
India
