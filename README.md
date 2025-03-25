# 🧠 Supervised Learning Project – Medical Diagnostics & Banking Marketing

## 📌 Project Overview

This project is divided into **two major parts**, each tackling real-world problems using **supervised machine learning algorithms**, data cleaning, feature engineering, and model performance evaluation.

---

## 🏥 Part A: Medical Diagnostics using Biomechanical Data

### 🧬 Domain: Medical / Biomechanics

### 📊 Datasets:
- `Part1 - Normal.csv`  
- `Part1 - Type_H.csv`  
- `Part1 - Type_S.csv`

Each file contains six biomechanical features and a masked medical **condition class** for patients. Class names are variants of `type_h`, `type_s`, and `normal`.

### 🧩 Objective:
Build a classification model that predicts the patient’s condition using biomechanical features.

### 🛠️ Steps Covered:
- Read and inspect all three datasets
- Standardize class labels and merge into a single dataset
- Perform detailed EDA:
  - Correlation heatmap
  - Pairplot by condition
  - Jointplot & boxplots
- Train a KNN model and evaluate performance
- Tune model parameters to improve accuracy, precision, recall

### 📌 Outcome:
Successfully classified medical conditions using biomechanical data with improved model performance through experimentation.

---

## 🏦 Part B: Banking Customer Loan Conversion Prediction

### 💼 Domain: Banking / Marketing

### 📊 Datasets:
- `Part2 - Data1.csv`  
- `Part2 - Data2.csv`

Each file contains financial and behavioral features of customers, used to predict whether the customer will opt for a **Loan on Credit Card** (`LoanOnCard`).

### 🧩 Objective:
Build a marketing-focused classification model to predict which customers are likely to convert, enabling targeted campaigns.

### 🛠️ Steps Covered:
- Merge datasets on customer `ID`
- Clean & convert binary features to categorical
- Visualize target class distribution & handle class imbalance
- Build a Logistic Regression model and evaluate performance
- Train additional models (SVM, KNN) and fine-tune hyperparameters
- Compare evaluation metrics and select the best model

### 📌 Outcome:
Enhanced prediction accuracy and recall for loan conversion using balanced datasets and optimized classifiers.

---

## ⚙️ Tools & Libraries Used

- Python (Jupyter Notebook)
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn

---

## 📁 Repository Structure

.
├── code/
│   └── Project_2_(Ishant_Kundra).ipynb           # Main notebook for both parts
│
├── dataset/
│   ├── Part1 - Normal.csv                        # Medical normal class data
│   ├── Part1 - Type_H.csv                        # Medical type_h class data
│   ├── Part1 - Type_S.csv                        # Medical type_s class data
│   ├── Part2 - Data1.csv                         # Banking dataset 1
│   └── Part2 - Data2.csv                         # Banking dataset 2
│
├── Problem Statement/
│   └── SL_Problem Statement.pdf                  # Detailed project brief
│
├── .gitignore
└── README.md                                     # This file

---

## 💡 Key Learnings

- Real-world data cleaning, merging, and preprocessing
- Classification model development for both medical and banking domains
- Visualization for class separation and feature correlation
- Model performance enhancement via hyperparameter tuning
- Handling class imbalance with resampling techniques

---

## ✍️ Author

**Ishant Kundra**  
📧 [ishantkundra9@gmail.com](mailto:ishantkundra9@gmail.com)  
🎓 Master’s in Computer Science | AIML Track
