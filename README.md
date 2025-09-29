# Elevatelabs_Tasks
AI/ML Internship
Titanic Data Preprocessing 🚢

This repository contains a ipynb file for data preprocessing on the Titanic dataset, typically used in machine learning tasks such as survival prediction. The notebook demonstrates essential steps in cleaning, transforming, and preparing the dataset for modeling.
Overview
The Titanic dataset is a classic machine learning dataset containing passenger details such as age, sex, ticket class, and survival status. The goal of preprocessing is to:

Handle missing values

Encode categorical features

Normalize/standardize numerical features

Create engineered features

Prepare the dataset for machine learning models

Features

Data loading and exploration

Handling missing values (e.g., Age, Cabin, Embarked)

Label encoding & one-hot encoding

Feature engineering (e.g., Family size, Title extraction)

Scaling numerical variables

Final cleaned dataset ready for ML training

TASK 2
Exploratory Data Analysis (EDA) on Titanic Dataset

The goal of this task is to perform Exploratory Data Analysis (EDA) on the Titanic dataset to understand patterns, trends, anomalies, and feature relationships using descriptive statistics and visualizations.
Exploratory Data Analysis (EDA) on Titanic Dataset

⚒️ Tools & Libraries

Python

Pandas → Data manipulation and summary statistics

Matplotlib → Data visualization

Seaborn → Advanced visualizations

NumPy → Numerical operations

TASK 3
Linear Regression on Housing Price Prediction Dataset
Objective

The aim of this task is to implement and understand simple and multiple linear regression using the Housing Price Prediction dataset. We evaluate the model performance and interpret coefficients to understand how features impact house prices.
Dataset

Dataset used: Housing Price Prediction Dataset
Download Here

Target Variable: price (House Price)
Features: Area, number of bedrooms, bathrooms, availability of features like basement, mainroad, airconditioning, parking, etc.

⚒️ Tools & Libraries

Python

Scikit-learn → Linear Regression, Train-test split, Metrics

Pandas → Data manipulation

Matplotlib & Seaborn → Visualization

NumPy → Numerical operations

🔍 Steps Performed

Data Preprocessing

Loaded dataset and checked structure.

Encoded categorical variables (e.g., mainroad, basement, airconditioning).

Handled missing values.

Splitting Dataset

Used 80% training and 20% testing.

Model Training

Implemented Linear Regression using sklearn.linear_model.LinearRegression.

Evaluation Metrics

MAE (Mean Absolute Error)

MSE (Mean Squared Error)

RMSE (Root Mean Squared Error)

R² Score

Visualization

Scatter plot: Actual vs Predicted Prices.

Residuals distribution.

Coefficient Analysis

Extracted regression coefficients to analyze feature importance.
Task 4
Logistic Regression Binary Classification 

Objective
Build a binary classifier using **Logistic Regression** and evaluate it with multiple metrics.

📂 Dataset
- **Breast Cancer Wisconsin Dataset** 
- Classes:
  - `0` → Malignant
  - `1` → Benign

⚙️ Steps
1. Load dataset.
2. Split into **train/test sets** (80/20).
3. Standardize features using `StandardScaler`.
4. Train a **Logistic Regression** model.
5. Evaluate using:
   - Accuracy
   - Confusion Matrix
   - Precision, Recall, F1-score
   - ROC-AUC Curve
6. Demonstrate **threshold tuning**.

📊 Results
- Achieved **~95% accuracy** on the test set.
- ROC-AUC score: **~0.99**
- Model performed well in distinguishing between malignant and benign tumors.

📈 Visualizations
- Confusion Matrix heatmap
- ROC Curve
 Key Learnings
- Logistic Regression is widely used for **binary classification**.
- The **sigmoid function** maps outputs to probabilities between 0 and 1.
- Metrics like **precision, recall, and ROC-AUC** provide deeper insights beyond accuracy.
- Threshold tuning helps adjust the model for specific needs (e.g., high recall for medical diagnosis).


