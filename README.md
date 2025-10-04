# Elevatelabs_Tasks
# Task 1
AI/ML Internship
Titanic Data Preprocessing 🚢
This repository contains a ipynb file for data preprocessing on the Titanic dataset, typically used in machine learning tasks such as survival prediction. The notebook demonstrates essential steps in cleaning, transforming, and preparing the dataset for modeling.
Overview
The Titanic dataset is a classic machine learning dataset containing passenger details such as age, sex, ticket class, and survival status.
The goal of preprocessing is to:
Handle missing values
Encode categorical features
Normalize/standardize numerical features
Create engineered features
Prepare the dataset for machine learning models
# Features
Data loading and exploration
Handling missing values (e.g., Age, Cabin, Embarked)
Label encoding & one-hot encoding
Feature engineering (e.g., Family size, Title extraction)
Scaling numerical variables
Final cleaned dataset ready for ML training

# TASK 2
Exploratory Data Analysis (EDA) on Titanic Dataset
The goal of this task is to perform Exploratory Data Analysis (EDA) on the Titanic dataset to understand patterns, trends, anomalies, and feature relationships using descriptive statistics and visualizations.
Exploratory Data Analysis (EDA) on Titanic Dataset
# Tools & Libraries
Python
Pandas → Data manipulation and summary statistics
Matplotlib → Data visualization
Seaborn → Advanced visualizations
NumPy → Numerical operations

# TASK 3
Linear Regression on Housing Price Prediction Dataset
# Objective
The aim of this task is to implement and understand simple and multiple linear regression using the Housing Price Prediction dataset. We evaluate the model performance and interpret coefficients to understand how features impact house prices.
# Dataset
Dataset used: Housing Price Prediction Dataset
Target Variable: price (House Price)
Features: Area, number of bedrooms, bathrooms, availability of features like basement, mainroad, airconditioning, parking, etc.
# Tools & Libraries
Scikit-learn → Linear Regression, Train-test split, Metrics
Pandas → Data manipulation
Matplotlib & Seaborn → Visualization
NumPy → Numerical operations
# Steps Performed
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
# Task 4
Logistic Regression Binary Classification 
Objective
Build a binary classifier using **Logistic Regression** and evaluate it with multiple metrics.

📂 Dataset
- **Breast Cancer Wisconsin Dataset** 
- Classes:
  - `0` → Malignant
  - `1` → Benign
# Steps
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

# Results
- Achieved **~95% accuracy** on the test set.
- ROC-AUC score: **~0.99**
- Model performed well in distinguishing between malignant and benign tumors.

# Visualizations
- Confusion Matrix heatmap
- ROC Curve
 Key Learnings
- Logistic Regression is widely used for **binary classification**.
- The **sigmoid function** maps outputs to probabilities between 0 and 1.
- Metrics like **precision, recall, and ROC-AUC** provide deeper insights beyond accuracy.
- Threshold tuning helps adjust the model for specific needs (e.g., high recall for medical diagnosis).
- # Task 5
- # Decision Trees & Random Forests 
 Objective
Learn **tree-based models** for classification using **Decision Trees** and **Random Forests**, and compare their performance.
📂 Dataset
- **Heart Disease Dataset** (used from Kaggle)
- Target: 
  - `0` → No Heart Disease
  - `1` → Heart Disease
# Steps
1. Load dataset.
2. Train **Decision Tree Classifier**.
   - Visualize the tree using `plot_tree`.
   - Analyze **overfitting** by controlling `max_depth`.
3. Train **Random Forest Classifier**.
   - Compare accuracy with Decision Tree.
   - Extract **feature importances**.
4. Evaluate both models
   - Accuracy
   - Confusion Matrix
   - Precision, Recall, F1-score
   - Cross-validation

#  Results
Result: Random Forest achieved near-perfect accuracy (~99.7% mean CV score).
Reason: The Heart Disease dataset has well-separated classes, and Random Forest’s bagging + multiple trees capture patterns effectively.
 While the model performs very well here, in real-world applications we rarely expect 100% — so cross-validation is important to confirm generalization.

#  Visualizations
- Decision Tree diagram
- Confusion Matrices
- Feature Importance bar plot

# Key Learnings
- Decision Trees split data using **entropy/information gain**.
- Random Forest uses **bagging** and multiple trees to improve accuracy.
- Feature importance shows which features influence predictions most.
- Cross-validation gives a more reliable performance estimate.

- # Task 6
- Dataset -Iris Dataset

Features:

Sepal Length

Sepal Width

Petal Length

Petal Width

Target: Species (Iris-setosa, Iris-versicolor, Iris-virginica)

# Tools & Libraries

Scikit-learn (KNeighborsClassifier, metrics, preprocessing)

Pandas & NumPy (data handling)

Matplotlib & Seaborn (visualization)

#  Steps Implemented

Load Dataset → Read Iris.csv.

Preprocessing → Dropped ID column, normalized features using StandardScaler.

Train-Test Split → 80% training, 20% testing.

Model Training → Trained KNN classifier with different values of K (3, 5, 7, 9, 11).

# Evaluation 

Accuracy score for each K.

Confusion Matrix.

Classification Report (Precision, Recall, F1-score).

Visualization →

Accuracy vs. K graph.

Confusion matrix heatmap.

Decision boundary plot (using Sepal Length & Sepal Width).

#  Results

Best K value: Varies depending on run (commonly K=5).

Accuracy: Achieved ~95–100% accuracy on test data.

Observations:

Normalization significantly improved results.

KNN performed well for multi-class classification.

Model is sensitive to noise and choice of K.

# Visualizations

Accuracy vs K plot

Confusion Matrix Heatmap

Decision Boundary (for 2D features)

# Task 7: Support Vector Machines (SVM)  

# Objective  
Implement and understand **Support Vector Machines (SVM)** for binary classification using both **Linear** and **RBF kernels**.  

---

#Dataset  
Source: Breast Cancer Dataset 
Features:30 numerical features (mean, standard error, worst value of cell nuclei).  
Target:** Binary classification  
  - `0 = malignant`  
  - `1 = benign`  


# Tools & Libraries   
Scikit-learn  
Pandas, NumPy  
Matplotlib, Seaborn 

# Steps Implemented  
1. Load Dataset** → Used sklearn’s built-in Breast Cancer dataset.  
2. Preprocessing** → Standardized features using `StandardScaler`.  
3. Model Training** →  
   - SVM with **Linear Kernel*
   - SVM with **RBF Kernel*
4. Hyperparameter Tuning** → Used `GridSearchCV` for `C` and `gamma`.  
5. Evaluation
   - Accuracy, Confusion Matrix, Classification Report  
   - Cross-validation  
6. Visualization** → Decision boundary (using 2 features).  

# Results  
Linear Kernel Accuracy:** ~96%  
RBF Kernel Accuracy:** ~98%  
Best Parameters (RBF):** e.g., `C=10, gamma=0.01`  
Cross-validation Accuracy (Linear):** ~95%  


# Visualizations  
 Confusion Matrix heatmap  
 Decision Boundary for 2 features  
 
# Task 8: K-Means Clustering  

# Objective  
Perform **unsupervised learning** using *K-Means clustering* to segment data into meaningful groups.  

# Dataset  
Source: Mall Customer Segmentation Dataset  
Features Used: 
Annual Income (k$)  
Spending Score (1–100)  
Target:** None (Unsupervised Learning).  

# Tools & Libraries 
Scikit-learn
Pandas, NumPy  
Matplotlib, Seaborn  
# Steps Implemented  
1. Load Dataset** → Read `Mall_Customers.csv`.  
2. Preprocessing** → Selected features, normalized using `StandardScaler`.  
3. Elbow Method** → Plotted inertia vs K to find optimal number of clusters.  
4. Model Training** → Applied K-Means clustering with optimal K (usually 5).  
5. Evaluation** → Computed **Silhouette Score** for cluster quality.  
6. Visualization** →  
   2D scatter plot with cluster assignments.  
   Centroids marked.  
   PCA-based visualization for high-dimensional extension. 

# Results  
*Optimal Clusters (K):** ~5 (from Elbow Method).  
  *Silhouette Score:** ~0.55 (indicates fairly good clustering).  
  Customers grouped into distinct clusters:  
  High income, high spenders  
  High income, low spenders  
  Average income, average spenders  
  Low income, high spenders  
  Low income, low spenders  

# Visualizations  
Elbow Method plot (to choose K).  
Cluster scatter plot (Annual Income vs Spending Score).  
PCA visualization for reduced dimensions. 



