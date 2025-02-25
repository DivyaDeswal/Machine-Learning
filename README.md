# 🤖 Machine Learning Algorithms
Welcome to my Machine Learning Algorithms repository! 🚀 This repo contains Jupyter notebooks implementing Supervised and Unsupervised Machine Learning algorithms with explanations and code examples.

## 📂 Algorithms Included
### 1️⃣ Linear Regression
📌 Type: Supervised Learning (Regression)

📌 Use Case: Predicting continuous values (e.g., house prices, sales forecasting).

🔧 Libraries Used: sklearn, pandas, numpy, matplotlib.

### 2️⃣ Logistic Regression
📌 Type: Supervised Learning (Classification)

📌 Use Case: Binary classification problems (e.g., spam detection, disease prediction).

🔧 Libraries Used: sklearn, numpy, matplotlib.

### 3️⃣ Decision Tree Algorithm
📌 Type: Supervised Learning (Classification & Regression)

📌 Use Case: Customer segmentation, credit risk assessment.

📌 How it Works:

The model splits data into branches based on feature conditions.
Uses Gini Index or Entropy to determine splits.

Forms a tree structure where leaves represent outcomes.

🔧 Libraries Used: sklearn.tree, pandas, matplotlib.

### 4️⃣ Random Forest Algorithm (Iris Dataset)
📌 Type: Supervised Learning (Classification)

📌 Use Case: Complex classification problems (e.g., medical diagnosis, image recognition).

📌 How it Works:

An ensemble method combining multiple decision trees.
Each tree votes, and the most common prediction is selected (majority voting).
Reduces overfitting compared to a single decision tree.

🔧 Libraries Used: sklearn.ensemble.RandomForestClassifier, pandas, matplotlib.

### 5️⃣ K-Means & Hierarchical Clustering
📌 Type: Unsupervised Learning (Clustering)

📌 Use Case: Customer segmentation, anomaly detection.

📌 How it Works:

K-Means Clustering
Defines K clusters (centroids).
Assigns each data point to the nearest cluster center.
Updates centroids until convergence.

#### Hierarchical Clustering
Creates a hierarchy of clusters using Agglomerative (bottom-up) or Divisive (top-down) methods.
Uses Dendrograms to visualize cluster merging.

🔧 Libraries Used: sklearn.cluster, scipy.cluster.hierarchy, matplotlib, seaborn.

#### 🛠️ Tools & Libraries Used
Python 🐍

Jupyter Notebook

Scikit-Learn (sklearn)

Pandas, NumPy, Matplotlib, Seaborn

#### 📌 How to Use
Clone the repository:
git clone https://github.com/DivyaDeswal/Machine-Learning.git

Open the notebooks in Jupyter or Google Colab.
Run the code and experiment with different models

## 📂 Classifiers
### Boosting Ensemble Techniques

Boosting is a powerful ensemble technique that combines multiple weak classifiers to form a strong classifier. The following methods are implemented in this repository:

### 1️⃣ Adaboost (Adaptive Boosting): 
A technique that adjusts weights iteratively to emphasize misclassified instances.

### 2️⃣ XGBoost (Extreme Gradient Boosting): 
An optimized gradient boosting library designed for speed and performance.

### 3️⃣ LightGBM (Light Gradient Boosting Machine): 
A gradient boosting framework that uses tree-based learning algorithms for faster training on large datasets.

## Dimensionality Reduction Techniques

### 📌 Objective

Dimensionality reduction helps reduce the number of input variables in a dataset while retaining as much meaningful information as possible. This technique is particularly useful for handling high-dimensional data.

### 🛠️ Techniques Used

## Principal Component Analysis (PCA):

Transforms high-dimensional data into a lower-dimensional space while retaining the most significant features.

Applied PCA to the liver patient dataset to aid in feature selection and model efficiency.

### 🔍 Implementation

## 1️⃣ Customer Segmentation for Retail

Objective: Segment customers based on purchasing behavior.

Techniques Used: Clustering algorithms such as K-Means and Hierarchical Clustering.

Use Case: Helps businesses identify customer groups and tailor marketing strategies accordingly.

Implementation:

Dataset consists of customer purchase history.

Clustering techniques applied to identify different customer segments.

Enhances customer engagement and improves business strategies.

## 2️⃣ Zomato Restaurant Price Prediction (Accuracy: 0.92)

Objective: Predict restaurant prices based on various factors such as location, cuisine, ratings, and reviews.

Techniques Used: Regression models including Linear Regression, Decision Trees, and Random Forest.

Use Case: Assists customers in understanding pricing trends and helps restaurant owners optimize pricing strategies.

Implementation:

Dataset includes restaurant details like type of cuisine, location, ratings, and price range.

Regression models trained to predict restaurant price range with high accuracy (0.92).

Enables better decision-making for both consumers and business owners.

