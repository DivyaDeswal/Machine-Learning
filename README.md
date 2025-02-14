# 🤖 Machine Learning Algorithms
Welcome to my Machine Learning Algorithms repository! 🚀 This repo contains Jupyter notebooks implementing Supervised and Unsupervised Machine Learning algorithms with explanations and code examples.

## 📂 Algorithms Included
### 1️⃣ Linear Regression
📌 Type: Supervised Learning (Regression)
📌 Use Case: Predicting continuous values (e.g., house prices, sales forecasting).
📌 How it Works:

###### Finds the best-fit line that minimizes the difference between predicted and actual values.
###### Uses the Least Squares Method to minimize errors.
####### Formula:
𝑦
=
𝑚
𝑥
+
𝑏
y=mx+b
where:
𝑦
y = predicted value
𝑥
x = feature variable
𝑚
m = slope of the line
𝑏
b = intercept
🔧 Libraries Used: sklearn, pandas, numpy, matplotlib.

### 2️⃣ Logistic Regression
📌 Type: Supervised Learning (Classification)
📌 Use Case: Binary classification problems (e.g., spam detection, disease prediction).
📌 How it Works:

Unlike Linear Regression, it predicts probability values between 0 and 1.
Uses the Sigmoid Function to transform outputs:
𝑃
(
𝑌
=
1
)
=
1
1
+
𝑒
−
(
𝑏
0
+
𝑏
1
𝑋
)
P(Y=1)= 
1+e 
−(b 
0
​
 +b 
1
​
 X)
 
1
​
 
If 
𝑃
(
𝑌
)
>
0.5
P(Y)>0.5, classify as 1, else classify as 0.
🔧 Libraries Used: sklearn, numpy, matplotlib.
