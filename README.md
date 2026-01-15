# WEEK 3 – LINEAR REGRESSION FROM SCRATCH (ASSESSMENT)

## 📌 Dataset Information
Official Kaggle Dataset: Salary Dataset – Simple Linear Regression  
Feature (X): `YearsExperience`  
Target (y): `Salary`

This dataset is suitable for simple linear regression because it contains a single numerical input feature and a continuous output, making the relationship easy to study and visualize.

---

## 🧩 TASK 1 — Data Understanding

### **Q1: Why is this dataset suitable for Linear Regression?**
The Salary dataset shows a reasonably linear relationship between years of experience and salary. As work experience increases, salary also tends to increase in a proportional manner. The dataset is small, clean, and has only one independent feature and one dependent continuous target, which makes it ideal for simple linear regression without requiring complex models or non-linear transformations.

### **Q2: What assumption does Linear Regression make about the relationship between X and y?**
Linear Regression assumes that the relationship between X (years of experience) and y (salary) can be represented using a straight line:

\[
y = wX + b
\]

It also assumes:
- Linear dependency between inputs and output
- Errors are normally distributed around the regression line
- Constant variance (homoscedasticity)
- Independence between observations
- No major outliers influencing the trend

---

## 🧩 TASK 5 — Comparison with Sklearn

After implementing Gradient Descent from scratch, the model was compared with sklearn’s `LinearRegression`.  
The coefficient (w) and intercept (b) values from both were very close.

### **Why values are similar but not identical?**
- The scratch model uses iterative optimization through gradient descent.
- Sklearn uses a closed-form analytical solution known as the Normal Equation:

\[
w = (X^T X)^{-1} X^T y
\]

Gradient descent approximates the optimum through repeated updates, while sklearn computes it directly in one step. Small numerical differences arise due to:
- Learning rate
- Number of epochs
- Convergence tolerance
- Floating-point precision

### **What sklearn does differently internally?**
Sklearn optimizes using:
✔ Closed-form mathematical solution  
✔ Internal preprocessing & numeric precision improvements  
✔ Efficient matrix operations  

The scratch model provides intuition about learning, whereas sklearn provides computational efficiency.

---

## 🧠 TASK 6 — Thinking Questions

### **Q1: If we train for 10,000 epochs instead of 1,000, will the model always improve? (120+ words)**

Training for 10,000 epochs does not guarantee continuous improvement. Once gradient descent reaches a point of convergence, additional epochs yield diminishing returns. After convergence, the loss value becomes flat and the weights stop changing significantly. In some cases, further training may even cause oscillations or divergence if the learning rate is too high. Moreover, training excessively on small datasets can cause overfitting, where the model begins to memorize noise instead of learning the true underlying pattern. In real machine learning systems, early stopping criteria or validation monitoring is used to prevent wasted computation and reduce the risk of overfitting. Therefore, more epochs only help up to the point where the loss meaningfully decreases. Beyond that, the model does not necessarily improve and might even perform worse on unseen data.

### **Q2: Why is gradient descent preferred over directly trying all possible values of w and b? (120+ words)**

Brute-force searching for all possible values of w and b is computationally infeasible. Since both are real-valued parameters, there are infinitely many possibilities. Searching this space would require enormous computation, especially as dimensions increase. Gradient descent provides an efficient alternative by using derivatives to follow the slope of the loss function, moving directly toward the minimum. This makes optimization scalable even for large datasets and high-dimensional models. Modern machine learning models often contain thousands to millions of parameters (e.g., neural networks), making brute-force search completely unrealistic. Gradient descent requires only local gradient information instead of exhaustive evaluation. It converges quickly, uses less memory, and works with continuous values. For these reasons, gradient descent remains one of the most important and practical optimization methods in machine learning.

---

## 📁 Submission Files Included
