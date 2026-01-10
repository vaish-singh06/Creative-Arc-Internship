# WEEK 3 – LINEAR REGRESSION FROM SCRATCH (ASSESSMENT)

##  Dataset Information
Official Kaggle Dataset: Salary Dataset – Simple Linear Regression  
Feature (X): `YearsExperience`  
Target (y): `Salary`

---

##  TASK 1 — Data Understanding

### **Q1: Why is this dataset suitable for Linear Regression?**
The Salary dataset shows a reasonably linear relationship between years of experience and salary. As work experience increases, salary also tends to increase in a proportional manner. The dataset is small, clean, and has only one independent feature and one dependent continuous target, making it ideal for simple linear regression.

### **Q2: What assumption does Linear Regression make about the relationship between X and y?**
Linear Regression assumes that the relationship between X (years of experience) and y (salary) can be represented using a straight line:

\[
y = wX + b
\]

Additional assumptions:
- Linear dependency between inputs and output
- Errors are normally distributed
- Constant variance (homoscedasticity)
- Independence between observations
- No major outliers influencing the trend

---

##  TASK 5 — Comparison with Sklearn

The gradient descent model from scratch and sklearn’s LinearRegression gave very close values for the coefficient and intercept.

### **Why similar but not identical?**
The scratch version uses iterative optimization (gradient descent), while sklearn uses a closed-form solution (Normal Equation). Numerical differences come from:
- Learning rate
- Epoch count
- Floating-point precision
- Convergence tolerance

### **What sklearn does internally?**
Sklearn uses efficient matrix operations, optimized numerical libraries, and analytical solutions for linear regression.

---

##  TASK 6 — Thinking Questions

### **Q1: About 10,000 Epochs (120+ words)**
Training for 10,000 epochs does not always improve a model. After convergence, improvements become marginal and the loss may flatten out. Excessive training can lead to overfitting where the model memorizes noise. It also wastes computation and may introduce oscillations if the learning rate is high. Early stopping techniques or validation sets are generally used to stop training once the model stops improving.

### **Q2: Gradient Descent vs Brute Force Search (120+ words)**
Brute-force search for all possible values of w and b is not feasible because w and b exist in continuous space. Gradient descent uses calculus to follow the slope of the loss function and reduce error efficiently. It scales well for larger datasets and high-dimensional models, while brute-force becomes exponentially more difficult. Gradient descent reduces computation time, memory, and complexity, making it practical for modern machine learning.

---

## 📁 Files Included
