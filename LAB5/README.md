# DS605 Lab 5 — Machine Learning with Scikit-learn and From Scratch

## Name : Diya Tilwani
## Student ID : 202618011
## Subject : DS605 (Machine Learning)


## Overview

This project is part of **DS605 Fundamentals of Machine Learning — Lab Assignment 5**.

The objective is to implement and compare machine learning models using two approaches:

1. **Scikit-learn implementations**
2. **Manual implementations using only NumPy and Pandas**

The project contains both a **regression task** and a **classification task** using the UCI Productivity Prediction of Garment Employees dataset.

The models are evaluated based on predictive performance as well as training and prediction time.

---

## Dataset

**Dataset:** Productivity Prediction of Garment Employees

**Source:** UCI Machine Learning Repository

The dataset contains information about garment production activities such as:

* Team
* Targeted productivity
* Standard minute value (`smv`)
* Work in progress (`wip`)
* Overtime
* Incentive
* Idle time
* Idle workers
* Number of style changes
* Number of workers
* Quarter
* Department
* Day
* Date
* Actual productivity

The dataset contains **1,197 records**.

### Missing Values

The main missing-value issue was found in the `wip` column:

* Missing values: **506**
* Percentage missing: approximately **42.3%**

For both implementations, missing numerical values were handled using the **median calculated from the training data**.

Categorical values were cleaned by removing leading and trailing whitespace before encoding.

---

# Tasks

## 1. Regression

The regression task predicts:

```text
actual_productivity
```

### Model

* Linear Regression

Two implementations were developed:

* Scikit-learn Linear Regression
* Manual Linear Regression using NumPy

The manual implementation uses the closed-form solution:

$$
\theta = (X^TX)^{-1}X^Ty
$$

A pseudoinverse was used in the implementation for numerical stability:

```python
theta = np.linalg.pinv(
    X.T @ X
) @ X.T @ y
```

### Regression Metrics

The following metrics were used:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

---

## 2. Classification

A binary classification target was created as:

```python
df["MeetsTarget"] = (
    df["actual_productivity"] >= df["targeted_productivity"]
).astype(int)
```

Therefore:

* `1` → Actual productivity meets or exceeds targeted productivity
* `0` → Actual productivity is below targeted productivity

`actual_productivity` was removed from the classification input features to avoid **target leakage**.

### Model

* Logistic Regression

Two implementations were developed:

* Scikit-learn Logistic Regression
* Manual Logistic Regression using NumPy

The manual implementation uses the sigmoid function:

$$
\sigma(z) = \frac{1}{1+e^{-z}}
$$

and optimizes the parameters using gradient descent.

---

# Data Preprocessing

The same train-test split was used for all models.

```text
Training samples: 957
Testing samples: 240
Train-test split: 80% / 20%
Random seed: 42
```

The following preprocessing steps were applied.

### Date Processing

The original `date` column was converted into:

* Year
* Month
* Day of month

The original date column was then removed.

### Numerical Features

Missing numerical values were filled using the **training-set median**.

Numerical features were standardized using:

$$
z = \frac{x-\mu}{\sigma}
$$

where the mean and standard deviation were calculated using only the training data.

### Categorical Features

The categorical columns were:

```text
quarter
department
day
```

They were cleaned using `.str.strip()` and converted into numerical features using one-hot encoding.

The manual implementation used:

```python
pd.get_dummies()
```

and then aligned the test columns with the training columns.

---

# Scikit-learn Implementation

For Scikit-learn, preprocessing was implemented using:

* `SimpleImputer`
* `StandardScaler`
* `OneHotEncoder`
* `ColumnTransformer`
* `Pipeline`

The models used were:

```python
LinearRegression()
```

and

```python
LogisticRegression(max_iter=1000)
```

Training and prediction times were measured using `time.perf_counter()`.

---

# Manual Implementation

The manual version uses only:

```text
NumPy
Pandas
```

No Scikit-learn models were used in the manual implementation.

### Manual Linear Regression

The coefficients were calculated using the closed-form solution with NumPy matrix operations.

### Manual Logistic Regression

The model was trained using gradient descent.

Baseline parameters:

```text
Learning rate = 0.01
Iterations = 5,000
```

An improvement experiment was then performed.

---

# Part C — Optimization

The manual Logistic Regression implementation was improved by experimenting with the gradient-descent parameters.

### Experiment 1

The number of iterations was increased:

```text
Learning rate = 0.01
Iterations = 10,000
```

Accuracy improved slightly from:

```text
75.00% → 75.42%
```

However, the increase in iterations also increased training time.

### Experiment 2

The learning rate was then increased:

```text
Learning rate = 0.05
Iterations = 10,000
```

This produced the final optimized manual Logistic Regression model.

The test accuracy improved to:

```text
76.67%
```

This was slightly higher than the Scikit-learn Logistic Regression accuracy of:

```text
76.25%
```

The optimized model was therefore used as the final manual classification model.

---

# Final Results

## Regression Comparison

| Model                          |      MAE |     RMSE |       R² | Training Time | Prediction Time |
| ------------------------------ | -------: | -------: | -------: | ------------: | --------------: |
| Scikit-learn Linear Regression | 0.105137 | 0.144063 | 0.326551 |    0.002947 s |      0.000465 s |
| Manual Linear Regression       | 0.105455 | 0.144229 | 0.325005 |    0.001706 s |      0.000062 s |

### Regression Observation

The manually implemented Linear Regression achieved performance very close to the Scikit-learn implementation.

The difference between the two models was small:

* MAE: 0.105455 vs 0.105137
* RMSE: 0.144229 vs 0.144063
* R²: 0.325005 vs 0.326551

This shows that the NumPy implementation successfully reproduced the main predictive behavior of Linear Regression.

---

## Classification Comparison

| Model                                    |     Accuracy |    Precision |       Recall |           F1 |  Training Time | Prediction Time |
| ---------------------------------------- | -----------: | -----------: | -----------: | -----------: | -------------: | --------------: |
| Scikit-learn Logistic Regression         |     0.762500 |     0.776699 |     0.935673 |     0.848806 |     0.010902 s |      0.000309 s |
| Manual Logistic Regression               |     0.750000 |     0.763033 |     0.941520 |     0.842932 |     0.089848 s |      0.000110 s |
| **Optimized Manual Logistic Regression** | **0.766667** | **0.780488** | **0.935673** | **0.851064** | **0.181816 s** |  **0.000396 s** |

### Classification Observation

The baseline manual Logistic Regression achieved an accuracy of **75.00%**.

After optimization, using a learning rate of **0.05** and **10,000 iterations**, the accuracy increased to **76.67%**.

The optimized manual implementation produced:

* Accuracy: **0.766667**
* Precision: **0.780488**
* Recall: **0.935673**
* F1 Score: **0.851064**

The optimized manual model achieved slightly higher accuracy and F1 score than the Scikit-learn model on the fixed test split.

However, the manual Logistic Regression required substantially more training time because the gradient-descent optimization was implemented explicitly rather than using Scikit-learn's optimized implementation.

---

# Overall Comparison

### Regression

The manual implementation produced nearly the same predictive performance as Scikit-learn.

This demonstrates that the mathematical formulation of Linear Regression can be reproduced effectively using NumPy matrix operations.

### Classification

The baseline manual Logistic Regression was slightly behind the Scikit-learn model.

After tuning the learning rate, the optimized manual model improved to **76.67% accuracy** and **0.851064 F1**, slightly exceeding the corresponding Scikit-learn results on the same test set.

### Execution Time

The experiments also demonstrate an important difference between implementation approaches.

Scikit-learn provides optimized implementations, while the manual Logistic Regression explicitly performs gradient-descent iterations. As a result, the manual classification model required more training time.

The prediction times for all models remained very small.

---

# Conclusion

This project demonstrates how machine learning algorithms can be implemented both with Scikit-learn and from scratch using NumPy and Pandas.

For **Linear Regression**, the manual closed-form implementation produced results very close to Scikit-learn, showing that the underlying mathematical operations can be reproduced successfully.

For **Logistic Regression**, the manual implementation used the sigmoid function and gradient descent. Increasing the number of iterations produced only a small improvement, while increasing the learning rate from **0.01 to 0.05** resulted in a more noticeable improvement in test performance.

The final optimized manual Logistic Regression achieved **76.67% accuracy** and **0.851064 F1 score**, compared with **76.25% accuracy** and **0.848806 F1 score** for Scikit-learn on the same fixed test set.

Overall, the experiment highlights the trade-off between:

* **Library implementations:** convenient, optimized, and faster to train
* **Manual implementations:** more transparent and useful for understanding the underlying mathematics and optimization process

The project also emphasizes the importance of consistent preprocessing, avoiding target leakage, using the same train-test split for fair comparison, vectorizing numerical operations, and tuning optimization parameters when implementing models manually.

---

# Technologies Used

```text
Python
Pandas
NumPy
Scikit-learn
Jupyter Notebook
Matplotlib / Seaborn (if used for EDA)
```

---

# Project Structure

A typical repository structure is:

```text
DS605-Lab5/
│
├── README.md
├── notebook.ipynb
└── garments_worker_productivity.csv
```

---

# How to Run

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd <repository-folder>
```

### 2. Install the required packages

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### 3. Open the notebook

```bash
jupyter notebook
```

Open the project notebook and run the cells from top to bottom.

---

# Key Learning Outcomes

Through this project, the following concepts were implemented and compared:

* Data preprocessing
* Missing-value imputation
* Categorical encoding
* Feature scaling
* Train-test splitting
* Linear Regression
* Logistic Regression
* Sigmoid function
* Gradient Descent
* Closed-form Linear Regression
* Classification metrics
* Regression metrics
* Execution-time comparison
* Vectorized NumPy operations
* Hyperparameter/optimization parameter experimentation
* Target leakage prevention
* Comparison of library and from-scratch implementations
