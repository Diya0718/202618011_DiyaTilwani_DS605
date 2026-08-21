# Hotel Booking Demand - Classification Pipeline Comparison

## Student Information

**Name:** Diya Tilwani  
**Student ID:** 202618011  
**Course:** MSc Data Science Sem 1  

## Assignment Objective

The objective of this assignment is to build and compare Scikit-learn preprocessing pipelines and evaluate classification models for predicting whether a hotel booking will be canceled.

## Dataset

Dataset: Kaggle Hotel Booking Demand Dataset

Target Variable:

`is_canceled`

- `0` = Booking was not canceled
- `1` = Booking was canceled

## Dataset Link

[Hotel Booking Demand Dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

## Preprocessing

The following preprocessing steps were performed:

- Checked missing values and their percentages.
- Removed columns with very high missingness where justified.
- Removed data leakage columns:
  - `reservation_status`
  - `reservation_status_date`
- Checked selected numerical features for extreme outliers using boxplots and the IQR method.
- Removed only clear/extreme outliers.
- Split the dataset using an 80-20 train-test split with stratification.

### Pipeline A

Numerical features:

- KNNImputer with `n_neighbors=5`
- StandardScaler

Categorical features:

- SimpleImputer using `most_frequent`
- OneHotEncoder using `handle_unknown="ignore"`

### Pipeline B

Numerical features:

- KNNImputer with `n_neighbors=5`
- MinMaxScaler

Categorical features:

- SimpleImputer using `most_frequent`
- OneHotEncoder using `handle_unknown="ignore"`

## Models

The following four experiments were performed:

1. Logistic Regression + Pipeline A
2. Logistic Regression + Pipeline B
3. Decision Tree + Pipeline A
4. Decision Tree + Pipeline B

## Evaluation Metrics

The models were evaluated using:

- Training Accuracy
- Testing Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Final Results

| Model + Pipeline | Training Accuracy | Testing Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|---:|
| Logistic Regression + Pipeline A | | | | | |
| Logistic Regression + Pipeline B | | | | | |
| Decision Tree + Pipeline A | | | | | |
| Decision Tree + Pipeline B | | | | | |

## Final Observations

1. [Write your best-performing model based on your results.]
2. [Compare the effect of StandardScaler and MinMaxScaler on Logistic Regression.]
3. [Comment on the effect of scaling on Decision Tree.]
4. [Comment on possible overfitting using training and testing accuracy.]
5. [Summarize insights from the confusion matrices.]
