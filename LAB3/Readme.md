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
| Logistic Regression + Pipeline A | 0.7968 | 0.7935 | 0.7737 | 0.6351 | 0.6976 |
| Logistic Regression + Pipeline B | 0.7970 | 0.7934 | 0.7752 | 0.6327 | 0.6967 |
| Decision Tree + Pipeline A | 0.9956 | 0.8461 | 0.7909 | 0.8017 | 0.7962 |
| Decision Tree + Pipeline B | 0.9956 | 0.8468 | 0.7920 | 0.8021 | 0.7970 |

## Final Observations

1. The best overall combination was **Decision Tree + Pipeline B**, which achieved the highest testing accuracy of **0.8468** and the highest F1-score of **0.7970**.

2. For Logistic Regression, StandardScaler and MinMaxScaler produced almost identical results. Pipeline A had a slightly higher testing accuracy and F1-score, so StandardScaler performed marginally better.

3. Scaling made very little difference for the Decision Tree. Pipeline A achieved a testing accuracy of **0.8461**, while Pipeline B achieved **0.8468**, confirming that Decision Trees are generally less sensitive to feature scaling.

4. Both Decision Tree models showed signs of overfitting. Their training accuracy was **0.9956**, while their testing accuracy was around **0.846**, resulting in a train-test difference of approximately **0.149**.

5. Logistic Regression showed better generalization, with a train-test accuracy difference of only about **0.003**. However, the confusion matrices and performance metrics showed that the Decision Tree models were better at predicting both canceled and non-canceled bookings.