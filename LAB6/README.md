# DS605 Lab Assignment 6
## Feature Extraction and Machine Learning with Image and Text Data

**Course:** DS605 – Fundamentals of Machine Learning  
**Assignment:** Lab Assignment 6  
**Name:** Diya Tilwani   
**StudentId:** 202618011

---

## 1. Overview

This project focuses on feature extraction and traditional machine learning for both image and text classification.

The assignment is divided into three parts:

- **Part A:** Image classification using the Asphalt Crack Dataset
- **Part B:** Email spam classification using the Email Spam Classification Dataset
- **Part C:** Improving image classification through enhanced feature representation

Traditional machine learning techniques are used throughout the project. No CNNs, deep learning models, or pretrained embeddings are used.

---

# Part A — Image Classification

## 2. Dataset

The Asphalt Crack Dataset contains 400 images divided equally into two classes.

| Class | Number of Images |
|---|---:|
| Crack | 200 |
| Non-Crack | 200 |
| **Total** | **400** |

The original images have dimensions of **448 × 448 × 3**.

The images were resized to **128 × 128** before feature extraction.

### Class Labels

- `1` → Crack
- `0` → Non-Crack

---

## 3. Image Preprocessing

The following preprocessing steps were performed:

1. Images were loaded using OpenCV.
2. Images were resized to 128 × 128 pixels.
3. Images were converted to grayscale.
4. Canny edge detection was applied.
5. Numerical image features were extracted.

The preprocessing pipeline was:

```text
Original Image
      ↓
Resize to 128 × 128
      ↓
Grayscale Conversion
      ↓
Intensity Features + Canny Edge Features
      ↓
Numerical Feature Representation
```

---

## 4. Image Feature Extraction

Six numerical features were initially extracted from each image.

| Feature | Description |
|---|---|
| Mean Brightness | Average grayscale intensity |
| Contrast | Standard deviation of grayscale intensity |
| Dark Ratio | Proportion of pixels with intensity below 50 |
| Bright Ratio | Proportion of pixels with intensity above 200 |
| Edge Count | Number of detected Canny edge pixels |
| Edge Density | Proportion of pixels identified as edges |

The resulting feature representation contained **6 numerical features per image**.

---

## 5. Train-Test Split

The image dataset was divided using an 80:20 stratified split.

- Training images: **320**
- Testing images: **80**

`random_state = 42` was used for reproducibility.

---

## 6. Image Classification Models

The following models were evaluated:

- Logistic Regression
- Scaled Logistic Regression
- Random Forest Classifier

StandardScaler was applied before the second Logistic Regression experiment.

---

## 7. Part A Results

| Model | Accuracy | Precision | Recall | F1-score | Training Time (s) | Prediction Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 62.50% | 60.42% | 72.50% | 65.91% | 0.0155 | ~0 |
| Scaled Logistic Regression | 92.50% | 92.50% | 92.50% | 92.50% | 0.0015 | 0.0010 |
| Random Forest | 93.75% | 94.87% | 92.50% | 93.67% | 0.0962 | 0.0065 |

---

## 8. Part A Confusion Matrices

### Scaled Logistic Regression

| Actual / Predicted | Non-Crack | Crack |
|---|---:|---:|
| Non-Crack | 37 | 3 |
| Crack | 3 | 37 |

Correct predictions: **74 / 80**

### Random Forest

| Actual / Predicted | Non-Crack | Crack |
|---|---:|---:|
| Non-Crack | 38 | 2 |
| Crack | 3 | 37 |

Correct predictions: **75 / 80**

---

## 9. Part A Observation

The initial Logistic Regression model achieved 62.50% accuracy. After applying StandardScaler, its accuracy increased to 92.50%, showing that feature scaling had a significant effect on Logistic Regression.

Random Forest achieved 93.75% accuracy and a 93.67% F1-score using the extracted image features.

The results demonstrate that preprocessing and model selection can significantly affect image classification performance.

---

# Part B — Email Spam Classification

## 10. Dataset

The Email Spam Classification Dataset contains:

- **5,172 emails**
- **3,000 word-frequency features**

The target variable is `Prediction`.

| Class | Meaning | Count |
|---|---|---:|
| 0 | Non-Spam | 3,672 |
| 1 | Spam | 1,500 |
| **Total** | | **5,172** |

No missing values were found in the provided dataset.

---

## 11. Text Representation

The provided dataset does not contain the original raw email text.

Instead, it already contains **3,000 word-frequency features** for each email.

Therefore, the existing word-frequency features were used directly as a bag-of-words representation.

CountVectorizer was not applied again because CountVectorizer requires raw text input, while the provided dataset is already converted into numerical word-frequency features.

The feature matrix used for classification was:

**5,172 × 3,000**

---

## 12. Train-Test Split

The dataset was divided using an 80:20 stratified split.

- Training emails: **4,137**
- Testing emails: **1,035**

`random_state = 42` was used for reproducibility.

---

## 13. Text Classification Models

The following models were evaluated:

- Multinomial Naive Bayes
- Logistic Regression

Both models were trained using the same word-frequency representation and the same train-test split.

---

## 14. Part B Results

| Model | Accuracy | Precision | Recall | F1-score | Training Time (s) | Prediction Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Multinomial Naive Bayes | 94.20% | 86.81% | 94.33% | 90.42% | 0.2000 | 0.0520 |
| Logistic Regression | 98.26% | 95.78% | 98.33% | 97.04% | 3.1167 | 0.0506 |

---

## 15. Part B Confusion Matrices

### Multinomial Naive Bayes

| Actual / Predicted | Non-Spam | Spam |
|---|---:|---:|
| Non-Spam | 692 | 43 |
| Spam | 17 | 283 |

Correct predictions: **975 / 1,035**

Misclassifications: **60**

### Logistic Regression

| Actual / Predicted | Non-Spam | Spam |
|---|---:|---:|
| Non-Spam | 722 | 13 |
| Spam | 5 | 295 |

Correct predictions: **1,017 / 1,035**

Misclassifications: **18**

---

## 16. Part B Observation

Multinomial Naive Bayes achieved 94.20% accuracy and a 90.42% F1-score.

Logistic Regression achieved 98.26% accuracy and a 97.04% F1-score.

On the given word-frequency representation, Logistic Regression achieved higher accuracy, precision, recall, and F1-score. However, it required more training time than Multinomial Naive Bayes.

The prediction times of both models were very similar.

---

# Part C — Improving Image Representation

## 17. Feature Improvement

The original image representation contained six numerical features.

To capture additional information about the distribution of pixel intensities, **16 normalized grayscale histogram features** were added.

The additional features were:

```text
hist_0
hist_1
hist_2
...
hist_15
```

Therefore:

- Original representation = **6 numerical features**
- Additional histogram features = **16**
- Improved representation = **22 numerical features**

The complete dataframe contains:

- 1 image column
- 22 numerical feature columns
- 1 label column

Total = **24 columns**

---

## 18. Improved Random Forest

The same Random Forest configuration was used for a fair comparison:

- `n_estimators = 100`
- `random_state = 42`

The same train-test split was also used.

---

## 19. Part C Results

| Model | Feature Representation | Accuracy | Precision | Recall | F1-score | Training Time (s) | Prediction Time (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| Random Forest | Original 6 features | 93.75% | 94.87% | 92.50% | 93.67% | 0.0962 | 0.0065 |
| Random Forest | 6 + 16 histogram features | **96.25%** | **95.12%** | **97.50%** | **96.30%** | 0.0825 | 0.0079 |

---

## 20. Improved Random Forest Confusion Matrix

| Actual / Predicted | Non-Crack | Crack |
|---|---:|---:|
| Non-Crack | 38 | 2 |
| Crack | 1 | 39 |

Correct predictions: **77 / 80**

Misclassifications: **3 / 80**

---

## 21. Before vs After Feature Improvement

| Metric | Original Random Forest | Improved Random Forest | Change |
|---|---:|---:|---:|
| Accuracy | 93.75% | 96.25% | +2.50 percentage points |
| Precision | 94.87% | 95.12% | +0.25 percentage points |
| Recall | 92.50% | 97.50% | +5.00 percentage points |
| F1-score | 93.67% | 96.30% | +2.63 percentage points |
| Misclassifications | 5 | 3 | -2 |

---

## 22. Part C Observation

Adding 16 normalized grayscale histogram features improved the Random Forest performance.

Accuracy increased from **93.75% to 96.25%**, while F1-score increased from **93.67% to 96.30%**.

Recall also increased from **92.50% to 97.50%**.

The number of incorrect predictions decreased from five to three on the 80-image test set.

This indicates that the additional histogram features provided useful information about the distribution of grayscale intensities and improved the representation of the asphalt images.

---

# 23. Overall Results

## Image Classification

The improved Random Forest achieved:

| Metric | Result |
|---|---:|
| Accuracy | **96.25%** |
| Precision | **95.12%** |
| Recall | **97.50%** |
| F1-score | **96.30%** |

## Text Classification

Logistic Regression achieved:

| Metric | Result |
|---|---:|
| Accuracy | **98.26%** |
| Precision | **95.78%** |
| Recall | **98.33%** |
| F1-score | **97.04%** |

The image and text results come from different datasets and classification tasks, so these values are not intended as a direct comparison between the two tasks.

---

# 24. Key Findings

### Image Classification

- Grayscale intensity statistics and Canny edge features provided useful information for asphalt crack classification.
- Logistic Regression benefited significantly from feature scaling.
- Random Forest achieved strong performance using the extracted image features.
- Adding normalized grayscale histogram features improved Random Forest performance.
- The improved Random Forest achieved 96.25% accuracy.

### Text Classification

- The supplied email dataset was already represented using word-frequency features.
- Multinomial Naive Bayes provided strong performance with lower training time.
- Logistic Regression achieved higher classification metrics.
- Logistic Regression achieved 98.26% accuracy and 97.04% F1-score.

### Feature Representation

The experiments demonstrate that feature representation can have a significant impact on traditional machine learning performance.

For the image task, adding histogram information improved classification without requiring deep learning or CNNs.

---

# 25. Technologies Used

### Programming Language

- Python

### Libraries

- NumPy
- Pandas
- OpenCV
- Matplotlib
- Scikit-learn

### Image Processing

- Image resizing
- Grayscale conversion
- Pixel intensity statistics
- Canny edge detection
- Grayscale histogram extraction

### Machine Learning

- Logistic Regression
- Random Forest Classifier
- Multinomial Naive Bayes
- StandardScaler
- Train-test split

### Evaluation

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Training Time
- Prediction Time

---

# 26. Project Structure

```text
DS605-Lab6/
│
├── Lab6.ipynb
├── README.md
├── image_features.csv
└── requirements.txt
```

The original datasets are not included in the repository.

The notebook contains the code required to load and process the datasets after the dataset paths are provided.



# 27. Reproducibility

A fixed `random_state = 42` was used for reproducibility.

The datasets were split using stratification to preserve the class distribution in the training and testing sets.

The same train-test split and Random Forest configuration were used when comparing the original and improved image representations.

---

# 28. Limitations

- The image dataset contains only 400 images.
- The image features are hand-crafted rather than learned using deep learning.
- The provided email dataset does not contain the original email text, so the existing word-frequency representation was used directly.
- Model performance is based on a single stratified train-test split.
- Performance may vary on different datasets or unseen data.

---

# 29. Conclusion

This assignment demonstrated the application of traditional machine learning techniques to both image and text classification.

For the image classification task, asphalt crack images were resized, converted to grayscale, and processed using intensity statistics and Canny edge detection. These characteristics were converted into numerical features and used with traditional machine learning classifiers.

The initial Logistic Regression model achieved 62.50% accuracy. After applying StandardScaler, its accuracy increased to 92.50%, demonstrating the importance of feature scaling. Random Forest achieved 93.75% accuracy using the original six image features.

The image representation was then improved by adding 16 normalized grayscale histogram features. This increased the numerical feature representation from six to 22 features. The improved Random Forest achieved 96.25% accuracy and a 96.30% F1-score, compared with 93.75% accuracy and a 93.67% F1-score for the original representation. The improved model reduced the number of errors on the test set from five to three.

For the email spam classification task, the supplied dataset already contained 3,000 word-frequency features for each email. Multinomial Naive Bayes and Logistic Regression were evaluated using this existing bag-of-words representation. Multinomial Naive Bayes achieved 94.20% accuracy, while Logistic Regression achieved 98.26% accuracy and a 97.04% F1-score.

Overall, the experiments demonstrate that **preprocessing, feature representation, scaling, and model selection can significantly affect traditional machine learning performance**. The image experiments particularly demonstrate that improving the representation of the input data can improve classification performance without using CNNs or other deep learning techniques.

---
