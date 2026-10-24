# DS605 Lab 8: Decision Trees and Support Vector Machines

## Overview

This lab explores Decision Tree and Support Vector Machine (SVM) classification using both from-scratch implementations and Scikit-learn. It covers impurity-based splitting, recursive tree construction, linear soft-margin SVM optimization, kernel-based classification, hyperparameter experiments, visualization, and model evaluation.

The experiments use the **Banknote Authentication dataset** for binary classification and a synthetic `make_moons` dataset to investigate nonlinear decision boundaries.

## Objectives

- Implement Gini impurity and entropy from scratch.
- Build a recursive Decision Tree classifier using NumPy and Python.
- Visualize decision trees, classification rules, and decision boundaries.
- Compare manual Decision Tree results with Scikit-learn.
- Implement a linear soft-margin SVM using hinge loss and subgradient descent.
- Compare the manual SVM with Scikit-learn's linear SVM.
- Investigate the effects of regularization, learning rate, iteration count, tree depth, and splitting criterion.
- Compare linear, polynomial, and RBF SVM kernels.
- Evaluate classification performance using accuracy, precision, recall, F1-score, training time, and prediction time.

## Dataset

### Banknote Authentication Dataset

The Banknote Authentication dataset contains numerical measurements extracted from images of genuine and forged banknotes.

| Property | Description |
|---|---|
| Dataset size | 1,372 observations |
| Input features | 4 |
| Target variable | `class` |
| Problem type | Binary classification |
| Missing values | None in the inspected dataset |
| Class 0 | 762 observations |
| Class 1 | 610 observations |

### Input Features

| Feature | Description |
|---|---|
| `variance` | Variance of the wavelet-transformed image |
| `skewness` | Skewness of the wavelet-transformed image |
| `curtosis` | Curtosis of the wavelet-transformed image |
| `entropy` | Entropy of the image |

The target variable `class` contains labels 0 and 1.

The dataset is reasonably balanced: approximately 55.54% of observations belong to class 0 and 44.46% belong to class 1.

### Exploratory Data Analysis

The initial inspection included:

- Displaying the first five observations.
- Checking dataset dimensions and column information.
- Examining missing values and class distribution.
- Computing descriptive statistics, including mean, standard deviation, quartiles, minimum, and maximum.
- Visualizing relationships between selected features and the class labels.

The features have different numerical scales. For example, `skewness` has a standard deviation of approximately 5.869, while `entropy` has a standard deviation of approximately 2.101. Therefore, standardization is used for SVM experiments.

## Methodology

### Part A: Decision Tree from Scratch

A Decision Tree classifier was implemented without relying on Scikit-learn's tree-building algorithm for the manual model.

The implementation includes:

1. Gini impurity and entropy calculations.
2. Searching candidate thresholds for each feature.
3. Computing impurity reduction to select the best split.
4. Recursive tree construction.
5. Stopping conditions based on purity, maximum depth, minimum samples, and the absence of a useful split.
6. Prediction for individual observations and batches of observations.
7. Printing tree rules and visualizing the resulting tree and decision boundaries.

The manual implementation was compared with Scikit-learn's `DecisionTreeClassifier` using both Gini impurity and entropy.

#### Decision Tree Hyperparameter Experiments

The experiments examined maximum depths of 1, 2, 3, 5, 10, and unlimited depth (`None`).

| Criterion | Maximum depth | Manual accuracy | Scikit-learn accuracy |
|---|---:|---:|---:|
| Entropy | 1 | 82.91% | 82.91% |
| Entropy | 2 | 87.64% | 87.64% |
| Entropy | 3 | 90.91% | 90.91% |
| Entropy | 5 | 93.45% | 93.45% |
| Entropy | 10 | 93.45% | 93.09% |
| Entropy | None | 93.45% | 93.45% |
| Gini | 1 | 82.91% | 82.91% |
| Gini | 2 | 87.64% | 87.64% |
| Gini | 3 | 90.91% | 90.91% |
| Gini | 5 | 93.09% | 93.09% |
| Gini | 10 | 92.00% | 92.36% |
| Gini | None | 92.00% | 92.00% |

**Observations**

- The highest observed test accuracy was **93.45%**, achieved by the manual entropy tree at depths 5, 10, and unlimited depth.
- The Scikit-learn entropy tree also achieved 93.45% at depths 5 and unlimited depth.
- Increasing tree depth improved performance initially, but further increases did not always improve accuracy.
- Manual and Scikit-learn results were generally close, supporting the behavior of the from-scratch implementation.
- Scikit-learn trained much faster than the manual implementation because its underlying implementation is optimized.

### Part B: Linear SVM from Scratch

A linear soft-margin SVM was implemented using NumPy, without using Scikit-learn's SVM optimizer for the manual model.

The implementation includes:

1. Standardizing features using training-set statistics.
2. Mapping class labels from `{0, 1}` to `{-1, +1}`.
3. Calculating hinge loss.
4. Computing subgradients for the weight vector and bias.
5. Updating parameters using gradient-based optimization.
6. Evaluating predictions and classification accuracy.
7. Visualizing the separating hyperplane, margin boundaries, and margin-violating training samples.

The objective function minimized by the manual Linear SVM is:

$$
J(w,b)=\frac{1}{2}\lVert w\rVert^2
+C\frac{1}{n}\sum_{i=1}^{n}
\max\left(0,1-y_i(w^\top x_i+b)\right)
$$

The objective consists of two terms:

1. **Regularization term:** $\frac{1}{2}\lVert w\rVert^2$ penalizes large weights and encourages a wider margin.
2. **Hinge loss term:** $C\frac{1}{n}\sum_{i=1}^{n}\max(0,1-y_i(w^\top x_i+b))$ penalizes observations that are misclassified or lie inside the margin.

Here, $w$ is the weight vector, $b$ is the bias, $C$ is the regularization parameter, $n$ is the number of training observations, $x_i$ is the feature vector, and $y_i\in\{-1,+1\}$ is the class label.



The manual implementation was compared with Scikit-learn's `SVC(kernel="linear")`.

#### Hyperparameter Experiments

The experiments investigated different values of \(C\), learning rates, and iteration counts.

**Observations**

- The regularization parameter \(C\) affects the balance between model complexity and hinge-loss penalties.
- Learning rate influences optimization stability and convergence.
- Increasing the number of iterations does not guarantee better test accuracy.
- The manual implementation is useful for understanding the optimization process, while Scikit-learn provides an optimized implementation.

The manual SVM's performance depends on the chosen hyperparameters and optimization procedure. Its results should be interpreted separately from those of the Scikit-learn implementation.

### Part C: Nonlinear SVM Kernels

A synthetic dataset was generated using Scikit-learn's `make_moons` function with 600 observations, noise level 0.20, and random state 42.

The data was divided into training and testing sets using an 80:20 split. Standardization was fitted on the training set and then applied to both sets.

Three SVM kernels were evaluated:

- Linear kernel
- Polynomial kernel of degree 3
- Radial Basis Function (RBF) kernel

#### Kernel Comparison Results

| Kernel | Accuracy | Precision | Recall | F1-score | Training time (s) | Prediction time (s) |
|---|---:|---:|---:|---:|---:|---:|
| RBF | 95.00% | 92.19% | 98.33% | 95.16% | 0.001602 | 0.000558 |
| Linear | 88.33% | 87.10% | 90.00% | 88.52% | 0.003826 | 0.000507 |
| Polynomial | 88.33% | 91.07% | 85.00% | 87.93% | 0.001321 | 0.000274 |

**Observations**

- The RBF kernel achieved the highest accuracy and F1-score.
- Its 98.33% recall indicates that it identified most positive-class observations.
- The linear kernel's straight decision boundary was less effective for the curved structure of the dataset.
- The polynomial kernel achieved higher precision than the linear kernel but lower recall.
- The polynomial kernel had the shortest measured training and prediction times in this run.

These results illustrate how nonlinear kernels can improve classification when a linear decision boundary is insufficient. Timing measurements may vary between runs.

### Part D: Final Model Evaluation

The final evaluation used all four Banknote Authentication features and the same train-test split established earlier in the notebook.

The following models were compared:

- Manual Decision Tree
- Scikit-learn Decision Tree
- Manual Linear SVM
- Scikit-learn Linear SVM
- Scikit-learn RBF SVM

The SVM models used standardized features. The scaler was fitted on the training data only to avoid test-data leakage.

#### Final Evaluation Results

| Model | Accuracy | Precision | Recall | F1-score | Training time (s) | Prediction time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Scikit-learn RBF SVM | 100.00% | 100.00% | 100.00% | 100.00% | 0.00362 | 0.00102 |
| Manual Decision Tree | 99.27% | 98.39% | 100.00% | 99.19% | 0.63822 | 0.00022 |
| Scikit-learn Decision Tree | 99.27% | 98.39% | 100.00% | 99.19% | 0.00188 | 0.00056 |
| Scikit-learn Linear SVM | 97.45% | 94.57% | 100.00% | 97.21% | 0.00488 | 0.00088 |
| Manual Linear SVM | 88.36% | 94.12% | 78.69% | 85.71% | 0.15154 | 0.00002 |

*Note: Training and prediction times are in seconds and are specific to the recorded run and execution environment.*

**Observations**

- The Scikit-learn RBF SVM achieved the highest observed test accuracy of 100%.
- Both Decision Tree implementations achieved 99.27% accuracy, with very similar precision, recall, and F1-score.
- The Scikit-learn linear SVM achieved 97.45% accuracy, outperforming the manual linear SVM in this experiment.
- The manual linear SVM achieved 88.36% accuracy and 78.69% recall, indicating that it missed more positive-class observations than the other models.
- Scikit-learn's Decision Tree trained considerably faster than the manual implementation.
- The manual Decision Tree and Scikit-learn Decision Tree produced closely matching predictions, while their execution times differed substantially.

The perfect RBF SVM test score is an observed result for this particular split, not proof of perfect performance on unseen data in general. Further validation with cross-validation or an independent dataset would be needed to assess generalization more robustly.

## Summary of Findings

| Experiment | Main finding |
|---|---|
| Manual Decision Tree | Successfully implements impurity-based splitting and recursive classification |
| Decision Tree comparison | Manual and Scikit-learn accuracies were generally similar |
| Manual Linear SVM | Demonstrates hinge-loss optimization and the effect of hyperparameters |
| Kernel SVM comparison | RBF performed best on the nonlinear `make_moons` dataset |
| Final evaluation | Scikit-learn RBF SVM achieved the highest observed Banknote test accuracy |

The experiments demonstrate the importance of model selection, hyperparameter tuning, feature scaling, implementation efficiency, and appropriate evaluation metrics.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## How to Run

### 1. Clone the repository

```bash
git clone <YOUR_PUBLIC_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

Replace the placeholders with the actual public repository URL and folder name.

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 4. Run the notebook

Open the Lab 8 notebook and execute the cells sequentially, from dataset loading and EDA through the final model evaluation.

The notebook retrieves the Banknote Authentication dataset from its configured public source, so an internet connection may be required when the dataset has not been downloaded locally.

## Conclusion

This lab provides a practical comparison of manual and library-based implementations of Decision Trees and SVMs. Implementing the algorithms from scratch develops an understanding of impurity reduction, recursive splitting, hinge loss, and subgradient optimization. The experiments also show that nonlinear kernels can be effective for complex boundaries, and that optimized library implementations can substantially reduce training time.

The final results highlight the strong classification performance of the RBF SVM on the selected Banknote Authentication test split, while the manual Decision Tree demonstrates that a from-scratch classifier can achieve accuracy comparable to a standard library implementation.

## Author

**Student:** Diya Tilwani  
**Course:** DS605 — Machine Learning  
**Assignment:** Lab 8 — Decision Trees and Support Vector Machines
