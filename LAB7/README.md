# DS605 Lab 7: Image Representation, PCA, Eigenfaces and Face Recognition

## 1. Project Overview

This project explores image representation and dimensionality reduction using Principal Component Analysis (PCA) on the ORL (Olivetti Research Laboratory), also known as the AT&T Face Database.

The project implements PCA using NumPy linear algebra operations, investigates eigenfaces, reconstructs facial images from reduced representations, and evaluates face recognition using nearest-neighbor matching and traditional machine learning classifiers.

The main objective is to reduce the dimensionality of face images while preserving important visual information and maintaining high recognition accuracy.

## 2. Objectives

- Load grayscale face images and represent them as numerical feature vectors.
- Create reproducible training and testing datasets while ensuring every subject appears in both sets.
- Implement PCA using eigendecomposition and Singular Value Decomposition (SVD).
- Compare the PCA implementations using eigenvalues and principal directions.
- Determine the number of principal components required to retain at least 90% of the variance.
- Visualize eigenfaces and reconstruct facial images using reduced representations.
- Evaluate dimensionality reduction through representation-size and reconstruction-error measurements.
- Compare raw-pixel and PCA-based nearest-neighbor face recognition.
- Evaluate KNN, Logistic Regression, and SVM classifiers using PCA features.
- Investigate how the number of principal components affects classification accuracy.

## 3. Dataset

**Dataset:** ORL / AT&T Face Database

The dataset contains grayscale facial images of 40 subjects, with 10 images per subject.

| Property | Value |
|---|---:|
| Number of subjects | 40 |
| Images per subject | 10 |
| Total images | 400 |
| Image dimensions | 112 × 92 pixels |
| Pixel values per image | 10,304 |
| Image representation | Flattened grayscale pixel vector |
| Original dataset size in samples | 400 |

Each image is converted into a one-dimensional vector containing 10,304 pixel values. These vectors form the feature matrix used for PCA and face recognition.

### Dataset access

The dataset was extracted locally and is not included in this repository. Obtain the ORL/AT&T Face Database from a legitimate source, such as the [AT&T Laboratories Cambridge ORL database page](https://cam-orl.co.uk/facedatabase.html), and place the extracted subject folders in a local dataset directory.

Expected folder structure:

```text
LAB7/
├── 202618011_Lab07.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── [dataset stored separately]
    ├── s1/
    ├── s2/
    ├── ...
    └── s40/
```

Update the dataset path in the notebook to match your local directory. Do not commit the dataset ZIP or individual face images to the public repository.

## 4. Train/Test Split

A reproducible split was created using NumPy's random number generator with seed 42. Each subject contributes six images to the training set and four images to the testing set.

| Dataset partition | Images | Features per image | Total feature values |
|---|---:|---:|---:|
| Training set | 240 | 10,304 | 2,472,960 |
| Testing set | 160 | 10,304 | 1,648,640 |
| Total | 400 | 10,304 | 4,121,600 |

All 40 subjects are represented in both partitions. The same split is used for the reported comparisons.

## 5. Methodology

### Part A: Image Representation and Dataset Preparation

1. Load the grayscale facial images.
2. Convert each image to a numerical array.
3. Flatten each 112 × 92 image into a vector of 10,304 pixel values.
4. Assign subject labels based on the subject folders.
5. Create a reproducible, subject-stratified training/testing split.

The resulting matrix dimensions are:

| Variable | Shape |
|---|---|
| Complete feature matrix `X` | `(400, 10304)` |
| Complete labels `y` | `(400,)` |
| Training features `X_train` | `(240, 10304)` |
| Testing features `X_test` | `(160, 10304)` |

### Part B: PCA and Eigenfaces

PCA reduces the dimensionality of facial images by finding directions that capture the greatest variance in the training data.

The training data is centered by subtracting the mean face from each image vector. PCA is then computed using two approaches:

- **Covariance-based eigendecomposition:** Uses the sample-space Gram matrix to avoid constructing a very large pixel-space covariance matrix.
- **Singular Value Decomposition:** Uses NumPy's SVD to obtain principal directions and singular values directly.

The two approaches were compared using their eigenvalues and corresponding principal directions.

#### PCA implementation comparison

| Measurement | Result |
|---|---:|
| Training samples | 240 |
| Original features per image | 10,304 |
| Nonzero principal directions retained by the covariance method | 239 |
| SVD principal directions returned | 240 |
| Average absolute cosine similarity of the first 20 corresponding directions | 1.0000 |

The centered training data has rank at most 239 because there are 240 training samples and their centered vectors sum to zero. Consequently, the final SVD direction is expected to have zero or near-zero variance.

The matching eigenvalues and principal directions confirm that the two implementations agree for the corresponding leading components, subject to numerical precision and sign ambiguity in eigenvectors.

#### Explained variance

The number of components was selected by examining cumulative explained variance.

| Measurement | Result |
|---|---:|
| Minimum components required to retain at least 90% variance | 85 |
| Cumulative variance retained by 85 components | 90.0868% |
| Variance threshold | 90% |

Therefore, **85 principal components** were used for the main PCA-based recognition and reconstruction experiments.

#### Eigenface visualization

The principal directions can be reshaped to 112 × 92 arrays and displayed as eigenfaces. These images represent major patterns of variation in the training faces, rather than individual identities.

The notebook includes visualizations of the mean face and leading eigenfaces.

#### Face reconstruction

Test images were projected into PCA space and reconstructed using the selected principal components.

For a centered image vector \(x\), reconstruction is performed using:

\[
\hat{x} = \bar{x} + (x-\bar{x})_k
\]

where \(\bar{x}\) is the mean face and \((x-\bar{x})_k\) is the reconstruction using the first \(k\) principal components.

For five sample test images, the observed reconstruction mean squared errors were:

| Test sample | Reconstruction MSE |
|---|---:|
| 1 | 388.38 |
| 2 | 529.27 |
| 3 | 412.15 |
| 4 | 342.30 |
| 5 | 462.87 |
| **Average** | **426.99** |

MSE is calculated from the original pixel values and their reconstructed values. Lower MSE indicates a closer pixel-level reconstruction.

#### Representation-size reduction

The original and PCA representations were compared using the number of stored numerical values, excluding the storage needed for the mean face and principal-component matrix.

| Measurement | Value |
|---|---:|
| Original representation | 2,472,960 values |
| Reduced representation | 20,400 values |
| PCA components | 85 |
| Compression ratio | 121.22:1 |
| Representation-size reduction | 99.18% |
| Variance retained | 90.09% |

The reduced training representation contains 240 × 85 = 20,400 values, compared with 240 × 10,304 = 2,472,960 original pixel values.

This comparison measures the reduction in image-representation values; it is not a measurement of total model file size or actual disk compression.

### Part C: Nearest-Neighbor Face Recognition

Nearest-neighbor recognition assigns each test image the identity of the closest training image. Euclidean distance is used to measure similarity.

Two representations were compared:

1. **Raw pixels:** Euclidean distance in the original 10,304-dimensional feature space.
2. **PCA features:** Euclidean distance in the 85-dimensional PCA space.

#### Recognition results

| Method | Correct predictions | Incorrect predictions | Accuracy |
|---|---:|---:|---:|
| Raw-pixel nearest neighbor | 155 / 160 | 5 | 96.88% |
| PCA nearest neighbor (85 components) | 154 / 160 | 6 | 96.25% |

Raw-pixel nearest neighbor achieved slightly higher accuracy, by approximately 0.63 percentage points. However, PCA reduced the dimensionality from 10,304 to 85 values per image.

The notebook also visualizes correct and incorrect PCA matches by displaying test images alongside their nearest training images.

### Part D: Traditional Machine Learning Classifier Comparison

Three classifiers were evaluated using the PCA-transformed training and testing images:

- K-Nearest Neighbors (KNN), with \(k=1\).
- Logistic Regression, with `max_iter=2000` and `random_state=42`.
- Support Vector Machine (SVM), using an RBF kernel, \(C=10\), and `gamma="scale"`.

#### Classifier comparison using 85 components

| Method | Accuracy | Correct predictions |
|---|---:|---:|
| PCA nearest neighbor | 96.25% | 154 / 160 |
| KNN (\(k=1\)) | 96.25% | 154 / 160 |
| Logistic Regression | 96.88% | 155 / 160 |
| SVM (RBF) | 97.50% | 156 / 160 |

KNN with one neighbor matches the PCA nearest-neighbor baseline because both select the closest training image in the same PCA feature space using Euclidean distance.

At 85 components, SVM achieved the highest accuracy among the three traditional classifiers.

#### Effect of the number of PCA components

The classifiers were also evaluated using 20, 40, 60, 85, and 120 principal components.

| PCA components | KNN accuracy | Logistic Regression accuracy | SVM (RBF) accuracy |
|---:|---:|---:|---:|
| 20 | 94.38% | 97.50% | 97.50% |
| 40 | 96.88% | 97.50% | **98.12%** |
| 60 | 96.88% | 97.50% | **98.12%** |
| 85 | 96.25% | 96.88% | 97.50% |
| 120 | 96.88% | **98.12%** | 97.50% |

The highest observed accuracy was **98.12%**, achieved by:

| Classifier | PCA components | Accuracy |
|---|---:|---:|
| SVM (RBF) | 40 | 98.12% |
| SVM (RBF) | 60 | 98.12% |
| Logistic Regression | 120 | 98.12% |

The results demonstrate that retaining more components does not necessarily improve recognition accuracy. The best configuration depends on the classifier and the feature representation.

## 6. Overall Results and Discussion

The experiments show that PCA can reduce the dimensionality of face images substantially while preserving useful identity information.

Key findings:

- The dataset contains 400 images representing 40 subjects.
- PCA retained 90.09% of the training-data variance using 85 components.
- The training image representation was reduced by 99.18% in terms of the number of numerical values.
- Raw-pixel nearest neighbor achieved 96.88% accuracy.
- PCA nearest neighbor achieved 96.25% accuracy using 85 components.
- SVM achieved 97.50% accuracy at 85 components.
- The highest observed accuracy was 98.12%, obtained with SVM at 40 or 60 components and Logistic Regression at 120 components.

The raw-pixel baseline slightly outperformed the PCA nearest-neighbor baseline at 85 components. Nevertheless, PCA provided a much smaller representation and enabled high accuracy with traditional classifiers.

These results should be interpreted in the context of the selected train/test split. Further evaluation with cross-validation or multiple reproducible splits would help assess the stability of the reported performance.

## 7. Tools and Technologies

- **Python:** Core programming language.
- **NumPy:** Image arrays, centering, covariance calculations, eigendecomposition, SVD, and PCA projections.
- **Pandas:** Organizing classifier results into comparison tables.
- **Matplotlib:** Eigenface visualization, reconstruction plots, and accuracy comparisons.
- **Pillow (PIL):** Loading grayscale face images.
- **Scikit-learn:** Traditional machine learning classifiers and accuracy evaluation.

PCA was implemented using NumPy rather than scikit-learn's PCA implementation.

## 8. Installation and Execution

### Prerequisites

Install Python 3.10 or later and create a virtual environment if desired.

Install the required packages:

```bash
pip install numpy pandas matplotlib Pillow scikit-learn jupyter
```

Alternatively, install from the repository requirements file:

```bash
pip install -r requirements.txt
```

### Run the notebook

1. Clone or download this repository.
2. Obtain and extract the ORL/AT&T Face Database separately.
3. Update the `dataset_path` variable in the notebook to point to the extracted dataset directory.
4. Open `202618011_Lab07.ipynb` in Jupyter Notebook or VS Code.
5. Execute the notebook cells in order.

Example dataset path:

```python
from pathlib import Path

dataset_path = Path(r"D:\DAUSem1\ML\Dataset\LAB7")
```

Replace this path with the location of the dataset on your own computer.

## 9. Repository Structure

```text
LAB7/
├── 202618011_Lab07.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── figures/
    ├── mean_face.png
    ├── eigenfaces.png
    ├── reconstruction.png
    └── accuracy_comparison.png
```

The `figures/` directory is optional if the notebook displays all plots inline and no figures have been exported separately.

## 10. Limitations and Future Improvements

- The evaluation uses one reproducible train/test split, so results may vary across different splits.
- The dataset is relatively small compared with modern face-recognition datasets.
- Pixel-based PCA may be sensitive to lighting, facial expression, and alignment changes.
- The reported compression ratio excludes the mean face and PCA basis storage.
- Component counts and classifier settings could be tuned more systematically using cross-validation.

Possible improvements include repeated stratified evaluation, additional classifier tuning, and testing the approach on images captured under more varied conditions.

## 11. Conclusion

This project demonstrates an end-to-end workflow for face-image representation, PCA-based dimensionality reduction, eigenface analysis, image reconstruction, and face recognition.

Using NumPy-based PCA, 85 principal components retained approximately 90.09% of the training-data variance while reducing the image-representation size by 99.18%. Traditional machine learning classifiers achieved high recognition accuracy, with a maximum observed test accuracy of 98.12%.

The results illustrate the trade-off between representation size, retained information, and classification performance, showing that PCA can provide an effective lower-dimensional representation for face-recognition experiments.

---

**Course:** DS605 — Fundamentals of Machine Learning  
**Lab 7 :** Image Representation, PCA, Eigenfaces and Face Recognition  
**Student ID:** 202618011
**Name:** Diya Tilwani