# Lab 02 – NumPy and Pandas Data Analysis

## Student Details

- **Assignment:** Lab 02 – NumPy and Pandas
- **Name:** Diya Tilwani
- **Student ID:** 202618011

## Dataset

The main dataset used for the Pandas analysis is the **Titanic Dataset**.

The dataset contains information about passengers aboard the Titanic, including passenger class, age, sex, family-related information, fare, and survival status.

The analysis uses NumPy and Pandas for numerical operations, data cleaning, transformation, statistical analysis, feature engineering, and visualization.

## Project Details

This assignment demonstrates the use of **NumPy and Pandas** for practical data analysis.

The work includes:

- NumPy array creation and numerical operations
- Data loading and inspection using Pandas
- Identification and analysis of missing values
- Handling missing values using statistical imputation
- Detection of outliers using the IQR method
- Data transformation and feature engineering
- Creation of 'FamilySize'
- Creation of 'IsAlone'
- Grouping and aggregation
- Survival-rate analysis
- Correlation analysis
- Data visualization
- Interpretation of results and observations

## Files in the Repository

```text
Lab02/
│
├── README.md
├── 202618011_Lab02.ipynb
├── data/
│   └── titanic.csv
├── cleaned_data/
│   └── titanic_cleaned.csv
└── figures/
    ├── ...
```

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook

## How to Run

1. Clone or download this repository.
2. Open '202618011_Lab02.ipynb' using Jupyter Notebook or JupyterLab.
3. Make sure the Titanic dataset is available in the 'data/' folder.
4. Run the notebook cells from top to bottom.

The notebook contains the complete code used for data loading, preprocessing, analysis, feature engineering, and visualization.

## Data Cleaning and Preprocessing

The Titanic dataset is examined for missing values and inconsistent data.

Missing values are analyzed using both counts and percentages. Appropriate methods are used to handle missing numerical and categorical values.

The notebook also identifies potential outliers in numerical variables using the **Interquartile Range (IQR)** method.

## Feature Engineering

Two additional features are created:

### FamilySize

'FamilySize' represents the total number of family members travelling with a passenger, including the passenger.

### IsAlone

'IsAlone' indicates whether a passenger travelled without other family members.

These features are used to further investigate relationships between family structure and passenger survival.

## Visualizations

The notebook generates figures to understand patterns in the Titanic dataset, including:

- Missing-value analysis
- Distributions of numerical variables
- Survival-related comparisons
- Correlation analysis
- Relationships between selected numerical variables

All final figures are stored in the 'figures/' directory.

## Key Observations

- The Titanic dataset contains missing values, particularly in the 'Age' and 'Cabin' columns.
- Missing-value percentages are calculated before applying imputation or other handling methods.
- Numerical missing values can be compared using mean, median, and other suitable approaches.
- The IQR method can be used to identify unusually high or low fare values.
- Passenger class, age, fare, and family-related variables provide useful information for studying survival patterns.
- 'FamilySize' and 'IsAlone' provide additional information that is not directly represented by a single original column.
- Correlation analysis helps identify relationships between numerical variables and survival.

> **Note:** The final numerical observations should be updated after the Titanic dataset loading issue in the notebook is corrected and Tasks 8–9 are re-run.

## Conclusion

This assignment demonstrates how NumPy and Pandas can be used together to perform a complete basic data-analysis workflow, from loading and cleaning data to feature engineering, statistical analysis, visualization, and interpretation of results.
