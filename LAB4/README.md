# 🏠 Airbnb Price Prediction

## 📌 Project Overview

This project is an end-to-end machine learning application for
predicting the **nightly price of an Airbnb listing in New York City**.

The project uses the **AB_NYC_2019** dataset and covers the complete
machine learning workflow:

1.  Data analysis and preparation
2.  Exploratory Data Analysis (EDA)
3.  Data preprocessing
4.  Feature selection
5.  Outlier analysis
6.  Regression model training
7.  Model comparison
8.  Hyperparameter tuning
9.  Overfitting/generalization analysis
10. Final model selection
11. Model serialization
12. Streamlit application development
13. Deployment

------------------------------------------------------------------------

# 🎯 Objective

The objective is to build a regression model that can estimate the
nightly Airbnb price using listing characteristics such as:

-   Location
-   Room type
-   Minimum nights
-   Number of reviews
-   Reviews per month
-   Host listing count
-   Availability
-   Latitude and longitude

The target variable is:

``` text
price
```

which represents the **nightly Airbnb price in USD**.

------------------------------------------------------------------------

# 📂 Dataset

### Dataset

**AB_NYC_2019.csv**

The original dataset contains:

-   **48,895 rows**
-   **16 columns**

Original columns:

``` text
id
name
host_id
host_name
neighbourhood_group
neighbourhood
latitude
longitude
room_type
price
minimum_nights
number_of_reviews
last_review
reviews_per_month
calculated_host_listings_count
availability_365
```

### Initial Missing Values

  Column                Missing Values
  ------------------- ----------------
  name                              16
  host_name                         21
  last_review                   10,052
  reviews_per_month             10,052
  Other columns                      0

------------------------------------------------------------------------

# 🧹 Data Cleaning and Preprocessing

## Missing Values

The following missing-value treatments were applied:

``` python
df["name"] = df["name"].fillna("Unknown")
df["host_name"] = df["host_name"].fillna("Unknown")
df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
```

The preprocessing pipeline also uses:

-   Median imputation for numerical features
-   Most-frequent imputation for categorical features

This makes the final pipeline robust to missing values during
prediction.

------------------------------------------------------------------------

# 🚨 Outlier Analysis

The IQR method was used to identify potential price outliers.

### Results

``` text
Q1 = 69
Q3 = 175
IQR = 106

Lower Bound = -90
Upper Bound = 334
```

The IQR method identified:

``` text
2,972 potential outliers
approximately 6.08% of the dataset
```

There were also:

``` text
11 records with price = 0
```

The zero-price records were removed because they represent invalid
target values.

High-price listings were **not automatically removed** because some
expensive Airbnb listings can represent legitimate luxury properties.

Therefore, instead of removing all statistical outliers, the strong
right-skew of the target was handled using a logarithmic target
transformation.

### Final approach

``` python
y_train_log = np.log1p(y_train)
```

After prediction, the value was converted back using:

``` python
predicted_price = np.expm1(prediction_log)
```

------------------------------------------------------------------------

# 🗑️ Removed Features

The following columns were removed before modeling:

``` text
id
host_id
name
host_name
last_review
```

### Reason

-   `id` and `host_id` are identifiers and do not provide meaningful
    generalizable pricing information.
-   `name` and `host_name` are free-text/name fields and were not used
    in the final model.
-   `last_review` is a historical date attribute and was not considered
    directly useful for predicting the historical nightly price.

The review-related numerical features were retained:

``` text
number_of_reviews
reviews_per_month
```

No additional `review_year` feature was used.

------------------------------------------------------------------------

# 🔧 Final Model Features

The final model uses **10 input features**:

### Numerical Features

``` text
latitude
longitude
minimum_nights
number_of_reviews
reviews_per_month
calculated_host_listings_count
availability_365
```

### Categorical Features

``` text
neighbourhood_group
neighbourhood
room_type
```

------------------------------------------------------------------------

# ⚙️ Preprocessing Pipeline

The project uses a Scikit-learn `ColumnTransformer` and `Pipeline`.

### Numerical preprocessing

``` text
Median Imputation
        ↓
StandardScaler
```

### Categorical preprocessing

``` text
Most-Frequent Imputation
        ↓
One-Hot Encoding
```

The categorical encoder uses:

``` python
OneHotEncoder(handle_unknown="ignore")
```

This allows the deployed application to handle unseen categorical values
without crashing.

------------------------------------------------------------------------

# 📊 Exploratory Data Analysis

The notebook contains analysis and visualizations covering:

-   Distribution of listings by room type
-   Price distribution by room type
-   Number of listings by neighbourhood group
-   Price distribution by neighbourhood group
-   Minimum nights vs. price
-   Number of reviews vs. price
-   Availability vs. price
-   Correlation heatmap
-   Top 10 neighbourhoods by listing count
-   Price distribution in the top 10 neighbourhoods
-   Numerical summary statistics
-   Feature skewness

### Important observations

The `price` variable was strongly right-skewed and contained a long
high-price tail.

The analysis also showed meaningful price differences across:

-   Room types
-   Neighbourhood groups
-   Neighbourhoods

These observations motivated the use of categorical encoding and
logarithmic target transformation.

------------------------------------------------------------------------

# 🤖 Machine Learning Models

The following regression models were evaluated:

1.  Linear Regression
2.  Decision Tree Regressor
3.  Random Forest Regressor
4.  Random Forest + Log Target
5.  Tuned Random Forest + Log Target
6.  Gradient Boosting + Log Target
7.  HistGradientBoosting + Log Target

The data was split into:

``` text
Training set: 80%
Test set: 20%
random_state = 42
```

Final split sizes:

``` text
X_train = 39,107 rows
X_test  = 9,777 rows
```

------------------------------------------------------------------------

# 📈 Model Results

Evaluation metrics:

-   **MAE** --- Mean Absolute Error
-   **RMSE** --- Root Mean Squared Error
-   **R²** --- Coefficient of Determination

  Model                                     MAE ↓       RMSE ↓        R² ↑
  ----------------------------------- ----------- ------------ -----------
  Linear Regression                         70.83       186.08       0.135
  Decision Tree                             73.20       283.31      -1.005
  Random Forest                             65.75       193.11       0.069
  **Random Forest + Log Target**        **55.88**   **175.54**   **0.230**
  Tuned Random Forest + Log Target          55.62       178.52       0.204
  Gradient Boosting + Log Target            58.18       186.11       0.135
  HistGradientBoosting + Log Target         56.08       178.39       0.205

------------------------------------------------------------------------

# 🏆 Final Model Selection

The final selected model is:

## Random Forest Regressor with Log-Transformed Target

The model was trained using:

``` python
RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
```

The target was transformed using:

``` python
np.log1p(price)
```

and converted back after prediction using:

``` python
np.expm1(prediction)
```

### Final Test Performance

``` text
MAE  = 55.88
RMSE = 175.54
R²   = 0.230
```

The log-transformed Random Forest was selected because it provided the
**best overall held-out test performance** among the evaluated models.

Although the tuned Random Forest produced a slightly lower MAE (55.62),
its RMSE and R² were worse than the selected baseline log-target Random
Forest. Therefore, the baseline log-target Random Forest was retained as
the final model.

------------------------------------------------------------------------

# 🔍 Overfitting / Generalization Check

For the selected final model:

### Training performance

``` text
MAE  = 42.02
RMSE = 179.34
R²   = 0.482
```

### Test performance

``` text
MAE  = 55.88
RMSE = 175.54
R²   = 0.230
```

The difference between training and test R² indicates **some degree of
overfitting**.

However, the selected model still provided better overall generalization
performance than the other evaluated models.

------------------------------------------------------------------------

# 🔎 Hyperparameter Tuning

`RandomizedSearchCV` was used for Random Forest hyperparameter tuning.

Parameters explored included:

``` python
n_estimators:
[100, 150, 200]

max_depth:
[10, 15, 20, 25]

min_samples_split:
[2, 5, 10]

min_samples_leaf:
[1, 2, 4]
```

The search used:

``` text
n_iter = 10
cv = 3
scoring = R²
random_state = 42
```

The tuned model achieved:

``` text
MAE  = 55.62
RMSE = 178.52
R²   = 0.204
```

Since the selected baseline log-target Random Forest performed better on
the held-out test set for RMSE and R², it was retained as the final
model.

------------------------------------------------------------------------

# 💾 Saved Model Pipeline

The final model was saved as:

``` text
airbnb_price_pipeline.pkl
```

The saved pipeline contains:

``` text
Preprocessing
    ↓
One-Hot Encoding / Scaling
    ↓
Random Forest Regressor
```

The Streamlit application loads this saved pipeline directly and does
**not retrain the model**.

This ensures that the deployed application uses the same trained model
developed during the notebook workflow.

------------------------------------------------------------------------

# 🌐 Streamlit Application

A Streamlit web application was developed for interactive prediction.

The user provides:

``` text
Neighbourhood Group
Neighbourhood
Latitude
Longitude
Room Type
Minimum Nights
Number of Reviews
Reviews per Month
Calculated Host Listings Count
Availability
```

The application then returns:

``` text
Estimated Nightly Price
```

The model prediction is generated using the saved pipeline and converted
from the log scale back to USD using `np.expm1()`.

### Example Test

One test input used in the application was:

``` text
Neighbourhood Group: Brooklyn
Neighbourhood: Williamsburg
Latitude: 40.7180
Longitude: -73.9600
Room Type: Entire home/apt
Minimum Nights: 3
Number of Reviews: 50
Reviews per Month: 1.5
Calculated Host Listings Count: 2
Availability: 200
```

The application produced an estimated nightly price of approximately:

``` text
$247
```

This was used as a functional test to verify that the deployed
application was successfully loading the saved model and returning a
prediction.

------------------------------------------------------------------------

# 🛠️ Technologies Used

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Matplotlib
-   Seaborn
-   Joblib
-   Streamlit
-   Jupyter Notebook

------------------------------------------------------------------------

# 📁 Project Structure

``` text
202618011_DiyaTilwani_DS605/
│
└── LAB4/
    │
    ├── 202618011_Lab04.ipynb
    ├── AB_NYC_2019.csv
    ├── Background.jpg
    ├── airbnb_price_pipeline.pkl
    ├── app.py
    ├── requirements.txt
    ├── README.md
    │
    └── screenshots/
        └── streamlit_app.png
```

------------------------------------------------------------------------

# 📦 Requirements

The project dependencies are listed in `requirements.txt`:

``` text
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
joblib
```

------------------------------------------------------------------------

# ▶️ Run the Application Locally

### 1. Clone the repository

``` bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

### 2. Navigate to the LAB4 folder

``` bash
cd 202618011_DiyaTilwani_DS605/LAB4
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Run Streamlit

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

# 🚀 Deployed Application

**Streamlit Cloud:**\
Add the final deployed application URL here.

------------------------------------------------------------------------

# 📸 Application Screenshot

A screenshot of the working Streamlit application is included in:

``` text
screenshots/streamlit_app.png
```

The screenshot demonstrates the application's user interface and
prediction output.

------------------------------------------------------------------------

# ⚠️ Deployment Issues Encountered and Resolved

During Streamlit deployment, two file path issues were encountered.

## 1. Background Image FileNotFoundError

Initially, the application used a local Windows path for the background
image:

``` text
D:\202618011_DS605\...
```

This worked locally but failed on Streamlit Cloud because the deployment
environment does not have access to the local Windows filesystem.

The path was changed to use the directory containing `app.py`:

``` python
background_image = os.path.join(
    os.path.dirname(__file__),
    "Background.jpg"
)
```

This allows the application to locate the image correctly in the
deployed repository.

## 2. Saved Model FileNotFoundError

The application initially loaded the model using:

``` python
joblib.load("airbnb_price_pipeline.pkl")
```

This caused a `FileNotFoundError` during deployment because the
application could not reliably locate the model file from the current
working directory.

It was changed to:

``` python
model_path = os.path.join(
    os.path.dirname(__file__),
    "airbnb_price_pipeline.pkl"
)

return joblib.load(model_path)
```

This makes the model path relative to the location of `app.py`.

Both `Background.jpg` and `airbnb_price_pipeline.pkl` are stored in the
same `LAB4` directory as `app.py`.

------------------------------------------------------------------------

# 👩‍💻 Author

**Diya Tilwani**

**Enrollment No.: 202618011**

### Course

**DS605 -- Fundamentals of Machine Learning**

### Assignment

**Lab Assignment 4 -- End-to-End Airbnb Price Prediction**

------------------------------------------------------------------------

# 📌 Conclusion

This project demonstrates a complete machine learning pipeline for
Airbnb price prediction, from data preparation and exploratory analysis
to model comparison, tuning, evaluation, serialization, and deployment.

The final Random Forest model with a logarithmically transformed target
achieved:

``` text
MAE  = 55.88
RMSE = 175.54
R²   = 0.230
```

The saved pipeline is integrated with a Streamlit application, allowing
users to enter Airbnb listing characteristics and receive an estimated
nightly price.
