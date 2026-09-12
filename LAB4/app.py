import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Airbnb(NYC) Price Predictor",
    page_icon="🏠",
    layout="wide"
)


# ==================================================
# BACKGROUND IMAGE
# ==================================================

import os

background_image = os.path.join(
    os.path.dirname(__file__),
    "Background.jpg"
)

with open(background_image, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    f"""
    <style>

    /* ==========================================
    FULL BACKGROUND
    ========================================== */

    .stApp {{
        background-image:
            linear-gradient(
                rgba(0, 0, 0, 0.20),
                rgba(0, 0, 0, 0.20)
            ),
            url("data:image/jpeg;base64,{encoded_image}");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}


   /* ==========================================
   RIGHT SIDEBAR - DARK IN ALL MODES
   ========================================== */

[data-testid="stSidebar"] {
    position: fixed !important;
    left: auto !important;
    right: 0 !important;
    top: 0 !important;

    width: 320px !important;
    height: 100vh !important;

    background-color: #000000 !important;

    z-index: 999 !important;
}

/* Sidebar inner area */
[data-testid="stSidebar"] > div:first-child {
    background-color: #000000 !important;

    height: 100vh !important;

    overflow-y: auto !important;
    overflow-x: hidden !important;

    padding-bottom: 30px !important;
}

/* Sidebar text */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: white !important;
}

/* Sidebar divider */
[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.25) !important;
}


    /* ==========================================
    MAIN CONTENT
    ========================================== */

    [data-testid="stMainBlockContainer"] {{
        width: calc(100% - 340px) !important;

        max-width: none !important;

        margin-left: 0 !important;
        margin-right: 340px !important;

        padding-left: 35px !important;
        padding-right: 35px !important;
    }}


    /* ==========================================
    TITLE
    ========================================== */

    .title {{
        font-size: 42px;
        font-weight: 700;
        color: white;

        margin-bottom: 5px;
    }}


    /* ==========================================
    SUBTITLE
    ========================================== */

    .subtitle {{
        font-size: 18px;
        color: white;

        margin-bottom: 25px;
    }}


    /* ==========================================
    SECTION TITLES
       ========================================== */

    .section-title {{
        font-size: 24px;
        font-weight: 600;
        color: white;

        margin-top: 25px;
        margin-bottom: 15px;
    }}


    /* ==========================================
    INFO MESSAGE
       ========================================== */

    [data-testid="stAlert"] {{
        background-color: rgba(20, 22, 30, 0.88) !important;

        border: 1px solid rgba(255, 255, 255, 0.35) !important;

        border-radius: 12px !important;
    }}

    [data-testid="stAlert"] p {{
        color: white !important;

        font-weight: 500 !important;
    }}


    /* ==========================================
    INPUT BOXES
       ========================================== */

    div[data-baseweb="select"] > div {{
        background-color: rgba(20, 22, 30, 0.92) !important;
    }}

    div[data-baseweb="input"] > div {{
        background-color: rgba(20, 22, 30, 0.92) !important;
    }}


    /* ==========================================
    INPUT LABELS
    ========================================== */

    label {{
        color: white !important;
    }}


   /* ==========================================
   PREDICTION BOX - DARK IN ALL MODES
   ========================================== */

[data-testid="stMetric"] {
    background-color: rgba(20, 22, 30, 0.92) !important;
    border: 1px solid rgba(255, 255, 255, 0.30) !important;
    border-radius: 15px !important;
    padding: 25px !important;
}

[data-testid="stMetricLabel"] {
    color: white !important;
}

[data-testid="stMetricValue"] {
    color: white !important;
    font-size: 42px !important;
}


    /* ==========================================
    SUCCESS MESSAGE
    ========================================== */

    div[data-testid="stNotification"] {{
        border-radius: 10px !important;
    }}


    /* ==========================================
    SIDEBAR TEXT
    ========================================== */

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p {{
        color: white !important;
    }}


    /* ==========================================
    MOBILE / SMALL SCREEN
    ========================================== */

    @media (max-width: 900px) {{

        [data-testid="stMainBlockContainer"] {{
            width: 100% !important;

            margin-right: 0 !important;

            padding-left: 20px !important;
            padding-right: 20px !important;
        }}

    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

import os

@st.cache_resource
def load_model():
    model_path = os.path.join(
        os.path.dirname(__file__),
        "airbnb_price_pipeline.pkl"
    )
    return joblib.load(model_path)

model = load_model()


# =========================
# HEADER
# =========================

st.markdown(
    """
    <h1 style="
        text-align: center;
        color: white;
        font-size: 48px;
        font-weight: 700;
        margin-bottom: 5px;
    ">
         Airbnb(NYC) Price Predictor
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="
        text-align: center;
        color: white;
        font-size: 20px;
        margin-bottom: 30px;
    ">
        Predict the estimated nightly price of an Airbnb listing
    </div>
    """,
    unsafe_allow_html=True
)

# ==================================================
# INFORMATION MESSAGE
# ==================================================

st.info(
    "📌 This model was trained using the Airbnb NYC 2019 dataset. "
    "The prediction represents a historical estimate and should not "
    "be interpreted as a current 2026 market price."
)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.markdown("""  
    """)
    
    st.markdown("""
    **Created By:** Diya Tilwani  
    """)

    st.divider()

    st.header("📊 About the Model")

    st.write(
        "This application uses a Random Forest Regression model "
        "trained with a logarithmic transformation of the target price."
    )

    st.markdown("---")

    st.write("**Model:** Random Forest Regressor")

    st.write("**Target:** Nightly Airbnb Price")

    st.write("**Dataset:** Airbnb NYC 2019")

    st.write("**Evaluation:** MAE, RMSE and R²")

    st.markdown("---")

    st.caption(
        "The complete preprocessing workflow is stored "
        "together with the trained model."
    )


# ==================================================
# LOCATION INFORMATION
# ==================================================

st.markdown(
    '<div class="section-title">📍 Location Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:

    neighbourhood_group = st.selectbox(
        "Neighbourhood Group",
        [
            "Bronx",
            "Brooklyn",
            "Manhattan",
            "Queens",
            "Staten Island"
        ]
    )


with col2:

    neighbourhood_options = [
        "Williamsburg",
        "Bedford-Stuyvesant",
        "Harlem",
        "Bushwick",
        "Upper West Side",
        "Hell's Kitchen",
        "East Village",
        "Upper East Side",
        "Crown Heights",
        "Midtown"
    ]

    neighbourhood = st.selectbox(
        "Neighbourhood",
        neighbourhood_options
    )


with col3:

    room_type = st.selectbox(
        "Room Type",
        [
            "Entire home/apt",
            "Private room",
            "Shared room"
        ]
    )


# ==================================================
# GEOGRAPHIC COORDINATES
# ==================================================

st.markdown(
    '<div class="section-title">🗺️ Geographic Coordinates</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)


with col1:

    latitude = st.number_input(
        "Latitude",
        min_value=40.4,
        max_value=41.0,
        value=40.7180,
        step=0.0001,
        format="%.6f"
    )


with col2:

    longitude = st.number_input(
        "Longitude",
        min_value=-74.3,
        max_value=-73.6,
        value=-73.9950,
        step=0.0001,
        format="%.6f"
    )


# ==================================================
# LISTING INFORMATION
# ==================================================

st.markdown(
    '<div class="section-title">🏡 Listing Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:

    minimum_nights = st.number_input(
        "Minimum Nights",
        min_value=1,
        max_value=365,
        value=3,
        step=1
    )


with col2:

    number_of_reviews = st.number_input(
        "Number of Reviews",
        min_value=0,
        max_value=1000,
        value=10,
        step=1
    )


with col3:

    reviews_per_month = st.number_input(
        "Reviews per Month",
        min_value=0.0,
        max_value=100.0,
        value=1.0,
        step=0.1
    )


# ==================================================
# HOST & AVAILABILITY
# ==================================================

st.markdown(
    '<div class="section-title">👤 Host & Availability</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)


with col1:

    calculated_host_listings_count = st.number_input(
        "Host's Number of Listings",
        min_value=1,
        max_value=500,
        value=1,
        step=1
    )


with col2:

    availability_365 = st.slider(
        "Availability (Days per Year)",
        min_value=0,
        max_value=365,
        value=200
    )


# ==================================================
# CREATE INPUT DATAFRAME
# ==================================================

input_data = pd.DataFrame({

    "neighbourhood_group": [
        neighbourhood_group
    ],

    "neighbourhood": [
        neighbourhood
    ],

    "latitude": [
        latitude
    ],

    "longitude": [
        longitude
    ],

    "room_type": [
        room_type
    ],

    "minimum_nights": [
        minimum_nights
    ],

    "number_of_reviews": [
        number_of_reviews
    ],

    "reviews_per_month": [
        reviews_per_month
    ],

    "calculated_host_listings_count": [
        calculated_host_listings_count
    ],

    "availability_365": [
        availability_365
    ]
})


# ==================================================
# PREDICTION BUTTON
# ==================================================

st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])


with col2:

    predict_button = st.button(
        "💰 Predict Airbnb Price",
        type="primary",
        use_container_width=True
    )


# ==================================================
# PREDICTION
# ==================================================

if predict_button:

    try:

        # Model predicts log(price + 1)
        prediction_log = model.predict(input_data)

        # Convert back to original price
        predicted_price = np.expm1(prediction_log[0])

        # ------------------------------------------
        # PREDICTION RESULT
        # ------------------------------------------

        st.markdown(
            '<div class="section-title">💰 Prediction Result</div>',
            unsafe_allow_html=True
        )

        st.metric(
            label="Estimated Nightly Price",
            value=f"${predicted_price:,.2f}"
        )

        st.success(
            "✅ Prediction generated successfully."
        )


        
        st.caption(
            "The prediction is generated using the trained "
            "Random Forest model with a log-transformed target."
        )


    except Exception as e:

        st.error(
            "An error occurred while generating the prediction."
        )

        st.exception(e)


# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.caption(
    "M.Sc. Data Science | Airbnb Price Prediction Project"
)
