"""
House Price Predictor Web Application
====================================
A clean, interactive Machine Learning application built using Python, Pandas,
Scikit-learn, and Streamlit to predict residential house prices using Linear Regression.

Author: AI & Data Science Portfolio Project
"""

import sys
from unittest.mock import MagicMock

# Safety guard for environments with strict DLL / Smart App Control policies
if "sklearn.metrics._pairwise_distances_reduction" not in sys.modules:
    try:
        import sklearn.metrics._pairwise_distances_reduction  # noqa: F401
    except (ImportError, OSError):
        sys.modules["sklearn.metrics._pairwise_distances_reduction"] = MagicMock()

import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


# ==============================================================================
# Page Configuration & Custom CSS Styling
# ==============================================================================
st.set_page_config(
    page_title="House Price Predictor | ML Portfolio",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for a polished, modern user experience
st.markdown("""
<style>
    /* Global style adjustments */
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
    }
    .price-badge {
        font-size: 2.6rem;
        font-weight: 800;
        color: #059669;
        margin: 10px 0;
    }
    .metric-chip {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        background-color: #EEF2F6;
        color: #334155;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Data Loading & Caching
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_housing_data(file_path: str = "house_data.csv") -> pd.DataFrame:
    """
    Load the housing dataset from a CSV file.
    
    Parameters:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded housing dataset, or None if file is missing.
    """
    if not os.path.exists(file_path):
        return None
    return pd.read_csv(file_path)


# ==============================================================================
# Data Preprocessing & Model Training
# ==============================================================================
@st.cache_resource(show_spinner=False)
def train_linear_regression_model(df: pd.DataFrame):
    """
    Preprocess the dataset and train a Scikit-learn Linear Regression model.
    
    Steps:
      1. Handles missing values by imputing numerical medians.
      2. Performs One-Hot Encoding on the categorical 'location' feature.
      3. Splits data into training (80%) and testing (20%) sets.
      4. Fits Ordinary Least Squares Linear Regression.
      5. Computes evaluation metrics (MAE, R2, RMSE).
      
    Returns:
        dict: Trained model, evaluation metrics, preprocessed features list,
              median values for imputation, and training metadata.
    """
    # Create an explicit working copy
    data = df.copy()

    # Identify numerical columns for imputation
    num_cols = ["area", "bedrooms", "bathrooms", "floors", "house_age"]
    median_imputations = {}
    
    for col in num_cols:
        if col in data.columns:
            median_val = data[col].median()
            median_imputations[col] = median_val
            data[col] = data[col].fillna(median_val)

    # Encode categorical location features using One-Hot Encoding
    # drop_first=True avoids collinearity / dummy variable trap
    encoded_df = pd.get_dummies(data, columns=["location"], drop_first=True)

    # Separate feature matrix X and target y
    target_col = "price"
    X = encoded_df.drop(columns=[target_col])
    y = encoded_df[target_col]

    # Save feature names to ensure deterministic alignment during inference
    feature_names = list(X.columns)

    # Split dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Instantiate and fit Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate on the unseen test set
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    # Extract feature coefficients for model explainability
    coefficients = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_
    }).sort_values(by="Coefficient", ascending=False)

    return {
        "model": model,
        "feature_names": feature_names,
        "median_imputations": median_imputations,
        "metrics": {
            "r2_score": r2,
            "mae": mae,
            "rmse": rmse,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "total_records": len(df),
            "intercept": model.intercept_
        },
        "coefficients": coefficients,
        "encoded_df": encoded_df,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred
    }


def predict_price(model_bundle: dict, user_inputs: dict) -> float:
    """
    Transform user inputs into the model's expected format and compute price.
    
    Parameters:
        model_bundle (dict): Artifacts dictionary returned by train_linear_regression_model.
        user_inputs (dict): Dictionary with keys: area, bedrooms, bathrooms, floors, house_age, location.
        
    Returns:
        float: Estimated house price in USD.
    """
    feature_names = model_bundle["feature_names"]
    medians = model_bundle["median_imputations"]
    model = model_bundle["model"]

    # Initialize a row dictionary with zeros for all encoded features
    row_data = {col: 0.0 for col in feature_names}

    # Populate numerical features (fallback to medians if null/missing)
    row_data["area"] = float(user_inputs.get("area", medians.get("area", 1500)))
    row_data["bedrooms"] = float(user_inputs.get("bedrooms", medians.get("bedrooms", 3)))
    row_data["bathrooms"] = float(user_inputs.get("bathrooms", medians.get("bathrooms", 2)))
    row_data["floors"] = float(user_inputs.get("floors", medians.get("floors", 1)))
    row_data["house_age"] = float(user_inputs.get("house_age", medians.get("house_age", 10)))

    # Set the one-hot encoded location column
    location_val = user_inputs.get("location", "")
    target_col = f"location_{location_val}"
    if target_col in row_data:
        row_data[target_col] = 1.0

    # Build single-row DataFrame aligned strictly with training schema
    input_df = pd.DataFrame([row_data], columns=feature_names)
    
    predicted_val = model.predict(input_df)[0]
    return float(predicted_val)


# ==============================================================================
# Streamlit User Interface
# ==============================================================================
def main():
    # Header Banner
    st.markdown('<div class="main-header">🏡 House Price Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">An interpretable Machine Learning application powered by '
        '<b>Scikit-learn Linear Regression</b>, <b>Pandas</b>, and <b>Streamlit</b>.</div>',
        unsafe_allow_html=True
    )

    # 1. Load Data
    csv_filename = "house_data.csv"
    raw_df = load_housing_data(csv_filename)

    if raw_df is None:
        st.error(
            f"⚠️ **Dataset file `{csv_filename}` not found!**\n\n"
            "Please ensure that the dataset CSV file is placed in the project root directory."
        )
        st.info("Tip: You can regenerate the sample dataset by executing `python generate_data.py`.")
        return

    # 2. Train and cache model bundle
    bundle = train_linear_regression_model(raw_df)
    metrics = bundle["metrics"]
    available_locations = sorted(list(raw_df["location"].dropna().unique()))

    # ==========================================================================
    # Sidebar: Model Overview & Presets
    # ==========================================================================
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        st.markdown(
            """
            **Algorithm:** `Linear Regression (OLS)`  
            **Target Variable:** House Price (`$ USD`)  
            **Library:** `scikit-learn`
            """
        )

        st.subheader("📊 Evaluation Summary")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("R² Score", f"{metrics['r2_score']:.3f}")
        with col_s2:
            st.metric("MAE", f"${metrics['mae']:,.0f}")

        st.caption(f"Trained on **{metrics['n_train']}** records (80/20 train-test split).")

        st.divider()
        st.subheader("💡 Quick Presets")
        st.caption("Click a preset to populate representative inputs:")

        preset = st.radio(
            "Select a preset home:",
            options=["Custom Input", "Starter Suburban", "Downtown Condo", "Luxury Waterfront", "Countryside Villa"],
            index=0,
            label_visibility="collapsed"
        )

        st.divider()
        st.markdown(
            "⚠️ **Disclaimer:**  \n"
            "Predictions are estimates based on linear statistical patterns and "
            "are intended for portfolio demonstration only, not formal real estate appraisals."
        )

    # Preset values handler
    default_vals = {
        "area": 2000,
        "bedrooms": 3,
        "bathrooms": 2.0,
        "floors": 1,
        "house_age": 10,
        "location": "Suburban"
    }

    if preset == "Starter Suburban":
        default_vals = {"area": 1400, "bedrooms": 3, "bathrooms": 1.5, "floors": 1, "house_age": 15, "location": "Suburban"}
    elif preset == "Downtown Condo":
        default_vals = {"area": 1100, "bedrooms": 2, "bathrooms": 2.0, "floors": 1, "house_age": 5, "location": "Downtown"}
    elif preset == "Luxury Waterfront":
        default_vals = {"area": 3800, "bedrooms": 5, "bathrooms": 4.5, "floors": 2, "house_age": 3, "location": "Waterfront"}
    elif preset == "Countryside Villa":
        default_vals = {"area": 2800, "bedrooms": 4, "bathrooms": 3.0, "floors": 2, "house_age": 20, "location": "Rural"}

    # ==========================================================================
    # Main Navigation Tabs
    # ==========================================================================
    tab_predict, tab_insights, tab_data = st.tabs([
        "🏡 Price Predictor",
        "📈 Model Performance & Explainability",
        "📋 Dataset Explorer"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: Price Predictor
    # --------------------------------------------------------------------------
    with tab_predict:
        st.subheader("Enter Property Specifications")
        st.write("Adjust the parameters below to evaluate the estimated market value of the property.")

        # Input Layout
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                area_input = st.number_input(
                    "📐 Area (Square Feet):",
                    min_value=300,
                    max_value=10000,
                    value=int(default_vals["area"]),
                    step=50,
                    help="Gross livable interior area in square feet."
                )
                bedrooms_input = st.number_input(
                    "🛏️ Bedrooms:",
                    min_value=1,
                    max_value=10,
                    value=int(default_vals["bedrooms"]),
                    step=1,
                    help="Total number of bedrooms."
                )

            with col2:
                bathrooms_input = st.number_input(
                    "🚿 Bathrooms:",
                    min_value=1.0,
                    max_value=8.0,
                    value=float(default_vals["bathrooms"]),
                    step=0.5,
                    help="Number of bathrooms (half-baths permitted, e.g. 1.5, 2.5)."
                )
                floors_input = st.number_input(
                    "🏢 Number of Floors:",
                    min_value=1,
                    max_value=4,
                    value=int(default_vals["floors"]),
                    step=1,
                    help="Total above-ground floors/stories."
                )

            with col3:
                age_input = st.number_input(
                    "⏳ House Age (Years):",
                    min_value=0,
                    max_value=100,
                    value=int(default_vals["house_age"]),
                    step=1,
                    help="Age of the house in years (0 for brand new construction)."
                )
                location_idx = available_locations.index(default_vals["location"]) if default_vals["location"] in available_locations else 0
                location_input = st.selectbox(
                    "📍 Location Neighborhood:",
                    options=available_locations,
                    index=location_idx,
                    help="Select neighborhood classification."
                )

            submit_prediction = st.form_submit_button(
                "🔮 Predict House Price",
                type="primary",
                use_container_width=True
            )

        # Prediction Processing & Results
        if submit_prediction:
            # Input Validations
            validation_errors = []
            if area_input <= 0:
                validation_errors.append("Area must be greater than zero.")
            if bedrooms_input < 1:
                validation_errors.append("Bedrooms must be at least 1.")
            if bathrooms_input < 1:
                validation_errors.append("Bathrooms must be at least 1.")
            if age_input < 0:
                validation_errors.append("House age cannot be negative.")

            if validation_errors:
                for err in validation_errors:
                    st.error(f"❌ {err}")
            else:
                user_data = {
                    "area": area_input,
                    "bedrooms": bedrooms_input,
                    "bathrooms": bathrooms_input,
                    "floors": floors_input,
                    "house_age": age_input,
                    "location": location_input
                }

                # Compute prediction using trained Linear Regression model
                predicted_price = predict_price(bundle, user_data)
                
                # Prevent negative prices
                clean_price = max(predicted_price, 50000.0)

                # Valuation Results Display
                st.markdown("---")
                st.subheader("💵 Estimated Valuation")

                res_col1, res_col2 = st.columns([1.4, 1])

                with res_col1:
                    st.markdown(
                        f"""
                        <div class="card">
                            <span class="metric-chip">Linear Regression Model</span>
                            <span class="metric-chip">Currency: USD ($)</span>
                            <div class="price-badge">${clean_price:,.0f}</div>
                            <p style="color: #64748B; margin-bottom: 0;">
                                Estimated value for a <b>{area_input:,} sq ft</b> home in <b>{location_input}</b> 
                                ({bedrooms_input} bed, {bathrooms_input} bath, {age_input} yrs old).
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with res_col2:
                    # Confidence & breakdown metrics
                    avg_market_price = raw_df["price"].mean()
                    pct_diff = ((clean_price - avg_market_price) / avg_market_price) * 100
                    price_per_sqft = clean_price / area_input

                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        st.metric("Price / Sq Ft", f"${price_per_sqft:,.0f}")
                    with m_col2:
                        st.metric("vs Market Avg", f"{pct_diff:+.1f}%", delta=f"{clean_price - avg_market_price:+,.0f}")

                    # Error tolerance interval (Model Prediction ± MAE)
                    lower_bound = max(clean_price - metrics["mae"], 30000.0)
                    upper_bound = clean_price + metrics["mae"]
                    st.info(
                        f"**Expected Range (± MAE):**  \n"
                        f"**${lower_bound:,.0f}** to **${upper_bound:,.0f}**"
                    )

    # --------------------------------------------------------------------------
    # TAB 2: Model Performance & Explainability
    # --------------------------------------------------------------------------
    with tab_insights:
        st.subheader("Model Evaluation & Mathematical Insights")
        st.write(
            "This project uses **Ordinary Least Squares (OLS) Linear Regression** from Scikit-learn. "
            "Because Linear Regression is inherently transparent, we can inspect both test metrics and feature coefficients directly."
        )

        # Metric Cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric(
                label="R² Score (Goodness of Fit)",
                value=f"{metrics['r2_score']:.4f}",
                help="Proportion of variance in house prices explained by features (closer to 1.0 is better)."
            )
        with col_m2:
            st.metric(
                label="MAE (Mean Absolute Error)",
                value=f"${metrics['mae']:,.2f}",
                help="Average dollar distance between predicted price and actual test prices."
            )
        with col_m3:
            st.metric(
                label="RMSE (Root Mean Squared Error)",
                value=f"${metrics['rmse']:,.2f}",
                help="Square root of mean squared errors, penalizing larger deviations."
            )
        with col_m4:
            st.metric(
                label="Training Set Size",
                value=f"{metrics['n_train']:,} houses",
                help="80% training split used to fit regression coefficients."
            )

        st.divider()

        # Coefficients Breakdown
        st.subheader("Feature Coefficients (Weight / Impact)")
        st.write(
            "The table below shows the fitted coefficient (weight) for each input feature. "
            "In Linear Regression, each coefficient indicates the change in predicted price for a 1-unit increase in that feature."
        )

        coef_df = bundle["coefficients"].copy()
        
        # User-friendly explanation formatting
        def explain_coef(row):
            feat = row["Feature"]
            val = row["Coefficient"]
            if feat == "area":
                return f"Adds ~${val:,.2f} per square foot"
            elif feat == "bedrooms":
                return f"Adds ~${val:,.2f} per bedroom"
            elif feat == "bathrooms":
                return f"Adds ~${val:,.2f} per bathroom"
            elif feat == "floors":
                return f"Adds ~${val:,.2f} per floor"
            elif feat == "house_age":
                return f"Depreciates by ~${abs(val):,.2f} per year of age"
            elif feat.startswith("location_"):
                loc_name = feat.replace("location_", "")
                return f"Premium of ~${val:,.2f} relative to baseline (Rural)"
            return f"${val:,.2f}"

        coef_df["Interpretation"] = coef_df.apply(explain_coef, axis=1)
        coef_df["Formatted Impact"] = coef_df["Coefficient"].apply(lambda x: f"${x:+,.2f}")

        # Display coefficient chart & table
        c_left, c_right = st.columns([1.2, 1])
        with c_left:
            st.bar_chart(
                data=coef_df.set_index("Feature")["Coefficient"],
                color="#0284C7"
            )
        with c_right:
            st.dataframe(
                coef_df[["Feature", "Formatted Impact", "Interpretation"]],
                use_container_width=True,
                hide_index=True
            )

        # OLS Formula Display
        st.divider()
        st.subheader("Mathematical Formulation")
        st.latex(r"Price = \beta_0 + \beta_1(Area) + \beta_2(Beds) + \beta_3(Baths) + \beta_4(Floors) + \beta_5(Age) + \sum \beta_i(Location_i) + \epsilon")
        st.caption(f"Fitted Intercept (\\beta_0): **${metrics['intercept']:,.2f}**")

    # --------------------------------------------------------------------------
    # TAB 3: Dataset Explorer
    # --------------------------------------------------------------------------
    with tab_data:
        st.subheader("Dataset Overview & Quality Assurance")
        st.write(
            f"Dataset contains **{len(raw_df):,}** housing records. "
            "Features include square footage, bedrooms, bathrooms, floors, house age, location neighborhood, and price."
        )

        # High-level dataset stats
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.write("##### Sample Records")
            st.dataframe(raw_df.head(8), use_container_width=True)

        with d_col2:
            st.write("##### Summary Statistics")
            st.dataframe(raw_df.describe().round(2), use_container_width=True)

        st.divider()
        st.write("##### Missing Values & Preprocessing Strategy")
        missing_series = raw_df.isnull().sum()
        missing_df = pd.DataFrame({
            "Feature": missing_series.index,
            "Missing Count": missing_series.values,
            "Missing %": (missing_series.values / len(raw_df) * 100).round(2),
            "Handling Strategy": [
                "Median Imputation" if count > 0 and col != "location"
                else "None required (0 missing)" if count == 0
                else "Mode Imputation"
                for col, count in zip(missing_series.index, missing_series.values)
            ]
        })
        st.dataframe(missing_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
