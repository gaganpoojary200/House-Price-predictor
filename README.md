# 🏡 House Price Predictor

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458.svg)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning web application designed to predict residential property prices based on key housing attributes. Built with **Scikit-learn (Linear Regression)**, **Pandas**, and an interactive, clean **Streamlit** user interface.

Created as a portfolio-ready demonstration of classical regression, proper data preprocessing, zero-data-leakage pipeline design, and interpretable machine learning.

---

## 📌 Table of Contents

- [Overview & Goal](#-overview--goal)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Dataset Details & Source](#-dataset-details--source)
- [Machine Learning Model](#-machine-learning-model)
  - [How Linear Regression Works Here](#how-linear-regression-works-here)
  - [Evaluation Metrics](#model-evaluation-metrics)
  - [Feature Coefficients & Interpretability](#feature-coefficients--interpretability)
- [Project Structure](#-project-structure)
- [Installation & Local Setup](#-installation--local-setup)
- [How to Run the Application](#-how-to-run-the-application)
- [Example Prediction Walkthrough](#-example-prediction-walkthrough)
- [Screenshots & UI Showcase](#-screenshots--ui-showcase)
- [Future Improvements](#-future-improvements)
- [License](#-license)

---

## 🎯 Overview & Goal

The primary goal of this application is to estimate residential property valuations using classical supervised machine learning. Rather than relying on black-box neural networks, this project utilizes **Ordinary Least Squares (OLS) Linear Regression** to guarantee complete mathematical explainability, transparent feature weighting, and fast real-time inference.

Users can configure property specifications through a web UI and instantly receive an estimated valuation, a ± MAE error tolerance interval, per-square-foot valuation metrics, and contextual comparisons against broader market averages.

---

## ✨ Key Features

- **Intuitive Property Configurator:** Easily enter or tweak house features including area (sq ft), bedrooms, bathrooms, floor count, house age, and neighborhood location.
- **One-Click Presets:** Instant pre-filling of representative home types (*Starter Suburban*, *Downtown Condo*, *Luxury Waterfront*, *Countryside Villa*).
- **Instant Valuation Engine:** Computes predicted house price in USD (`$`) using a model trained on 1,500 property records.
- **Robust Input Validation:** Guards against invalid parameters (negative ages, zero area, missing inputs) with user-friendly alerts.
- **Interactive Model Explainability:**
  - Complete metrics panel ($R^2$ score, MAE, RMSE, training record count).
  - Feature coefficient bar chart and breakdown showing dollar-for-dollar impacts (e.g. `+$165/sq ft`, `-$1,800/yr of age`).
  - Mathematical formulation displayed in LaTeX.
- **Integrated Dataset Explorer:** View sample data, statistical distributions (`describe()`), and automated missing-value handling logs.
- **Zero Data Leakage:** Preprocessing and median imputations strictly fit on training data, preventing information bleed.

---

## 🛠 Tech Stack

| Technology | Purpose |
| :--- | :--- |
| **Python** | Core programming language |
| **Streamlit** | Interactive web application framework & UI |
| **Scikit-learn** | Linear Regression modeling, dataset splitting, evaluation metrics |
| **Pandas** | Data wrangling, tabular manipulation, One-Hot Encoding |
| **NumPy** | Numerical computations and vector operations |

---

## 📊 Dataset Details & Source

The dataset (`house_data.csv`) comprises **1,500 residential property records** modeled after realistic real-estate economic fundamentals.

### Features Included

1. `area`: Total livable interior floor space in square feet (650 – 4,800 sq ft).
2. `bedrooms`: Number of bedrooms (1 to 6).
3. `bathrooms`: Number of bathrooms (1.0 to 5.0, supporting half-baths).
4. `floors`: Number of above-ground levels/stories (1 to 3).
5. `house_age`: Structure age in years (0 to 50 years).
6. `location`: Neighborhood category (`Downtown`, `Suburban`, `Uptown`, `Waterfront`, `Rural`).
7. `price`: House market valuation in **US Dollars ($ USD)** (Target variable).

### Currency & Limitations
- **Currency:** US Dollars (`$ USD`).
- **Missing Value Handling:** A realistic ~1.5% missing value rate was introduced in raw data (`house_age`, `bathrooms`) to demonstrate robust median imputation.
- **Limitations:** The dataset focuses on structural and primary location factors. Micro-amenities (school ratings, lot size, HOA fees, renovation finishes) are unmodeled and would require expanded feature collection for production real-estate appraisal.

> **Reproducibility Note:** The dataset can be regenerated anytime with deterministic seeds using `python generate_data.py`.

---

## 🧠 Machine Learning Model

### How Linear Regression Works Here

Linear regression assumes that the target variable ($Price$) is a linear combination of the input features plus an intercept ($\beta_0$) and an error term ($\epsilon$):

$$\text{Price} = \beta_0 + \beta_1(\text{Area}) + \beta_2(\text{Bedrooms}) + \beta_3(\text{Bathrooms}) + \beta_4(\text{Floors}) + \beta_5(\text{Age}) + \sum_{k} \beta_k(\text{Location}_k) + \epsilon$$

1. **Handling Categoricals:** The `location` column is transformed using **One-Hot Encoding** (`drop_first=True`) with `Rural` serving as the baseline reference category.
2. **Train/Test Split:** An 80/20 train-test split (`random_state=42`) allocates 1,200 samples for training and 300 unseen samples for test evaluation.
3. **Optimization:** Scikit-learn's `LinearRegression` solves for the coefficient vector $\beta$ using Ordinary Least Squares (`scipy.linalg.lstsq`).

### Model Evaluation Metrics

Evaluated on the unseen 20% test partition (300 records):

| Metric | Value | Description |
| :--- | :--- | :--- |
| **$R^2$ Score** | **0.977** (97.7%) | Explains 97.7% of the total variance in test house prices. |
| **MAE (Mean Absolute Error)** | **~$19,384** | The average absolute difference between predicted and actual home prices. |
| **RMSE (Root Mean Squared Error)** | **~$24,450** | Standard deviation of the prediction residuals. |
| **Training Sample Count** | **1,200** | Number of property observations used for fitting. |

### Feature Coefficients & Interpretability

Because Linear Regression is linear and additive, each coefficient provides actionable real-world insights:
- **`area`**: Adds approximately **+$165** to home value per square foot.
- **`bedrooms`**: Adds approximately **+$18,000** per additional bedroom.
- **`bathrooms`**: Adds approximately **+$22,000** per additional bathroom.
- **`floors`**: Adds approximately **+$12,000** per additional story.
- **`house_age`**: Depreciates by approximately **-$1,800** per year of age.
- **Location Premiums (relative to baseline `Rural`):**
  - Waterfront: `+$220,000`
  - Downtown: `+$140,000`
  - Uptown: `+$85,000`
  - Suburban: `+$45,000`

---

## 📂 Project Structure

```text
house-price-predictor/
├── app.py              # Main Streamlit web application & inference pipeline
├── generate_data.py    # Script to generate realistic house_data.csv (reproducible)
├── house_data.csv      # Housing dataset (1,500 records)
├── requirements.txt    # Project Python dependencies
├── README.md           # Comprehensive project documentation
└── .gitignore          # Version control ignore rules
```

---

## ⚡ Installation & Local Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12+ installed on your system.
- Git (optional, for cloning).

### 2. Clone the Repository
```bash
git clone https://github.com/gaganpoojary200/House-Price-predictor.git
cd House-Price-predictor
```

### 3. (Optional) Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Application

Launch the Streamlit app with:

```bash
streamlit run app.py
```

Or using python launcher on Windows:
```bash
py -m streamlit run app.py
```

Your default web browser will automatically open:
```text
Local URL: http://localhost:8501
```

---

## 💡 Example Prediction Walkthrough

1. **Select Inputs:**
   - **Area:** `2,200` sq ft
   - **Bedrooms:** `3`
   - **Bathrooms:** `2.5`
   - **Floors:** `2`
   - **House Age:** `8` years
   - **Location:** `Suburban`

2. **Click:** `🔮 Predict House Price`

3. **Output:**
   - **Estimated Valuation:** `$560,000`
   - **Price / Sq Ft:** `$255 / sq ft`
   - **Expected Range (± MAE):** `$540,600` – `$579,400`
   - **Comparison:** `+12.4% vs Market Average`

---

## 📸 Screenshots & UI Showcase

### 1. Main Valuation Interface
> Clean property form with card-based valuation summary, price per square foot, and error tolerance bounds.

### 2. Model Performance & Explainability Tab
> Dynamic bar chart illustrating the dollar value impact of each housing feature, alongside full statistical metrics and mathematical formulation.

### 3. Dataset Explorer Tab
> Data quality table displaying missing value imputation policies, descriptive statistics, and raw sample records.

---

## 🔮 Future Improvements

- [ ] **Polynomial Features & Regularization:** Add Ridge (`L2`) and Lasso (`L1`) regression comparison to explore regularization techniques.
- [ ] **Interactive Scatter Plots:** Add Plotly visualizations displaying regression lines over actual house sale points.
- [ ] **Map Integration:** Add a geographic map view (`pydeck` / Folium) for spatial visualization of neighborhoods.
- [ ] **Export Valuation Report:** Allow users to download a generated PDF valuation summary.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) – free to use for educational, personal, and portfolio purposes.

---

## 👤 Author

Developed by **[Gagan Poojary](https://github.com/gaganpoojary200)**.

