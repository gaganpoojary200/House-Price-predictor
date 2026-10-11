"""
Script to generate a realistic housing dataset for the House Price Predictor project.
Produces house_data.csv with 1,500 representative housing records.
"""

import numpy as np
import pandas as pd

def generate_housing_data(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)

    # Locations and distributions
    locations = ["Downtown", "Suburban", "Uptown", "Waterfront", "Rural"]
    loc_probs = [0.25, 0.35, 0.20, 0.10, 0.10]
    
    # Location premiums in USD
    loc_premium_map = {
        "Rural": 0,
        "Suburban": 45000,
        "Uptown": 85000,
        "Downtown": 140000,
        "Waterfront": 220000,
    }

    # Generate synthetic features
    location_choices = np.random.choice(locations, size=n_samples, p=loc_probs)
    
    # Area between 650 sq ft and 4,800 sq ft (log-normal distribution skewed towards 1,800)
    area = np.clip(np.random.normal(loc=2100, scale=750, size=n_samples), 650, 4800).round(-1)
    
    # Bedrooms correlated with area
    bedrooms = np.clip(np.round(1 + (area / 800) + np.random.normal(0, 0.6, n_samples)), 1, 6).astype(int)
    
    # Bathrooms correlated with bedrooms and area
    bathrooms = np.clip(np.round(bedrooms * 0.7 + (area / 1500) + np.random.normal(0, 0.4, n_samples)), 1, 5).astype(int)
    
    # Floors (1 to 3)
    floors = np.random.choice([1, 2, 3], size=n_samples, p=[0.45, 0.45, 0.10])
    
    # House age (0 to 50 years)
    house_age = np.clip(np.random.exponential(scale=15, size=n_samples), 0, 50).round().astype(int)

    # Calculate realistic market price formula (Linear relationship + noise)
    # Base price: $65,000
    # Price per sq ft: $165
    # Value per bedroom: $18,000
    # Value per bathroom: $22,000
    # Value per floor: $12,000
    # Depreciation per year of age: -$1,800
    # Location premium
    # Realistic noise: std dev $25,000
    loc_premiums = np.array([loc_premium_map[loc] for loc in location_choices])
    
    base_price = 65000
    price = (
        base_price
        + area * 165.0
        + bedrooms * 18000.0
        + bathrooms * 22000.0
        + floors * 12000.0
        - house_age * 1800.0
        + loc_premiums
        + np.random.normal(0, 25000, n_samples)
    )
    
    # Ensure minimum reasonable price (e.g. at least $75,000)
    price = np.clip(price, 75000, 1800000).round(-2)

    df = pd.DataFrame({
        "area": area.astype(int),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "floors": floors,
        "house_age": house_age,
        "location": location_choices,
        "price": price.astype(int)
    })

    # Introduce a small number of realistic missing values (1.5%) to demonstrate robust preprocessing
    missing_age_indices = np.random.choice(n_samples, size=int(n_samples * 0.015), replace=False)
    missing_bath_indices = np.random.choice(n_samples, size=int(n_samples * 0.010), replace=False)
    
    df.loc[missing_age_indices, "house_age"] = np.nan
    df.loc[missing_bath_indices, "bathrooms"] = np.nan

    return df

if __name__ == "__main__":
    df = generate_housing_data(1500)
    df.to_csv("house_data.csv", index=False)
    print(f"Dataset generated successfully with {len(df)} rows.")
    print("Columns:", list(df.columns))
    print("Sample:\n", df.head(3))
