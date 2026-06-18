# California Housing Price Prediction

A machine learning project that predicts median house values in California using regression models, built with scikit-learn.

## Overview

This project uses the California Housing dataset to predict `median_house_value` based on features like income, location, and housing characteristics. It covers the full ML pipeline — from data cleaning to model tuning.

## Dataset

The dataset includes:
- `longitude`, `latitude` — geographic location
- `housing_median_age` — age of houses in the block
- `total_rooms`, `total_bedrooms` — housing density
- `population`, `households` — area demographics
- `median_income` — strongest predictor of house value
- `ocean_proximity` — categorical (INLAND, NEAR BAY, NEAR OCEAN, etc.)
- `median_house_value` — target variable

## Workflow

1. **Exploratory Data Analysis**
   - Correlation heatmap to identify key predictors
   - Histograms to inspect feature distributions

2. **Data Preprocessing**
   - Log transformation on skewed features (`total_rooms`, `total_bedrooms`, `population`, `households`)
   - One-hot encoding of `ocean_proximity` using `pd.get_dummies()`
   - Feature engineering: `bedroom_ratio`, `household_rooms`

3. **Feature Scaling**
   - `StandardScaler` to normalize feature ranges

4. **Modeling**
   - Baseline: `LinearRegression`
   - Improved: `RandomForestRegressor`
   - Hyperparameter tuning via `GridSearchCV` with 5-fold cross-validation

## Hyperparameters Tuned

| Parameter | Values Tested |
|-----------|--------------|
| `n_estimators` | 100, 200 |
| `max_depth` | 5, 10, None |
| `min_samples_split` | 2, 5, 10 |
| `max_features` | sqrt, log2, None |

## Results

| Model | R² Score |
|-------|----------|
| Linear Regression | ~0.65 |
| Random Forest (tuned) | ~0.82 |

## Tech Stack

- Python
- pandas, numpy
- scikit-learn
- seaborn, matplotlib
- Jupyter Lab

## Key Learnings

- Log transformation helps normalize right-skewed features
- One-hot encoding avoids false ordinal relationships in categorical data
- Feature engineering (ratios) can outperform raw counts
- GridSearchCV with cross-validation prevents overfitting to a single train/test split
