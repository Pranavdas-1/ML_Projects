# California Housing Price Prediction

A machine learning project that predicts median house values in California using regression models, built with scikit-learn.

## Overview

This project uses the California Housing dataset to predict `median_house_value` based on features like income, location, and housing characteristics. It covers the full ML pipeline — from data cleaning to model tuning with GridSearchCV.

## Dataset

The dataset includes:
- `longitude`, `latitude` — geographic location
- `housing_median_age` — age of houses in the block
- `total_rooms`, `total_bedrooms` — housing density
- `population`, `households` — area demographics
- `median_income` — strongest predictor of house value
- `ocean_proximity` — categorical (INLAND, NEAR BAY, NEAR OCEAN, ISLAND, <1H OCEAN)
- `median_house_value` — target variable

## Workflow

1. **Data Cleaning**
   - Dropped missing values with `dropna()`

2. **Exploratory Data Analysis**
   - Histograms to inspect feature distributions
   - Correlation heatmap to identify key predictors
   - Scatterplot of geographic location colored by house value

3. **Data Preprocessing**
   - Log transformation on skewed features (`total_rooms`, `total_bedrooms`, `population`, `households`)
   - One-hot encoding of `ocean_proximity` using `pd.get_dummies()`
   - Feature engineering: `bedroom_ratio`, `household_rooms`

4. **Feature Scaling**
   - `StandardScaler` — fit on training data, applied to both train and test sets

5. **Modeling**
   - Baseline: `LinearRegression`
   - Improved: `RandomForestRegressor`
   - Hyperparameter tuning via `GridSearchCV` with 5-fold cross-validation

## Hyperparameters Tuned

| Parameter | Values Tested |
|-----------|--------------|
| `n_estimators` | 100, 200, 250 |
| `min_samples_split` | 2, 4 |
| `max_depth` | None, 4, 8 |

Scoring metric: `neg_mean_squared_error`

## Results

| Model | R² Score |
|-------|----------|
| Linear Regression | 0.670 |
| Random Forest (default) | 0.819 |
| Random Forest (GridSearchCV tuned) | 0.820 |

## Tech Stack

- Python
- pandas, numpy
- scikit-learn
- seaborn, matplotlib
- Jupyter Lab

## Key Learnings

- Log transformation helps normalize right-skewed features
- One-hot encoding avoids false ordinal relationships in categorical data
- Feature engineering (ratios) improved correlation with the target
- Random Forest significantly outperformed Linear Regression on this dataset
- GridSearchCV with cross-validation gave only a marginal improvement over the default Random Forest, suggesting the default parameters were already close to optimal for this data
