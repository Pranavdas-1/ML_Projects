# Flight Price Prediction using Machine Learning

## Overview

This project predicts airline ticket prices using Machine Learning techniques and historical flight booking data. The goal is to identify the factors that influence flight prices and build a predictive model capable of estimating ticket costs accurately.

## Dataset

The dataset contains information about:

* Airline
* Source City
* Destination City
* Departure Time
* Arrival Time
* Number of Stops
* Travel Class
* Flight Duration
* Days Left Before Departure
* Ticket Price (Target Variable)

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Jupyter Notebook

## Data Preprocessing

* Removed unnecessary columns
* Encoded categorical variables using One-Hot Encoding
* Converted travel class into numerical format
* Factorized stop information
* Prepared data for machine learning models

## Feature Engineering

Created machine-learning-ready features from:

* Airline information
* Source and destination cities
* Departure and arrival timings
* Travel class
* Number of stops
* Flight duration
* Days remaining before departure

## Model Used

### Random Forest Regressor

The Random Forest model was trained to predict flight ticket prices based on travel-related features.

## Hyperparameter Tuning

Model performance was improved using:

* RandomizedSearchCV
* Cross Validation

## Results

| Metric   | Value   |
| -------- | ------- |
| R² Score | 0.982   |
| MAE      | 1456.92 |
| RMSE     | 3030.76 |

## Feature Importance

The most influential features affecting ticket prices were:

1. Travel Class
2. Flight Duration
3. Days Left Before Departure
4. Airline
5. Source and Destination Cities

## Visualizations

The project includes:

* Actual vs Predicted Price Scatter Plot
* Feature Importance Analysis
* Data Exploration and Insights

## Learning Outcomes

Through this project, I gained hands-on experience in:

* Data Preprocessing
* Feature Engineering
* Regression Analysis
* Random Forest Regression
* Hyperparameter Tuning
* Model Evaluation
* Machine Learning Pipelines

## Future Improvements

* Deploy the model using Streamlit
* Experiment with XGBoost and LightGBM
* Build a web interface for real-time predictions
