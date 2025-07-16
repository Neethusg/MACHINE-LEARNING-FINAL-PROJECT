# MACHINE-LEARNING-FINAL-PROJECT
ML FINAL PROJECT
# MACHINE LEARNING FINAL PROJECT 

# 🚲 Bike Rental Demand Prediction

## 📌 Project Goals

- Perform exploratory data analysis (EDA) to understand key factors influencing bike demand.
- Preprocess the dataset for machine learning models.
- Train and evaluate multiple regression models.
- Compare model performance and select the best-performing one.

## 📊 Dataset

- **Source**: [UCI Machine Learning Repository – Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

## 🔍 EDA Highlights

- Demand peaks during rush hours (8 AM, 5–6 PM)
- Temperature, season, and working days strongly influence demand
- Weather and humidity negatively impact rentals

##  Data Preprocessing

- Converted categorical variables (e.g., season, weekday) to one-hot encoded features
- Scaled numeric features
- Splitted dataset into training and test sets (80:20)

## 🤖 Machine Learning Models

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

## Model Evaluation

Evaluation Metrics:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score
