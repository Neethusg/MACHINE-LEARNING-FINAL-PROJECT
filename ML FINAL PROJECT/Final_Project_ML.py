
** MACHINE LAERNING PROJECT**

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from google.colab import files
uploaded = files.upload()
df = pd.read_csv('Bike Rental.csv')

df.info()

df.head(6)

df.tail(10)

print(df.shape)

print(df.isnull().sum()) # To find the null value

print(df.describe())

print(df['yr'].value_counts().head(10))

print(df['weekday'].value_counts())

print(df.columns)

print(df.columns.tolist()) # To see as a list

# Check unique values of categorical features
categorical_features = ['season', 'weathersit', 'mnth', 'hr', 'weekday', 'holiday', 'workingday', 'yr']
for col in categorical_features:
    print(f"\nUnique values in '{col}': {df[col].unique()}")

# Plot distribution of total bike rentals
plt.figure(figsize=(8,5))
sns.histplot(df['cnt'], bins=50, kde=True)
plt.title('Distribution of Total Bike Rentals (cnt)')
plt.xlabel('Count')
plt.ylabel('Frequency')
plt.show()

# Distribution of registered and casual users
plt.figure(figsize=(12,5))
sns.histplot(df['registered'], bins=50, color='green', label='Registered Users', kde=True)
sns.histplot(df['casual'], bins=50, color='orange', label='Casual Users', kde=True)
plt.title('Distribution of Registered and Casual Users')
plt.xlabel('Count')
plt.legend()
plt.show()

# Distribution of weather features
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
sns.histplot(df['temp'], bins=30, kde=True, color='skyblue')

plt.title('Temperature Distribution')
plt.subplot(1,3,2)
sns.histplot(df['hum'], bins=30, kde=True, color='lightgreen')

plt.title('Humidity Distribution')
plt.subplot(1,3,3)
sns.histplot(df['windspeed'], bins=30, kde=True, color='lightcoral')
plt.title('Windspeed Distribution')
plt.tight_layout()
plt.show()

# Relationship between hour of day and bike rentals
plt.figure(figsize=(10,5))
sns.lineplot(x='hr', y='cnt', data=df, ci=None)
plt.title('Average Bike Rentals by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Average Count')
plt.show()

# Bike rentals by season
plt.figure(figsize=(8,5))
sns.boxplot(x='season', y='cnt', data=df)
plt.title('Bike Rentals by Season')
plt.xlabel('Season')
plt.ylabel('Count')
plt.show()

# Bike rentals by weather situation
plt.figure(figsize=(8,5))
sns.boxplot(x='weathersit', y='cnt', data=df)
plt.title('Bike Rentals by Weather Situation')
plt.xlabel('Weather Situation')
plt.ylabel('Count')
plt.show()

# Bike rentals on working day vs non-working day
plt.figure(figsize=(6,5))
sns.boxplot(x='workingday', y='cnt', data=df)
plt.title('Bike Rentals: Working Day vs Non-Working Day')
plt.xlabel('Working Day (0=No, 1=Yes)')
plt.ylabel('Count')
plt.show()

# Correlation heatmap of numeric features
plt.figure(figsize=(12,8))
corr = df[['temp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Rentals by day of week
plt.figure(figsize=(8,5))
sns.boxplot(x='weekday', y='cnt', data=df)
plt.title('Bike Rentals by Day of Week')
plt.xlabel('Weekday (0=Sunday)')
plt.ylabel('Count')
plt.show()

# Rentals by month
plt.figure(figsize=(8,5))
sns.boxplot(x='mnth', y='cnt', data=df)
plt.title('Bike Rentals by Month')
plt.xlabel('Month')
plt.ylabel('Count')
plt.show()

# Feature Engineering

categorical_cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
df[categorical_cols] = df[categorical_cols].astype('category')

df = pd.get_dummies(df, columns=['season', 'weathersit'], drop_first=True)

print(df.head())

# Outlier Detection and Normalization

numerical_cols = ['temp', 'atemp', 'hum', 'windspeed', 'cnt']

# Z-score method
z_scores = np.abs(stats.zscore(df[numerical_cols]))
outliers = (z_scores > 3).any(axis=1)
print(f"Number of outliers: {outliers.sum()}")

import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df['temp'])
plt.show()

# Calculate Z-scores
z_scores = np.abs(stats.zscore(df[numerical_cols]))

z_scores

# keeping rows where all z-scores ≤ 3
filtered_entries = (z_scores < 3).all(axis=1)

# Removing outliers
df_clean = df[filtered_entries].copy()

print(f"Original shape: {df.shape}")

print(f"after removing outliers: {df_clean.shape}")

df_clean = df[~outliers]

# Scaling
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print(df_clean[numerical_cols].describe())

"""# MODEL TRAINING"""

# 1. LINEAR REGRESSION

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

print(df.columns)

y = df['cnt']

df = df.drop(columns=['dteday'])

X = df.drop(columns=['cnt', 'casual', 'registered'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training features shape:", X_train.shape)

print("Testing features shape:", X_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE:', rmse)
print('R²:', r2_score(y_test, y_pred))

plt.figure(figsize=(10, 4))
plt.plot(y_test.values, label="Actual", marker='o')
plt.plot(y_pred, label="Predicted", marker='x')
plt.title("Actual vs Predicted cnt (Test Set)")
plt.xlabel("Sample Index")
plt.ylabel("cnt")
plt.legend()
plt.grid(True)
plt.show()

# Dicision tree regressior

# Initialize the Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(random_state=42)

dt_regressor = DecisionTreeRegressor(
    max_depth=10,             # limit the depth of the tree
    min_samples_split=5,      # minimum samples to split a node
    min_samples_leaf=3,       # minimum samples per leaf
    random_state=42
)

# Train the model
dt_regressor.fit(X_train, y_train)

# Predict on test data
y_pred = dt_regressor.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")

print(f"R-squared (R2): {r2:.2f}")

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

from sklearn.tree import plot_tree

plt.figure(figsize=(15,8))
plot_tree(
    dt_regressor,
    feature_names=X_train.columns,
    filled=True,
    rounded=True,
    fontsize=8,
    max_depth=3
)
plt.show()

# Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=42, n_estimators=100)

# Train the model
rf_regressor.fit(X_train, y_train)

# Predict on test data
y_pred = rf_regressor.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Random Forest Regressor Mean Squared Error: {mse:.2f}")

print(f"Random Forest Regressor R-squared: {r2:.2f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("Actual Count")
plt.ylabel("Predicted Count")
plt.title("Random Forest: Actual vs Predicted Bike Demand")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.tight_layout()
plt.show()

"""# MODEL EVALUATION"""

#1. Mean Squared Error (MSE)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")

#2. Root Mean Squared Error (RMSE)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

#3. Mean Absolute Error (MAE)

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

#4. R-squared (R²)

r2 = r2_score(y_test, y_pred)
print(f"R-squared (R²): {r2:.2f}")

"""# MODEL COMPARISON"""

# Models to compare
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
}

from sklearn.pipeline import Pipeline

results = []

for name, model in models.items():
    pipeline = Pipeline([
        ('regressor', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    })

results_df = pd.DataFrame(results).sort_values(by='RMSE')
print(results_df)

"""# CONCLUSION

The Random Forest Regressor (or Gradient Boosting) is recommended for predicting hourly bike rental demand due to its strong performance in RMSE, MAE, and R². These models can support operational decisions, such as allocating bikes to stations based on demand patterns.
"""

