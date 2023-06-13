import pandas as pd
import numpy as np
import xgboost

davitt_road = pd.read_csv('../data/davitt_road/davitt_road_2022.csv')

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

std_scl = StandardScaler()
data = davitt_road[['month', 'day', 'hour']]

features = std_scl.fit_transform(data)
target = davitt_road[['pm10', 'pm2.5']]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create and fit the Random Forest regressor
random_forest = RandomForestRegressor(random_state=42)
random_forest.fit(X_train, y_train)

# Predict the target variables for the test set
rf_pred = random_forest.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, rf_pred)
mae = mean_absolute_error(y_test, rf_pred)

print("Random Forest MSE:", mse)
print("Random Forest MAE:", mae)

print("---------------------------------")
xgb_regressor = xgboost.XGBRegressor()
xgb_regressor.fit(X_train, y_train)

xgb_pred = xgb_regressor.predict(X_test)

mse = mean_squared_error(y_test, xgb_pred)
mae = mean_absolute_error(y_test, xgb_pred)

print("XGBoost MSE:", mse)
print("XGBoost MAE:", mae)



