# =============================
# IMPORT LIBRARIES
# =============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import math

# =============================
# LOAD DATA
# =============================
df = pd.read_csv("Bengaluru_House_Data.csv")   # FIXED

# =============================
# CLEANING
# =============================
df.drop(columns=["area_type","society","balcony","availability"], inplace=True, errors="ignore")

# Extract BHK
df["bhk"] = df["size"].str.extract(r"(\d+)").astype(float)
df.drop("size", axis=1, inplace=True)

# Convert sqft
df["total_sqft"] = pd.to_numeric(df["total_sqft"], errors="coerce")

# Fill missing (FIXED)
df["bhk"] = df["bhk"].fillna(df["bhk"].median())
df["bath"] = df["bath"].fillna(df["bath"].median())
df["price"] = df["price"].fillna(df["price"].median())

# Drop invalid sqft
df.dropna(subset=["total_sqft"], inplace=True)

# =============================
# REMOVE OUTLIERS
# =============================
df["price_per_sqft"] = df["price"] / df["total_sqft"]
upper = df["price_per_sqft"].quantile(0.95)
df = df[df["price_per_sqft"] <= upper]

# =============================
# LOCATION ENCODING
# =============================
loc_counts = df["location"].value_counts()
valid_locs = loc_counts[loc_counts >= 10].index
df = df[df["location"].isin(valid_locs)]

df = pd.get_dummies(df, columns=["location"], drop_first=True)

# =============================
# TRAIN DATA
# =============================
X = df.drop(["price","price_per_sqft"], axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================
# MODELS
# =============================
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# =============================
# EVALUATION FUNCTION
# =============================
def evaluate(name, model):
    pred = model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    print(f"\n{name}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 : {r2:.4f}")
    print(f"MAE: {mae:.2f}")
    print("-"*30)

evaluate("Linear Regression", lr)
evaluate("Random Forest", rf)

# =============================
# SAVE BEST MODEL (RF)
# =============================
with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)

with open("feature_columns.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("Model saved successfully")