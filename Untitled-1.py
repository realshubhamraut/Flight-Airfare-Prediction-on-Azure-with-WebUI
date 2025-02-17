# %%
# ### Libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# %%
from sklearn.linear_model import LinearRegression
# Lasso
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesRegressor
# Ridge
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
# DecisionTree
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
# RandomForestRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings("ignore")

# %%
# # Explore DataSets

# %%
df1 = pd.read_csv("data/train.csv")
# Size Of Dataset
df1.shape

# %%
df1.head(5)

# %%
df1.tail()

# %%
# **Missing Values**

# %%
df1.isnull().sum()

# %%
df1.info()

# %%
df1.describe()

# %%
# ### Data Analysis And Visualization on train.csv

# %%
df1["Airline"].unique()

# %%
for i in df1:
    print(f"Number of unique {i}s -->> {df1[i].nunique()}")

# %%
for i in df1:
    print(df1[i].value_counts())

# %%
sns.countplot(x="Airline", data=df1)

plt.title("Airline Distribution", fontweight="bold")
plt.xlabel("Airline")
plt.ylabel("Count")

plt.xticks(rotation=45)
plt.show()

# %%
sns.countplot(x="Source", data=df1, color="skyblue")

plt.title("Source Distribution", fontweight="bold")
plt.xlabel("Count")
plt.ylabel("Source")
plt.xticks(rotation=45)

plt.show()

# %%
sns.countplot(x=df1["Destination"], color="orange", alpha=0.5)

plt.title("Destination Distribution", fontweight="bold")
plt.xlabel("Count")
plt.ylabel("Destination")
plt.xticks(rotation=45)
plt.show()

# %%
top_routes = df1["Route"].value_counts().sort_values(ascending=False)[:10]
top_routes

# %%
top_routes.plot(kind="bar", color="y", alpha=0.5)

plt.title("Top 10 Used Routes", fontweight="bold")
plt.ylabel("Count")

plt.show()

# %%
# Less used routes
bottom_routes = df1["Route"].value_counts().sort_values(ascending=True)[:10]
bottom_routes

# %%
df1["Total_Stops"].value_counts()

# %%
sns.countplot(x=df1["Total_Stops"], color="brown", alpha=0.6)
plt.show()

# %%
sns.histplot(x=df1["Price"], kde=True, color="black")

plt.show()

# %%
sns.catplot(x="Airline", y="Price", data=df1.sort_values("Price", ascending=False), kind="boxen", height=6, aspect=3)

plt.xticks(rotation=90)

plt.show()

# %%
sns.catplot(x="Source", y="Price", data=df1.sort_values("Price", ascending=False), kind="boxen", color="black", height=6, aspect=3)

plt.show()

# %%
sns.catplot(x="Destination", y="Price", data=df1.sort_values("Price", ascending=False), hue="Total_Stops", aspect=3)

plt.title("Destination with Price", fontweight="bold")

plt.show()

# %%
# Most priced routes
top_1000 = df1.sort_values("Price", ascending=False).head(1000)

plt.figure(figsize=(18, 10))
plt.bar(top_1000["Route"], top_1000["Price"])

plt.title("Destination with Price", fontweight="bold")
plt.xlabel("Route")
plt.ylabel("Price")

plt.xticks(rotation=90)

plt.show()

# %%
# ---

# %%
# ## Feature Engineering

# %%
df1.info()

# %%
df1["Date_of_Journey"].unique()

# %%
# Create three features from "Date_of_Journey" and drop it
df1["Date"] = df1["Date_of_Journey"].str.split("/").str[0]
df1["Month"] = df1["Date_of_Journey"].str.split("/").str[1]
df1["Year"] = df1["Date_of_Journey"].str.split("/").str[2]

df1.drop("Date_of_Journey", inplace=True, axis=1)

df1["Date"] = df1["Date"].astype(int)
df1["Month"] = df1["Month"].astype(int)
df1["Year"] = df1["Year"].astype(int)

# %%
# Handle "Total_Stops"
df1["Total_Stops"].unique()

# %%
df1["Total_Stops"].replace(np.nan, "1 stop", inplace=True)
df1["Stops"] = df1["Total_Stops"].replace("non-stop", "1 stop")
df1["Stops"] = df1["Stops"].str.split(" ").str[0]
df1["Stops"] = df1["Stops"].astype(int)
df1.drop("Total_Stops", axis=1, inplace=True)

# %%
df1.head(4)

# %%
df1["Arrival_Time"].unique()

# %%
# Remove date from "Arrival_Time"
df1["Arrival_Time"] = df1["Arrival_Time"].str.split(" ").str[0]

# %%
# Create "Arrival_hour" and "Arrival_min"
df1["Arrival_hour"] = df1["Arrival_Time"].str.split(":").str[0]
df1["Arrival_min"] = df1["Arrival_Time"].str.split(":").str[1]
df1.drop("Arrival_Time", axis=1, inplace=True)

# %%
df1["Arrival_hour"] = df1["Arrival_hour"].astype(int)
df1["Arrival_min"] = df1["Arrival_min"].astype(int)

# %%
df1.head(4)

# %%
df1["Dep_Time"].unique()

# %%
# Create "Dep_hour" and "Dep_min"
df1["Dep_hour"] = df1["Dep_Time"].str.split(":").str[0]
df1["Dep_min"] = df1["Dep_Time"].str.split(":").str[1]
df1.drop("Dep_Time", axis=1, inplace=True)
df1["Dep_hour"] = df1["Dep_hour"].astype(int)
df1["Dep_min"] = df1["Dep_min"].astype(int)

# %%
df1.head(4)

# %%
df1["Additional_Info"].unique()

# %%
df1["Additional_Info"] = df1["Additional_Info"].replace("No Info", "No info")

# %%
print(df1.columns)

# %%
df1["Duration"].unique()

# %%
# Create "Duration_hour" and "Duration_min"
df1["Duration_hour"] = df1["Duration"].str.split(" ").str[0]
df1["Duration_hour"] = df1["Duration_hour"].str.split("h").str[0]

df1["Duration_min"] = df1["Duration"].str.split(" ").str[1]
df1["Duration_min"] = df1["Duration_min"].str.split("m").str[0]

df1.drop("Duration", axis=1, inplace=True)

# %%
df1["Duration_min"].unique()

# %%
df1["Duration_hour"].unique()

# %%
df1["Duration_min"] = df1["Duration_min"].replace(np.nan, "0")
df1["Duration_hour"] = df1["Duration_hour"].replace("5m", "5")

# Convert to int
df1["Duration_min"] = df1["Duration_min"].astype(int)
df1["Duration_hour"] = df1["Duration_hour"].astype(int)

# %%
df1["Route"].unique()

# %%
# Split "Route" into multiple columns
df1["Route_1"] = df1["Route"].str.split(" → ").str[0]
df1["Route_2"] = df1["Route"].str.split(" → ").str[1]
df1["Route_3"] = df1["Route"].str.split(" → ").str[2]
df1["Route_4"] = df1["Route"].str.split(" → ").str[3]
df1["Route_5"] = df1["Route"].str.split(" → ").str[4]

df1["Route_1"].fillna("None", inplace=True)
df1["Route_2"].fillna("None", inplace=True)
df1["Route_3"].fillna("None", inplace=True)
df1["Route_4"].fillna("None", inplace=True)
df1["Route_5"].fillna("None", inplace=True)

df1.drop("Route", axis=1, inplace=True)

# %%
df1.head(5)

# %%
df1.info()

# %%
df1.isnull().sum()

# %%
# No Null value in dataset

# %%
df1.head(5)

# %%
# Remove Year column because this data is from the same year

# %%
df1.drop("Year", axis=1, inplace=True)

# %%
# ---

# %%
# ### Model Training
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# =========================
# 1) Load dataset
# =========================
df = pd.read_csv("data/train.csv")

# Drop "Route" entirely if present
df.drop(columns=["Route"], errors="ignore", inplace=True)

# =========================
# 2) Parse Date_of_Journey
# =========================
df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y")
df["Date"] = df["Date_of_Journey"].dt.day
df["Month"] = df["Date_of_Journey"].dt.month

# We no longer need the original Date_of_Journey
df.drop(["Date_of_Journey"], axis=1, inplace=True)

# =========================
# 3) Parse Arrival_Time => Arrival_hour, Arrival_min
# =========================
def extract_time(t):
    # Remove any trailing date part (e.g., "04:25 07 Jun") and keep "HH:MM"
    return t.split()[0].strip()

df["Arrival_Time"] = df["Arrival_Time"].apply(extract_time)
df["Arrival_hour"] = df["Arrival_Time"].apply(lambda x: int(x.split(":")[0]))
df["Arrival_min"] = df["Arrival_Time"].apply(lambda x: int(x.split(":")[1]))
df.drop(["Arrival_Time"], axis=1, inplace=True)

# =========================
# 4) Parse Dep_Time => Dep_hour, Dep_min
# =========================
df["Dep_Time"] = df["Dep_Time"].apply(extract_time)
df["Dep_hour"] = df["Dep_Time"].apply(lambda x: int(x.split(":")[0]))
df["Dep_min"] = df["Dep_Time"].apply(lambda x: int(x.split(":")[1]))
df.drop(["Dep_Time"], axis=1, inplace=True)

# =========================
# 5) Parse Duration => Duration_hour, Duration_min
# =========================
def parse_duration(d):
    d = d.lower().strip()
    # Possible patterns: "10h 55m", "2h", "50m", etc.
    hours = 0
    mins = 0
    if "h" in d:
        parts = d.split("h")
        hours = int(parts[0])
        if "m" in parts[1]:
            mins = int(parts[1].replace("m", "").strip()) if parts[1] else 0
    elif "m" in d:
        mins = int(d.replace("m", "").strip())
    return hours, mins

df["Duration_hour"], df["Duration_min"] = zip(*df["Duration"].apply(parse_duration))
df.drop(["Duration"], axis=1, inplace=True)

# =========================
# 6) Convert Total_Stops => 0 if non-stop, else numeric
# =========================
if "Total_Stops" in df.columns:
    df["Total_Stops"] = df["Total_Stops"].str.lower().fillna("0")
    df["Total_Stops"] = df["Total_Stops"].replace("non-stop", "0 stop")
    # e.g. "1 stop" => 1
    df["Total_Stops"] = df["Total_Stops"].apply(lambda s: int(s.split()[0]) if "stop" in s else 0)

# =========================
# 7) Split into X and y
# =========================
# The columns your Streamlit app expects:
col_list = [
    "Airline", "Source", "Destination", "Additional_Info",
    "Date", "Month", "Total_Stops",
    "Dep_hour", "Dep_min",
    "Arrival_hour", "Arrival_min",
    "Duration_hour", "Duration_min"
]

# Make sure each column exists
use_cols = [c for c in col_list if c in df.columns]
X = df[use_cols].copy()
y = df["Price"].copy()

# =========================
# 8) Label encode categorical features
# =========================
encoders = {}
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Save encoders
joblib.dump(encoders, "models/encoders.pkl")
print("Encoders saved successfully (Route excluded).")

# =========================
# 9) Split data
# =========================
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Also save test data if you wish
joblib.dump(x_test, "models/x_test.pkl")
joblib.dump(y_test, "models/y_test.pkl")

print(f"Shape of x_train: {x_train.shape}")
print(f"Shape of x_test: {x_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

# ---------------------------------------------------
# Optional: Lasso for feature importance
# ---------------------------------------------------
model_lasso = SelectFromModel(Lasso(alpha=0.005, max_iter=500, random_state=0))
model_lasso.fit(x_train, y_train)
print("Lasso-chosen columns:", x_train.columns[model_lasso.get_support()])

# ---------------------------------------------------
# ExtraTreesRegressor for feature importance
# ---------------------------------------------------
etr = ExtraTreesRegressor()
etr.fit(x_train, y_train)

plt.figure(figsize=(10, 5))
feature_imp = pd.Series(etr.feature_importances_, index=x_train.columns)
feature_imp.nlargest(len(x_train.columns)).plot(kind="bar")
plt.title("ExtraTreesRegressor Feature Importances")
plt.show()

# ---------------------------------------------------
# Model comparisons
# ---------------------------------------------------
def evaluate_model(clf, x_tr, y_tr, x_te, y_te):
    clf.fit(x_tr, y_tr)
    preds = clf.predict(x_te)
    r2 = r2_score(y_te, preds)
    mae = mean_absolute_error(y_te, preds)
    mse = mean_squared_error(y_te, preds)
    rmse = np.sqrt(mse)
    return r2, mae, mse, rmse

models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "SVR": SVR()
}

results = {}
for name, reg in models.items():
    r2, mae, mse, rmse = evaluate_model(reg, x_train, y_train, x_test, y_test)
    results[name] = {"R2": r2, "MAE": mae, "MSE": mse, "RMSE": rmse}

results_df = pd.DataFrame.from_dict(results, orient="index")
print("\nComparison of models:\n", results_df)

best_model_name = max(results, key=lambda nm: results[nm]["R2"])
best_model = models[best_model_name]
best_model.fit(x_train, y_train)
joblib.dump(best_model, "models/best_model.pkl")
print(f"\n'{best_model_name}' is the best model, saved to 'models/best_model.pkl'.")

