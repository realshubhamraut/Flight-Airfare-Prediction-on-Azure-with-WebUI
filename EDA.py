# %% [markdown]
# ### Libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

# %% [markdown]
# ---
# %% [markdown]
# ### Explore DataSets

# %%
df1 = pd.read_csv("data/train.csv")
print("Shape of df1:", df1.shape)
print(df1.head(5))
print(df1.tail())
print(df1.isnull().sum())
print(df1.info())
print(df1.describe())

# %% [markdown]
# ### EDA

# %%
print("Unique Airlines:", df1["Airline"].unique())
for col in df1:
    print(f"Number of unique {col}s -->> {df1[col].nunique()}")
    
for col in df1:
    print(df1[col].value_counts())

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
print("Top 10 routes:\n", top_routes)
top_routes.plot(kind="bar", color="y", alpha=0.5)
plt.title("Top 10 Used Routes", fontweight="bold")
plt.ylabel("Count")
plt.show()

# %%
bottom_routes = df1["Route"].value_counts().sort_values(ascending=True)[:10]
print("Bottom routes:\n", bottom_routes)

# %%
print("Total_Stops value counts:")
print(df1["Total_Stops"].value_counts())
sns.countplot(x=df1["Total_Stops"], color="brown", alpha=0.6)
plt.show()

# %%
sns.histplot(x=df1["Price"], kde=True, color="black")
plt.show()

# %%
sns.catplot(x="Airline", y="Price", data=df1.sort_values("Price", ascending=False),
            kind="boxen", height=6, aspect=3)
plt.xticks(rotation=90)
plt.show()

# %%
sns.catplot(x="Source", y="Price", data=df1.sort_values("Price", ascending=False),
            kind="boxen", color="black", height=6, aspect=3)
plt.show()

# %%
sns.catplot(x="Destination", y="Price", data=df1.sort_values("Price", ascending=False),
            hue="Total_Stops", aspect=3)
plt.title("Destination with Price", fontweight="bold")
plt.show()

# %%
top_1000 = df1.sort_values("Price", ascending=False).head(1000)
plt.figure(figsize=(18, 10))
plt.bar(top_1000["Route"], top_1000["Price"])
plt.title("Destination with Price", fontweight="bold")
plt.xlabel("Route")
plt.ylabel("Price")
plt.xticks(rotation=90)
plt.show()

# %% [markdown]
# ---
# %% [markdown]
# ### Feature Engineering

# %%
print(df1.info())
print("Unique Date_of_Journey values:", df1["Date_of_Journey"].unique())

# Convert Date_of_Journey and extract day, month and year; then drop the original column.
df1["Date_of_Journey"] = pd.to_datetime(df1["Date_of_Journey"], format="%d/%m/%Y")
df1["Date"] = df1["Date_of_Journey"].dt.day
df1["Month"] = df1["Date_of_Journey"].dt.month
df1["Year"] = df1["Date_of_Journey"].dt.year 
df1.drop("Date_of_Journey", axis=1, inplace=True)

df1["Date"] = df1["Date"].astype(int)
df1["Month"] = df1["Month"].astype(int)
df1["Year"] = df1["Year"].astype(int)

# Handle Total_Stops
print("Before stops processing:", df1["Total_Stops"].unique())
df1["Total_Stops"].replace(np.nan, "1 stop", inplace=True)
df1["Stops"] = df1["Total_Stops"].replace("non-stop", "1 stop")
df1["Stops"] = df1["Stops"].str.split(" ").str[0]
df1["Stops"] = df1["Stops"].astype(int)
df1.drop("Total_Stops", axis=1, inplace=True)
# Rename Stops back to Total_Stops for consistency
df1.rename(columns={"Stops": "Total_Stops"}, inplace=True)
print(df1.head(4))

# Process Arrival_Time
print("Unique Arrival_Time:", df1["Arrival_Time"].unique())
df1["Arrival_Time"] = df1["Arrival_Time"].str.split(" ").str[0]
df1["Arrival_hour"] = df1["Arrival_Time"].str.split(":").str[0]
df1["Arrival_min"] = df1["Arrival_Time"].str.split(":").str[1]
df1.drop("Arrival_Time", axis=1, inplace=True)
df1["Arrival_hour"] = df1["Arrival_hour"].astype(int)
df1["Arrival_min"] = df1["Arrival_min"].astype(int)
print(df1.head(4))

# Process Dep_Time
print("Unique Dep_Time:", df1["Dep_Time"].unique())
df1["Dep_hour"] = df1["Dep_Time"].str.split(":").str[0]
df1["Dep_min"] = df1["Dep_Time"].str.split(":").str[1]
df1.drop("Dep_Time", axis=1, inplace=True)
df1["Dep_hour"] = df1["Dep_hour"].astype(int)
df1["Dep_min"] = df1["Dep_min"].astype(int)
print(df1.head(4))

# Standardize Additional_Info
df1["Additional_Info"] = df1["Additional_Info"].replace("No Info", "No info")

print("Columns:", df1.columns)

# Process Duration into Duration_hour and Duration_min
print("Unique Duration values:", df1["Duration"].unique())
df1["Duration_hour"] = df1["Duration"].str.split(" ").str[0]
df1["Duration_hour"] = df1["Duration_hour"].str.split("h").str[0]
df1["Duration_min"] = df1["Duration"].str.split(" ").str[1]
df1["Duration_min"] = df1["Duration_min"].str.split("m").str[0]
df1.drop("Duration", axis=1, inplace=True)
df1["Duration_min"] = df1["Duration_min"].replace(np.nan, "0")
df1["Duration_hour"] = df1["Duration_hour"].replace("5m", "5")
df1["Duration_min"] = df1["Duration_min"].astype(int)
df1["Duration_hour"] = df1["Duration_hour"].astype(int)
print("Duration_min unique:", df1["Duration_min"].unique())
print("Duration_hour unique:", df1["Duration_hour"].unique())

# Process Route: split into multiple columns and then drop it.
print("Unique Routes:", df1["Route"].unique())
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

print(df1.head(5))
print(df1.info())
print("Null counts:\n", df1.isnull().sum())

# Uncomment the next line if you wish to drop "Year" when training the model.
# df1.drop("Year", axis=1, inplace=True)

# %% [markdown]
# ---
# %% [markdown]
# ### Model Training
# %%
df = pd.read_csv("data/train.csv")
df.drop(columns=["Route"], errors="ignore", inplace=True)
df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y")
df["Date"] = df["Date_of_Journey"].dt.day
df["Month"] = df["Date_of_Journey"].dt.month
df["Year"] = df["Date_of_Journey"].dt.year
df.drop(["Date_of_Journey"], axis=1, inplace=True)

def extract_time(t):
    # Remove any trailing date part (e.g., "04:25 07 Jun") and keep "HH:MM"
    return t.split()[0].strip()

df["Arrival_Time"] = df["Arrival_Time"].apply(extract_time)
df["Arrival_hour"] = df["Arrival_Time"].apply(lambda x: int(x.split(":")[0]))
df["Arrival_min"] = df["Arrival_Time"].apply(lambda x: int(x.split(":")[1]))
df.drop(["Arrival_Time"], axis=1, inplace=True)

if "Dep_Time" in df.columns:
    df.drop(["Dep_Time"], axis=1, inplace=True) 

def parse_duration(d):
    d = d.lower().strip()
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

if "Total_Stops" in df.columns:
    df["Total_Stops"] = df["Total_Stops"].str.lower().fillna("0")
    df["Total_Stops"] = df["Total_Stops"].replace("non-stop", "0 stop")
    df["Total_Stops"] = df["Total_Stops"].apply(lambda s: int(s.split()[0]) if "stop" in s else 0)

# Define the columns to use for training.
# To match your FastAPI prediction, we exclude the "Year" feature.
col_list = [
    "Airline", "Source", "Destination", "Additional_Info",
    "Date", "Month",
    "Total_Stops",
    "Dep_hour", "Dep_min",
    "Arrival_hour", "Arrival_min",
    "Duration_hour", "Duration_min"
]

use_cols = [c for c in col_list if c in df.columns]
X = df[use_cols].copy()
y = df["Price"].copy()

# Encode categorical features using LabelEncoder.
encoders = {}
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].str.strip())
    encoders[col] = le
joblib.dump(encoders, "models/encoders.pkl")
print("Encoders saved successfully (Route excluded).")

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
joblib.dump(x_test, "models/x_test.pkl")
joblib.dump(y_test, "models/y_test.pkl")
print(f"Shape of x_train: {x_train.shape}")
print(f"Shape of x_test: {x_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

# Use Lasso for feature importance.
model_lasso = SelectFromModel(Lasso(alpha=0.005, max_iter=500, random_state=0))
model_lasso.fit(x_train, y_train)
print("Lasso-chosen columns:", x_train.columns[model_lasso.get_support()])

etr = ExtraTreesRegressor()
etr.fit(x_train, y_train)
plt.figure(figsize=(10, 5))
feature_imp = pd.Series(etr.feature_importances_, index=x_train.columns)
feature_imp.nlargest(len(x_train.columns)).plot(kind="bar")
plt.title("ExtraTreesRegressor Feature Importances")
plt.show()

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
# %% [markdown]
# ---