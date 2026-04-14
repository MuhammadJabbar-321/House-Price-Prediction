import pandas as pd
import numpy as np
import joblib
import tkinter as tk
from tkinter import messagebox

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("house_data.csv")
df = df.dropna()

X = df.drop("Price", axis=1)
y = df["Price"]

# -----------------------------
# DETECT COLUMNS (FIXED FOR FUTURE PANDAS)
# -----------------------------
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()

print("Numeric columns:", numeric_features)
print("Categorical columns:", categorical_features)

# -----------------------------
# PREPROCESSOR
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# -----------------------------
# MODEL PIPELINE
# -----------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# -----------------------------
# TRAIN / TEST SPLIT (IMPORTANT FIX)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

print("Model trained successfully!")

# -----------------------------
# SAVE MODEL (OPTIONAL BUT GOOD)
# -----------------------------
joblib.dump(model, "house_model.pkl")

# -----------------------------
# SIMPLE SET EXAMPLE FIX (YOUR ERROR FIXED)
# -----------------------------
numse = {7}
print(numse)

print(numse.union({1, 2, 3, 4}))   # correct
numse.update({1, 2, 3, 4})         # modifies original
print(numse)

# -----------------------------
# GUI APP
# -----------------------------
root = tk.Tk()
root.title("House Price Predictor")
root.geometry("400x450")

entries = {}

# -----------------------------
# INPUT FIELDS
# -----------------------------
for col in X.columns:
    tk.Label(root, text=col).pack()
    entry = tk.Entry(root)
    entry.pack()
    entries[col] = entry

# -----------------------------
# PREDICT FUNCTION
# -----------------------------
def predict_price():
    try:
        input_data = {}

        for col in X.columns:
            val = entries[col].get()

            if val == "":
                raise ValueError(f"{col} cannot be empty")

            if col in numeric_features:
                input_data[col] = float(val)
            else:
                input_data[col] = val

        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)[0]

        messagebox.showinfo("Prediction", f"Estimated Price: {prediction:,.2f}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# -----------------------------
# BUTTON
# -----------------------------
tk.Button(root, text="Predict Price", command=predict_price).pack(pady=20)

# -----------------------------
# RUN APP
# -----------------------------
root.mainloop()