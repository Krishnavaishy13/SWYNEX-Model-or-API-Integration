import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Sample historical daily vegetable sales & wholesale dataset
data = [
    # Item, Day, Weekend_Festival, Wholesale_INR, Prev_Sold_Kg, Spoilage_Kg, Weather, Action
    ["Tomato", "Monday", 0, 28, 45, 2, "Clear", "Maintain"],
    ["Tomato", "Saturday", 1, 32, 72, 1, "Clear", "High_Restock"],
    ["Tomato", "Wednesday", 0, 45, 30, 8, "Rainy", "Reduce"],
    ["Onion", "Sunday", 1, 24, 90, 0, "Clear", "High_Restock"],
    ["Onion", "Tuesday", 0, 22, 55, 2, "Clear", "Maintain"],
    ["Onion", "Thursday", 0, 38, 40, 6, "Rainy", "Reduce"],
    ["Potato", "Friday", 0, 20, 65, 1, "Clear", "Maintain"],
    ["Potato", "Saturday", 1, 21, 95, 0, "Clear", "High_Restock"],
    ["Potato", "Monday", 0, 30, 42, 5, "Extreme Heat", "Reduce"],
    ["Spinach", "Sunday", 1, 15, 38, 1, "Clear", "High_Restock"],
    ["Spinach", "Wednesday", 0, 18, 22, 6, "Extreme Heat", "Reduce"],
    ["Spinach", "Friday", 0, 14, 28, 2, "Clear", "Maintain"],
    ["Tomato", "Sunday", 1, 30, 78, 1, "Clear", "High_Restock"],
    ["Onion", "Saturday", 1, 25, 88, 1, "Clear", "High_Restock"],
    ["Spinach", "Tuesday", 0, 20, 18, 7, "Rainy", "Reduce"],
    ["Potato", "Wednesday", 0, 19, 60, 1, "Clear", "Maintain"],
    ["Tomato", "Thursday", 0, 42, 34, 7, "Extreme Heat", "Reduce"],
    ["Onion", "Friday", 0, 23, 62, 2, "Clear", "Maintain"],
]

columns = [
    "Item_Name",
    "Day_of_Week",
    "Is_Weekend_or_Festival",
    "Wholesale_Price_Per_Kg",
    "Previous_Day_Sold_Kg",
    "Unsold_Spoilage_Kg",
    "Weather_Condition",
    "Procurement_Action",
]

df = pd.DataFrame(data, columns=columns)

X = df.drop("Procurement_Action", axis=1)
y = df["Procurement_Action"]

# 2. Preprocessing & Random Forest Pipeline
categorical_cols = ["Item_Name", "Day_of_Week", "Weather_Condition"]
numeric_cols = [
    "Is_Weekend_or_Festival",
    "Wholesale_Price_Per_Kg",
    "Previous_Day_Sold_Kg",
    "Unsold_Spoilage_Kg",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]
)

# 3. Train/Test Split & Evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

macro_f1 = f1_score(y_test, y_pred, average="macro")
print("=== MODEL EVALUATION ===")
print(f"Macro F1-Score: {macro_f1:.2f}\n")
print(classification_report(y_test, y_pred, zero_division=0))

# 4. Prototype Inference on New Morning Mandi Inputs
new_morning_inputs = pd.DataFrame(
    [
        ["Tomato", "Saturday", 1, 29, 75, 1, "Clear"],
        ["Spinach", "Thursday", 0, 22, 19, 7, "Extreme Heat"],
        ["Potato", "Tuesday", 0, 20, 58, 2, "Clear"],
    ],
    columns=X.columns,
)

predictions = model.predict(new_morning_inputs)
new_morning_inputs["Recommended_Action"] = predictions

print("=== DAILY PROCUREMENT RECOMMENDATIONS ===")
for _, row in new_morning_inputs.iterrows():
    print(
        f"- {row['Item_Name']} ({row['Day_of_Week']}, {row['Weather_Condition']}, "
        f"₹{row['Wholesale_Price_Per_Kg']}/kg) -> {row['Recommended_Action']}"
    )