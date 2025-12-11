import pickle
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import pandas as pd
import os

VALID_MODEL_TYPES = (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)

def load_tree(model_path):
    print(f"Loading model from: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    if not isinstance(model, VALID_MODEL_TYPES):
        raise ValueError("The loaded model is not a valid Decision Tree type!")
    print(f"Model loaded type: {type(model)}")
    return model

def get_features_from_user(feature_names):
    feature_values = []
    print("\nEnter values for each feature (numeric):")
    for feature in feature_names:
        while True:
            val = input(f"{feature}: ")
            try:
                val = float(val)
                feature_values.append(val)
                break
            except ValueError:
                print("Please enter a numeric value.")
    return feature_values

def load_dataset(choice, feature_names=None):
    while True:
        path = input("Enter file path: ").strip()
        if path == "":
            default_name = "dataset.csv" if choice == 2 else "dataset.xlsx"
            path = os.path.join(os.getcwd(), default_name)
            print(f"No path provided, using default: {path}")

        if not os.path.isfile(path):
            print(f"File not found: {path}. Try again.")
            continue

        try:
            if choice == 2:
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
            break
        except Exception as e:
            print(f"Error reading file: {e}")
            continue

    if feature_names is None or len(feature_names) == 0:
        feature_names = list(df.columns)
        if "Loan_Approved" in feature_names:
            feature_names.remove("Loan_Approved")

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    return df[feature_names]
