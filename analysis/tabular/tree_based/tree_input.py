import pandas as pd
import pickle
import os


def load_tree(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def load_dataset(feature_names=None, path_override=None):
    if path_override:
        path = path_override
    else:
        raise ValueError("Path must be provided via config")

    # Auto-detect file type from extension
    file_ext = os.path.splitext(path_override)[1].lower()

    try:
        # Load based on file extension
        if file_ext == '.csv':
            df = pd.read_csv(path_override)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(path_override)
        else:
            raise ValueError(
                f"Unsupported file format: {file_ext}\n"
                f"Supported formats: .csv, .xlsx, .xls"
            )
    except Exception as e:
        raise ValueError(f"Error reading dataset from {path_override}: {e}")

    if feature_names is None:
        feature_names = list(df.columns)

    return df[feature_names]


