import shap
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def compute_shap_values(model, X_df):
    
    if not isinstance(X_df, pd.DataFrame):
        X_df = pd.DataFrame(X_df)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)

    return explainer, shap_values, X_df


def show_shap_values(shap_array, feature_names, preds=None):
    shap_array = np.array(shap_array)

    # Multi-class → select predicted class
    if shap_array.ndim == 3:
        if preds is None:
            raise ValueError("Predictions required for multi-class SHAP display")
        reshaped = []
        for i in range(len(preds)):
            cls = int(preds[i])
            reshaped.append(shap_array[i, :, cls])
        shap_array = np.array(reshaped)

    if shap_array.ndim == 1:
        shap_array = shap_array.reshape(1, -1)

    print("\nSHAP values per feature:")
    for i, row in enumerate(shap_array):
        if preds is not None:
            print(f"Row {i + 1} (predicted: {preds[i]}):")
        else:
            print(f"Row {i + 1}:")
        for j, feat in enumerate(feature_names):
            print(f"  {feat}: {float(row[j]):.4f}")


def save_results_to_excel(X_df, shap_array, feature_names, output_dir):
    shap_array = np.array(shap_array)
    if shap_array.ndim == 1:
        shap_array = shap_array.reshape(1, -1)

    df_shap = pd.DataFrame(shap_array, columns=[f"SHAP_{f}" for f in feature_names])
    df_out = pd.concat([X_df.reset_index(drop=True), df_shap], axis=1)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "shap_results.xlsx")
    df_out.to_excel(out_path, index=False)

    print(f"SHAP results saved to {out_path}")


def plot_shap_values(shap_values, X_df, feature_names, output_dir, preds=None):
    
    os.makedirs(output_dir, exist_ok=True)

    shap_array = np.array(shap_values)

    # Multi-class → pick predicted class
    if shap_array.ndim == 3:
        if preds is None:
            raise ValueError("Predictions required for multi-class SHAP plots")
        reshaped = []
        for i in range(len(preds)):
            cls = int(preds[i])
            reshaped.append(shap_array[i, :, cls])
        shap_array = np.array(reshaped)

    if shap_array.ndim == 1:
        shap_array = shap_array.reshape(1, -1)

    # Summary plot
    plt.figure()
    shap.summary_plot(shap_array, X_df, show=False)
    plt.savefig(os.path.join(output_dir, "shap_summary.png"))
    plt.close()

    # Bar plot
    plt.figure()
    shap.summary_plot(shap_array, X_df, plot_type="bar", show=False)
    plt.savefig(os.path.join(output_dir, "shap_bar.png"))
    plt.close()

    print(f"SHAP plots saved to {output_dir}")

