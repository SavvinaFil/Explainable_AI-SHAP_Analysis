# This analysis is for decision trees and random forests - Supports both classification and regression
import numpy as np
import os
import shap
from datetime import datetime
from .tree_input import load_tree, load_dataset
from output.results import (
    compute_shap_values,
    show_shap_values,
    plot_shap_values,
    save_results_to_excel
)
from output.generate_notebook import generate_analysis_notebook


def is_classifier(model):
    # Determine if the model is classifier or regressor
    if hasattr(model, 'estimators_'):
        # Check the base estimator
        base_model = model.estimators_[0] if model.estimators_ else model
        return is_classifier(base_model)

    model_class_name = type(model).__name__.lower()

    # Check for common classifier names
    if 'classifier' in model_class_name:
        return True
    elif 'regressor' in model_class_name:
        return False

    # Check for predict_proba method (classifiers have this)
    if hasattr(model, 'predict_proba'):
        return True

    # Default to classifier for backward compatibility
    return True


def convert_regression_to_classes(predictions, n_bins=5):
    # Convert continuous regression predictions into discrete classes.

    # Create bins using quantiles for balanced distribution
    bin_edges = np.quantile(predictions, np.linspace(0, 1, n_bins + 1))

    # Ensure unique bin edges
    bin_edges = np.unique(bin_edges)
    actual_n_bins = len(bin_edges) - 1

    # Assign predictions to bins
    class_predictions = np.digitize(predictions, bin_edges[1:-1])

    # Create human-readable labels
    class_labels = {}
    for i in range(actual_n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        class_labels[str(i)] = f"Range [{lower:.2f}, {upper:.2f}]"

    return class_predictions, bin_edges, class_labels


def validate_model(model, expected_package, expected_model_type):
    # Validate that the loaded model matches the expected package and type
    model_class_name = type(model).__name__
    model_module = type(model).__module__

    # Check package
    if expected_package == "sklearn":
        if not model_module.startswith("sklearn"):
            print(f"WARNING: Expected sklearn model, but got {model_module}")
    elif expected_package == "xgboost":
        if not model_module.startswith("xgboost"):
            print(f"WARNING: Expected xgboost model, but got {model_module}")

    # Handle MultiOutput wrappers
    if model_class_name in ["MultiOutputRegressor", "MultiOutputClassifier"]:
        base_estimator = model.estimators_[0] if hasattr(model, 'estimators_') and model.estimators_ else None
        if base_estimator:
            base_name = type(base_estimator).__name__
            print(f"Model validated: {model_class_name} wrapping {base_name} from {model_module}")
        else:
            print(f"Model validated: {model_class_name} from {model_module}")
        return

    # Check model type
    model_type_mapping = {
        "decision_tree": ["DecisionTreeClassifier", "DecisionTreeRegressor"],
        "random_forest": ["RandomForestClassifier", "RandomForestRegressor"],
        "gradient_boosting": ["GradientBoostingClassifier", "GradientBoostingRegressor"],
        "xgboost": ["XGBClassifier", "XGBRegressor"]
    }

    if expected_model_type in model_type_mapping:
        expected_classes = model_type_mapping[expected_model_type]
        if model_class_name not in expected_classes:
            print(f"WARNING: Expected {expected_model_type}, but got {model_class_name}")

    print(f"Model validated: {model_class_name} from {model_module}")


def run_tabular_analysis(config):
    # Extract config parameters
    model_path = config["model_path"]
    dataset_path = config["dataset_path"]
    output_dir = config.get("output_dir")
    generate_plots = config.get("generate_plots")
    save_excel = config.get("save_excel")
    dataset_scope = config.get("dataset_scope")
    generate_notebook = config.get("generate_notebook")

    # Get feature names and output labels from config
    feature_names = config.get("feature_names", None)
    output_labels = config.get("output_labels", {})

    # Get package and model_type for validation
    expected_package = config.get("package")
    expected_model_type = config.get("model_type")

    # Load model
    model = load_tree(model_path)

    # Validate model
    validate_model(model, expected_package, expected_model_type)

    # Detect if model is classifier or regressor
    is_classification = is_classifier(model)

    if not is_classification:
        print(f"Model type: Regression")

    # Load dataset with feature names from config
    X_df = load_dataset(feature_names=feature_names, path_override=dataset_path)

    if feature_names is None:
        feature_names = list(X_df.columns)

    # Dataset start and stop from config
    if dataset_scope == "subset":
        start = config.get("subset_start", 0)
        end = config.get("subset_end", len(X_df))
        X_sample = X_df.iloc[start:end]
        print(f"Using dataset subset [{start}:{end}]")
    else:
        X_sample = X_df
        print("Using full dataset")

    # Compute SHAP values
    explainer, shap_values, X_df_aligned = compute_shap_values(model, X_sample)

    # Check if we have multiple outputs
    is_multi_output = hasattr(model, 'estimators_') and len(model.estimators_) > 1
    num_outputs = len(model.estimators_) if is_multi_output else 1

    # For multi-output models, create explainers list for waterfall plots
    if is_multi_output and not is_classification and hasattr(model, 'estimators_'):
        all_explainers = []
        for estimator in model.estimators_:
            exp = shap.TreeExplainer(estimator)
            all_explainers.append(exp)
    else:
        all_explainers = None

    # Calculate the predictions
    try:
        preds = model.predict(X_df_aligned)

        # Handle MultiOutput predictions
        if preds.ndim == 2 and preds.shape[1] > 1:
            all_outputs = preds
            preds_for_main = preds[:, 0]
        else:
            all_outputs = None
            preds_for_main = preds

        if is_classification:
            unique_classes, class_counts = np.unique(preds_for_main, return_counts=True)

            print("\nModel predictions (Classification):")
            for cls, cnt in zip(unique_classes, class_counts):
                percentage = (cnt / len(preds_for_main)) * 100
                # Use output labels if available
                class_label = output_labels.get(str(int(cls)), f"Class {cls}")
                print(f"  - {class_label}: {cnt} samples ({percentage:.1f}%)")

        else:
            # Convert to classes for visualization purposes
            n_bins = config.get("regression_bins", 5)
            preds_binned, bin_edges, auto_labels = convert_regression_to_classes(preds_for_main, n_bins=n_bins)

            # For regression, use auto-generated labels for bins, not output_labels
            bin_labels_for_display = auto_labels  # Use auto-generated bin ranges

            # Update predictions to binned version for plotting
            original_preds = preds_for_main.copy()  # Keep original for Excel
            preds = preds_binned
            unique_classes, class_counts = np.unique(preds, return_counts=True)

    except Exception as e:
        print(f"Error when calculating predictions: {e}")
        import traceback
        traceback.print_exc()
        return

    # Show SHAP values console
    if not is_classification:
        # For regression, use bin labels instead of output_labels
        show_shap_values(shap_values, feature_names, preds, bin_labels_for_display)
    else:
        show_shap_values(shap_values, feature_names, preds, output_labels)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save results to Excel
    if save_excel:
        shap_array = np.array(shap_values)

        # For regression, save both original and binned predictions
        if not is_classification:
            # Add original regression values to DataFrame
            save_results_to_excel(
                X_df_aligned, shap_array, feature_names, preds, output_dir,
                bin_labels_for_display, original_predictions=original_preds
            )
        else:
            save_results_to_excel(X_df_aligned, shap_array, feature_names, preds, output_dir, output_labels)
    else:
        print("Excel output disabled (config).")

    # Create plots
    plots_output_dir = None
    if generate_plots:
        selected_plots = [
            'beeswarm',
            'bar',
            'violin',
            'dependence',
            'decision_map',
            'interactive_decision_map',
            'heatmap',
            'interactive_heatmap',
            'waterfall'
        ]

        # Common folder with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        task_type = "classification" if is_classification else "regression"
        plots_output_dir = os.path.join(output_dir, f"{timestamp}_{task_type}_plots")
        os.makedirs(plots_output_dir, exist_ok=True)

        plot_shap_values(
            shap_values,
            X_df_aligned,
            feature_names,
            preds,
            plots_output_dir,
            selected_plots=selected_plots,
            explainer=explainer,
            output_labels=output_labels if is_classification else output_labels,
            # For classification: original labels, for regression: output names
            is_multi_output=is_multi_output and not is_classification,  # Only for multi-output regression
            all_outputs=all_outputs if not is_classification else None,
            model=model if not is_classification else None,
            all_explainers=all_explainers  # Pass explainers for waterfall plots
        )
    else:
        print("Plot generation disabled (config).")

    # Generate Jupyter Notebook with analysis
    if generate_notebook and plots_output_dir is not None:
        # Prepare model information for the notebook
        model_info = {
            'model_type': type(model).__name__,
            'task_type': 'Classification' if is_classification else 'Regression',
            'n_features': len(feature_names),
            'n_classes': len(unique_classes),
            'n_samples': len(X_df_aligned),
            'feature_names': feature_names,
            'classes': unique_classes.tolist(),
            'output_labels': bin_labels_for_display if not is_classification else output_labels
        }

        # Add regression-specific info
        if not is_classification:
            model_info['prediction_range'] = f"[{original_preds.min():.2f}, {original_preds.max():.2f}]"
            model_info['n_bins'] = len(unique_classes)

        try:
            notebook_path = generate_analysis_notebook(
                plots_output_dir,
                model_info=model_info
            )
            print(f"Notebook generated: {notebook_path}")

        except Exception as e:
            print(f"\nError generating notebook: {e}")
            print("All plots are still available in the output directory.")
            import traceback
            traceback.print_exc()
    elif not generate_plots:
        print("\nNotebook generation skipped (plots were not generated)")
    else:
        print("\nNotebook generation disabled (config).")