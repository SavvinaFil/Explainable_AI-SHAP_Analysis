# Model-Agnostic Explainability Analysis for AI models

This document describes the **architecture, design principles
and implementation** of the TreeShap explainability framework. It provides the theoretical foundation and system design for understanding how SHAP analysis is performed on tree-based models.


## 1. Overview

This project provides a modular framework for analyzing machine learning models 
(Decision Trees, Random Forests, timeseries) using SHAP explainability.

It supports:

- Classification
- Regression
- Multi-output Regression
- Automated SHAP analysis
- Plot generation
- Excel export
- Jupyter notebooks

The architecture is designed to be:
- Modular
- Extensible
- Config-driven
- Separated into logical layers

---

## 2. System Architecture

The system follows a layered modular architecture:

```
User (config.json)
        │
        ▼
ANALYSIS_ROUTER
        │
        ▼
Selected Analysis Module
        │
        ├── Model Loading
        ├── Dataset Loading
        ├── Validation
        ├── Prediction
        ├── SHAP Computation
        ├── Visualization
        ├── Excel Export
        ├── Plots
        └── Notebook Generation
```

---

## 3. Project Structure

### 3.1 Data & Model Creation

- `create_example.py`  
  Generates synthetic classification dataset and trains a RandomForestClassifier.

- `create_example_2.py`  
  Generates synthetic multi-output regression dataset for energy forecasting and trains a MultiOutput RandomForestRegressor.

These are example pipelines for demonstration.

---

### 3.2 Core Analysis Layer

#### `analysis/__init__.py`

Acts as a router for different analysis types.

```python
ANALYSIS_ROUTER = {
    "tabular": run_tabular_analysis,
    "timeseries": run_timeseries_analysis
}
```

The router dynamically selects the appropriate analysis pipeline 
based on the `analysis_type` specified in `config.json`
---

### 3.3.1 Tabular Analysis Module

Located in:

```
analysis/tabular/tree_based/
```

Main entry point:
- `run_tabular_analysis(config)`

Responsibilities:
- Load model
- Validate model type
- Load dataset
- Compute SHAP values
- Generate plots
- Save results
- Generate notebook

### 3.3.2 Time-Series Analysis Module

Located in:

```
analysis/timeseries/
```

Main entry point:
- `run_timeseries_analysis(config)`

Responsibilities:
- Load time-series model
- Load sequential dataset
- Handle temporal windowing
- Generate predictions
- Compute SHAP explanations (model-dependent)
- Produce time-aware visualizations
- Export results and notebook

The time-series module is designed to support deep learning or sequential models
while maintaining compatibility with the same reporting system used for tabular models.

---

**Responsibilities:**
1. **Model Loading & Validation:**
   - Loads `.pkl` file
   - Checks feature count compatibility

2. **Dataset Processing:**
   - Loads CSV/Excel files
   - Applies feature subset
   - Handles subset slicing

3. **SHAP Computation:**
   - Computes exact Shapley values
   - Handles multi-class/multi-output cases

4. **Output Generation:**
   - Orchestrates plot creation
   - Exports Excel file
   - Generates Jupyter notebook

---

#### `output/results.py`

**Main Functions:**

1. `compute_shap_values()`
   - Creates TreeExplainer
   - Returns: explainer, shap_values, aligned_df

2. `plot_shap_values()`
   - Generates all visualization types
   - Handles per-class and unified plots

3. `save_results_to_excel()`
   - Exports features + predictions + SHAP values
   - Handles both classification and regression

**Plot Types Generated:**
- Beeswarm, Bar, Violin (global importance)
- Dependence (feature interactions)
- Decision (prediction paths)
- Heatmap (sample overview)
- Waterfall (individual explanations)

---

### 3.4 Model & Dataset Handling

#### `tree_input.py` (Tabular)

Provides:

- `load_tree(model_path)`
- `load_dataset(feature_names, path_override)`

Supports:
- CSV
- Excel (.xlsx, .xls)

---

### 3.5 SHAP Processing & Output

Located in:

```
output/results.py
```

Key responsibilities:

- Compute SHAP values
- Display SHAP values
- Save results to Excel
- Generate:
  - Beeswarm plots
  - Bar plots
  - Violin plots
  - Dependence plots
  - Decision Map plots
  - Heatmaps
  - Waterfall plots

Supports:
- Binary classification
- Multi-class classification
- Regression
- Multi-output regression

---

## 4. Execution Flow (Tabular Analysis)

1. User defines `config.json`
2. Router selects analysis module (tabular or timeseries)
3. Model is loaded
4. Model is validated
5. Dataset is loaded
6. Optional dataset slicing is applied
7. SHAP values are computed
8. Predictions are generated
9. Results exported:
   - Console output
   - Excel file
   - Plots
   - Jupyter notebook

---

## 5. Design Principles

### ✔ Configuration Driven
All behavior is controlled by `config.json`.

### ✔ Model Agnostic (Tree-Based)
Supports:
- sklearn
- XGBoost
- PyTorch

### ✔ Safe for Multi-Output Models
Automatically unwraps base estimators where required.

### ✔ Reproducible
Uses fixed random seeds in examples.

### ✔ Extensible
New analysis modules can be added via `ANALYSIS_ROUTER`.

---

## 6. Supported Model Types

| Type | Supported |
|------|-----------|
| DecisionTreeClassifier | ✅ |
| RandomForestClassifier | ✅ |
| DecisionTreeRegressor | ✅ |
| RandomForestRegressor | ✅ |
| GradientBoosting | ✅ |
| XGBoost | ✅ |
| MultiOutputRegressor | ✅ |
| MultiOutputClassifier | ✅ |
| Time-Series Models | ✅ |

---

## 7. Output Artifacts

The system can generate:

- SHAP value console summary
- Excel file with:
  - Features
  - Predictions
  - SHAP values
- Structured plot directory
- Auto-generated Jupyter notebook

---

## 8. Configuration Specification

###  Required Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `analysis` | string | Analysis type | `"tabular"` or `"timeseries"` |
| `package` | string | ML library | `"sklearn"`, `"xgboost"` |
| `model_type` | string | Model architecture | `"decision_tree"`, `"random_forest"`, `"gradient_boosting"` |
| `model_path` | string | Path to `.pkl` file | `"models/rf_model.pkl"` |
| `dataset_path` | string | Path to data (.csv/.xlsx) | `"data/features.csv"` |
| `feature_names` | array | **Ordered** feature list | `["Age", "Income", "Years"]` |


### Optional Fields

| Field | Default | Description |
|-------|---------|-------------|
| `output_dir` | `"output"` | Results directory |
| `output_labels` | Auto | Class/output names |
| `generate_plots` | `true` | Create visualizations |
| `save_excel` | `true` | Export to Excel |
| `dataset_scope` | `"whole"` | Use `"subset"` for large data |
| `regression_bins` | `5` | Bins for regression viz |

### 8.3 Critical Rules
---

## 9. Validation & Error Handling

### 9.1 Pre-Execution Checks

The system validates before SHAP computation:
```python
# 1. File Existence
assert os.path.exists(model_path), "Model file not found"
assert os.path.exists(dataset_path), "Dataset file not found"

# 2. Model Compatibility
assert isinstance(model, SUPPORTED_MODELS), "Unsupported model type"

# 3. Feature Consistency
assert len(feature_names) == model.n_features_in_, "Feature count mismatch"

# 4. Dataset Validity
assert all(f in dataset.columns for f in feature_names), "Missing features"

# 5. Subset Bounds (if applicable)
if dataset_scope == "subset":
    assert 0 <= subset_start < subset_end <= len(dataset)
```

### 9.2 Runtime Warnings

**Model Type Mismatch:**
```
WARNING: Expected random_forest, but got DecisionTreeClassifier
```
→ Analysis proceeds with actual model type

**Missing Output Labels:**
```
INFO: No output_labels provided. Using defaults: Class 0, Class 1, ...
```
→ Auto-generated labels used

---

## 9. Summary

This framework provides a production-ready explainability pipeline 
for both tabular and time-series machine learning models with automated reporting.

It bridges:

Model → SHAP → Visualization → Reporting

In a fully configurable and extensible architecture.

