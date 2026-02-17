# Tabular Analysis Tutorial

This tutorial explains how to use this repository and run SHAP analysis on tree-based AI 
models using the configuration system.

This guide applies only to tabular tree-based models.


**What you'll learn:**
-  How to prepare your model
-  How to configure `config.json`
-  How to run the analysis
-  How to interpret the results

**Prerequisites:**
- A trained tree-based model (Decision Tree, Random Forest, etc.)
- Basic understanding of your model's features

---

# 1. Prepare Your Model

You must have:

- A trained model saved as `.pkl`
- A dataset in `.csv` or `.xlsx` format

Example (also available in the repository):

```bash
python create_example.py
```

This generates these two files:
- `realistic_dataset.csv`
- `realistic_decision_tree.pkl`

---

# 2. Fill in config.json

Fill in the `config.json` file with your model info.

IMPORTANT: Some fields in `config.json` are only used for time‑series analysis.
For tabular analysis you can leave them as empty strings `("")` or omit them.
Focus only on the fields shown in the examples below.

Example for binary classification (`create_example.py`):

```json
{
  "analysis_type": "tabular",
  "model_path": "realistic_decision_tree.pkl",
  "dataset_path": "realistic_dataset.csv",

  "package": "sklearn",
  "model_type": "random_forest",

  "feature_names": [
    "Age",
    "Income per Year",
    "Years of Employment"
  ],

  "output_labels": {
    "0": "Loan Rejected",
    "1": "Loan Approved"
  },

  "dataset_scope": "full",

  "generate_plots": true,
  "save_excel": true,
  "generate_notebook": true,

  "output_dir": "analysis_output"
}
```

---

# 3. Config Parameters Explained

### analysis

Specifies which analysis pipeline to use.
Must be:

```
"analysis": "tabular"
```

---

### model_path
Path to `.pkl` model file.

---

### dataset_path
Path to dataset file `(.csv or .xlsx/.xls)`.

---

### feature_names
List of input feature columns used for training.

IMPORTANT: must match the model training features in the same order.

```python
# Training
X_train = df[["Age", "Income", "Years"]]
model.fit(X_train, y)

# config.json MUST have
"feature_names": ["Age", "Income", "Years"]  # ✓
```

**Why:** Wrong order → wrong SHAP values.

---


### output_labels
Mapping of class index to readable label.

Used for:
- Console output
- Plots
- Excel file

Example:

```
"output_labels": {
  "0": "Loan Rejected",
  "1": "Loan Approved"
}
```

---

### dataset_scope

Controls how much of the dataset is used for explainability.

Options:

-`"full"`:  use the entire dataset

-`"subset"`: use only a row range

If using subset, indicate which rows:

```json
"dataset_scope": "subset",
"subset_start": 0,
"subset_end": 200
```

---

### generate_plots

Select: 

-`true`: If you want to generate plots

-`false`: If you don't want to generate plots

---

### save_excel
Whether to generate plots.

-`true`: generate all selected plots

-`false`: skip plot generation

---

### generate_notebook
Whether to generate a Jupyter notebook with the analysis.

-`true`: generate notebook

-`false`: skip notebook generation

---

### output_dir

The directory where results will be saved.

This is already selected by default unless you want to choose something different. 

Example:

```
"output_dir": "analysis_output"
```

---

# 4. Running the Analysis

Run the `main.py` file to run the whole explainability analysis.

From the project root:

```bash
python main.py
```

This reads `config.json`, 
routes to the correct analysis module, runs SHAP analysis and generates all outputs.

From Python:

```python
import json
from analysis import ANALYSIS_ROUTER

with open("config.json") as f:
    config = json.load(f)

ANALYSIS_ROUTER[config["analysis_type"]](config)
```

---

# 5. Regression Example

Example for multi-output regression (create_example_2.py):

```json
{
  "analysis_type": "tabular",
  "model_path": "energy_forecasting_model.pkl",
  "dataset_path": "energy_forecasting_dataset.csv",

  "package": "sklearn",
  "model_type": "random_forest",

  "feature_names": [
    "Wind Speed",
    "Temperature",
    "Previous Power",
    "Previous Load",
    "Hour"
  ],

  "dataset_scope": "full",

  "regression_bins": 5,

  "generate_plots": true,
  "save_excel": true,
  "generate_notebook": true,

  "output_dir": "energy_analysis"
}
```

For regression:
- Predictions are automatically binned for visualization.
- Original values are preserved in Excel output.

---

# 6. Output Structure

After running the analysis you'll find something like:
`

```text
analysis_output/
│
├── shap_results_TIMESTAMP.xlsx
│   └── Contains: Features + Predictions + SHAP values
│
├── 2025-01-01_12-00-00_classification_plots/
│   │
│   ├── analysis_notebook.ipynb  ← Start from here!!!
│   │
│   ├── Global Plots:
│   │   ├── shap_bar_unified.png         (Feature importance)
│   │   ├── shap_beeswarm_Class0.png     (Distribution)
│   │   └── shap_heatmap_unified.png     (All samples)
│   │
│   ├── Per-Class Plots:
│   │   ├── shap_bar_Class1.png
│   │   ├── shap_bar_Class0.png
│   │   └── ...
│   │
│   └── waterfall_plots/              (Individual explanations)
│       ├── waterfall_sample_0_Class0.png
│       ├── waterfall_mean_Class1.png
│       └── ...
```

**Recommended viewing order:**
1. Open **Jupyter Notebook** (`SHAP_Analysis_Report_*.ipynb`)
2. Check **Excel file** for raw SHAP values
3. Explore **plots** for details
---

# 7. Multi-Output Regression

If using MultiOutputRegressor:

- Separate SHAP plots are generated for each output.
- Unified feature‑importance plot (mean |SHAP| across outputs) is generated.
- Separate waterfall plots per output.

No special config is needed beyond providing the correct model and dataset.

---

# 8. Common Errors

### ❌ Feature mismatch
The `feature_names` list does not match the features used during training.

SOLUTION: ensure same names and order).

### ❌ Unsupported model type
Only tree‑based models (and their multi‑output wrappers) are supported in the tabular.

### ❌ Wrong dataset format
Supported:
- `.csv`
- `.xlsx`
- `.xls`

---

# 9. Best Practices

### ✔️Always use the same feature order as during training. 
### ✔️ Keep `generate_plots` and `save_excel` enabled when exploring a new model.  
### ✔️ Use `"dataset_scope": "subset"` with small ranges during debugging on large datasets.
### ✔️ Store `config.json` together with the model for full reproducibility.


---

# 10. Summary

Workflow:

1. Provide model `(.pkl)`
2. Provide dataset `(.csv / .xlsx)`
3. Create `config.json` with the correct parameters  
4. Run analysis (`python main.py`)  
5. Inspect:
   - Excel report
   - SHAP plots
   - Generated notebook  

You now have a complete explainability pipeline for tabular tree‑based models, 
supporting classification and regression with automated reporting.