The formal definition of the Shapley value for a feature $i$ is:

$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(n - |S| - 1)!}{n!} [v(S \cup \{i\}) - v(S)]$$

### How are they computed?
The computation follows a "leave-one-out" logic across every possible subset of features:

* **Coalitions:** The algorithm creates all possible combinations (coalitions) of input features.
* **Marginal Contribution:** For each coalition, it measures how the prediction changes when a specific feature is added versus when it is absent.
* **Weighted Average:** The final SHAP value is the weighted average of these marginal contributions across all permutations.

This process ensures the analysis is **additive**: the sum of all SHAP values plus the base value (the average model output) will exactly equal the final prediction.