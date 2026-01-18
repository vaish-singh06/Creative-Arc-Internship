Dataset Used
Name: Heart Disease Dataset
Source: Kaggle
Task Type: Binary Classification (Presence vs Absence of heart disease)

Task 1 — Data Understanding

Dataset Shape, Columns & Types
After loading the dataset:
• Rows: ~303 (depending on version)
• Columns: ~14
• Target Column: target
• Feature Columns: age, trestbps, chol, thalach, oldpeak, cp, thal, slope, restecg, etc.

Target Column
target (0 = no disease, 1 = disease)

Feature Characteristics
• Numerical features representing medical measurements
• Encoded categorical features representing symptom categories
• No single feature alone predicts the target strongly
Minimal preprocessing applied:
• checked missing values
• basic encoding already provided in the dataset
• scaling only applied for KNN
• train/validation split applied

Why the Dataset Is Not Linearly Separable
Heart disease is influenced by multiple interacting medical factors. Features such as age, cholesterol, heart rate, and chest pain do not change in a simple linear manner with respect to the target. Patients with high cholesterol can be in either class, and younger patients can still show disease symptoms. These patterns form overlapping clusters rather than separable linear boundaries. Because of this, a single straight line (or hyperplane) cannot cleanly separate healthy vs diseased individuals. Models that capture non-linear interactions (Decision Trees, Random Forest, KNN) perform better on this dataset.

Why Model Selection Is Difficult on This Dataset

1.Mixed feature types (numerical + categorical encoded) introduce different statistical behavior.

2.Non-linear relationships cause linear models to show high bias.

3.Dataset is small, increasing overfitting risk in high capacity models.

4.Class boundaries overlap due to medical variability.

5.Real-world medical noise means some samples contradict typical patterns.

These factors make model choice non-trivial. Low-capacity models underfit (high bias), while high-capacity models overfit (high variance). This dataset therefore forces careful tuning and comparison rather than relying on accuracy alone.

Conclusion
The Heart Disease dataset is ideal for evaluating model selection and bias–variance reasoning because it contains non-linear structure, noisy real-world patterns, and a mix of feature types. It is complex enough to challenge models while small enough to experiment on effectively.