# 🌱 Organic Farming Suitability Prediction

An end-to-end Machine Learning project designed to evaluate and predict agricultural land suitability for organic farming based on soil chemistry, historical agricultural practices, and climatic/environmental conditions.

---

## 📌 Project Overview

Transitioning conventional farmland to certified organic farming requires favorable soil health and low historical chemical residue. This project builds a supervised classification pipeline to automate the suitability assessment (`Suitable` vs. `Not Suitable`), helping agronomists, farmers, and policymakers make data-driven decisions.

### Key Objectives
- **Data Preprocessing & Cleaning:** Handle missing values, scale numeric soil parameters, and encode categorical crop/soil features via scikit-learn pipelines.
- **Model Benchmarking:** Train and evaluate multiple classification models (Logistic Regression, Random Forest, Decision Tree, and K-Nearest Neighbors).
- **Explainability:** Compute permutation feature importances to highlight the critical soil parameters dictating organic farming feasibility.
- **Production Artifacts:** Persist the best-performing pipeline into a serialized model ready for immediate inference.

---

## 📂 Repository Structure

```text
Machine-Learning-Project/
├── .gitignore                                 # Git ignore configuration for Jupyter & Python
├── README.md                                  # Project documentation and summary
├── Untitled.ipynb                             # Complete EDA, preprocessing, training & evaluation notebook
└── project_outputs/                           # Exported artifacts and benchmark metrics
    ├── best_organic_farming_model.pkl         # Trained scikit-learn pipeline (joblib)
    ├── model_comparison.csv                   # Performance metrics across all tested models
    └── permutation_feature_importance.csv     # Permutation feature importance rankings
```

### Folder & File Descriptions

- **`Untitled.ipynb`**:
  The primary Jupyter notebook containing:
  1. **Exploratory Data Analysis (EDA):** Class distribution, correlation heatmaps, feature distribution plots.
  2. **Preprocessing Pipelines:** Median imputation and standard scaling for numeric features; most-frequent imputation and one-hot encoding for categorical variables.
  3. **Model Training & Cross-Validation:** Multi-model comparative analysis.
  4. **ROC Curves & Confusion Matrices:** Comprehensive diagnostic plots.
  5. **Inference Function:** Reusable function to predict on unseen soil sample data.

- **`project_outputs/`**:
  - **`best_organic_farming_model.pkl`**: Serialized pipeline combining preprocessing and the top-performing classifier, allowing predictions without manual feature transformation.
  - **`model_comparison.csv`**: Tabular record of test set accuracy, precision, recall, F1-score, and ROC-AUC.
  - **`permutation_feature_importance.csv`**: Mean and standard deviation of importance scores for all predictor variables.

---

## 📊 Model Performance Comparison

The models were evaluated on a held-out test split (20%) using standard classification metrics:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Best)** | **80.50%** | **78.75%** | **74.12%** | **0.7636** | **0.8891** |
| Random Forest | 78.00% | 77.33% | 68.24% | 0.7250 | 0.8873 |
| K-Nearest Neighbors (KNN) | 76.50% | 77.14% | 63.53% | 0.6968 | 0.8144 |
| Decision Tree | 75.50% | 70.93% | 71.76% | 0.7135 | 0.7501 |

> 🏆 **Selected Best Model:** **Logistic Regression** achieved the highest test F1-score (**0.7636**) and top ROC-AUC (**0.8891**), offering both superior generalization and high interpretability.

---

## 🔍 Key Feature Importances

Permutation feature importance analysis revealed the primary factors driving organic suitability:

| Rank | Feature | Description | Importance (Mean) |
| :---: | :--- | :--- | :---: |
| 1 | `Organic_Carbon_pct` | Soil Organic Carbon (%) | +0.1977 |
| 2 | `pH` | Soil pH Level | +0.0756 |
| 3 | `Available_P_kg_ha` | Available Phosphorus (kg/ha) | +0.0678 |
| 4 | `Previous_Fertilizer_Usage`| Historical synthetic fertilizer use | +0.0508 |
| 5 | `Available_N_kg_ha` | Available Nitrogen (kg/ha) | +0.0445 |
| 6 | `Available_K_kg_ha` | Available Potassium (kg/ha) | +0.0296 |

**Insight:** Soil organic carbon and balanced soil pH are the strongest determinants of organic farming suitability, followed by residual nutrient balances and prior chemical dependency.

---

## 🚀 Quickstart & Usage

### 1. Prerequisites & Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Atanumondal70/Machine-Learning-Project.git
cd Machine-Learning-Project
pip install numpy pandas scikit-learn matplotlib seaborn joblib
```

### 2. Loading the Model for Inference

The trained pipeline accepts raw tabular inputs matching the training feature schema and handles all preprocessing internally:

```python
import joblib
import pandas as pd

# Load the saved model pipeline
pipeline = joblib.load("project_outputs/best_organic_farming_model.pkl")

# Sample input DataFrame (with required predictor columns)
sample_data = pd.DataFrame([{
    "Organic_Carbon_pct": 0.85,
    "pH": 6.8,
    "Available_N_kg_ha": 280.0,
    "Available_P_kg_ha": 22.5,
    "Available_K_kg_ha": 190.0,
    "Previous_Fertilizer_Usage": "Low",
    "Previous_Pesticide_Usage": "None",
    "Years_Chemical_Input": 1.0,
    "Soil_Moisture_pct": 24.0,
    "Soil_Type": "Loamy",
    "Crop": "Wheat",
    "State": "Punjab",
    "Annual_Rainfall_mm": 650.0,
    "Mean_Temperature_C": 25.0,
    "EC_dS_m": 0.45,
    "Available_S_mg_kg": 15.0,
    "Available_Zn_mg_kg": 1.2,
    "Available_Fe_mg_kg": 5.0,
    "Available_Cu_mg_kg": 0.8,
    "Available_Mn_mg_kg": 4.0,
    "Available_B_mg_kg": 0.5
}])

# Make predictions
prediction = pipeline.predict(sample_data)
probabilities = pipeline.predict_proba(sample_data)

print(f"Prediction: {prediction[0]}")
print(f"Probabilities: {probabilities[0]}")
```

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn`
- **Visualization:** `matplotlib`, `seaborn`
- **Model Serialization:** `joblib`
