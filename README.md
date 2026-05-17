# 🫀 ECG Arrhythmia Classification Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-green)
![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-brightgreen)
![CatBoost](https://img.shields.io/badge/CatBoost-Boosting-yellow)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-red)
![LIME](https://img.shields.io/badge/LIME-Explainability-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> Classifying ECG heartbeat arrhythmia types using **12 machine learning classifiers** — with SMOTE oversampling, stratified K-Fold cross-validation, Friedman statistical testing, TOPSIS multi-criteria ranking, and SHAP/LIME explainability.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Advanced Analysis](#advanced-analysis)
- [Results](#results)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Contributors](#contributors)

---

## 📖 Overview

This project applies **12 machine learning classifiers** to ECG arrhythmia data to automatically detect and classify abnormal heartbeat types. The complete pipeline covers:

- Data loading and exploratory analysis from two ECG datasets
- Class imbalance handling with **SMOTE** (Synthetic Minority Over-sampling Technique)
- Hyperparameter tuning via **RandomizedSearchCV**
- Training and evaluation using **Stratified K-Fold Cross-Validation**
- Multi-criteria model ranking using **TOPSIS**
- Statistical significance testing via **Friedman Test** (with post-hoc Nemenyi test)
- Model explainability using **SHAP** and **LIME** on the best-performing models

---

## 📊 Datasets

### 1. ECG Arrhythmia Classification Dataset (Kaggle)
| Property | Detail |
|----------|--------|
| **Source** | [Kaggle – ECG Arrhythmia Classification Dataset](https://www.kaggle.com/datasets/sadmansakib7/ecg-arrhythmia-classification-dataset) |
| **Domain** | 12-lead ECG signal features |
| **Target** | Multi-class arrhythmia type |

### 2. INCART 2-Lead Arrhythmia Database
| Property | Detail |
|----------|--------|
| **Source** | INCART 2-lead Arrhythmia Database (INCART 2-lead Arrhythmia Database.csv) |
| **Format** | 2-lead ECG feature records |
| **Beat Types** | Normal (N), Ventricular Ectopic Beat (VEB), and other arrhythmia classes |
| **Features** | Pre/Post-RR intervals, P/T/R/S/Q peak amplitudes, QRS interval, PQ interval, QT interval, ST interval, QRS morphology coefficients (per lead) |

> Both datasets contain morphological ECG features extracted beat-by-beat, representing waveform characteristics such as peak amplitudes, interval durations, and QRS morphology.

---

## 📁 Project Structure

```
Arrhythmia_Classification_Project/
│
├── Code/
│   ├── Smote_K_Fold_Arrhythmia.ipynb                        # SMOTE + 12 classifiers + K-Fold CV + SHAP + LIME
│   ├── Smote_Friedman_Test_Among_12_Models_Arrhythmia.ipynb # Friedman statistical test across all 12 models
│   ├── Topsis_Arrhythmia.ipynb                              # TOPSIS multi-criteria ranking
│   ├── Shap_Lime_Best_Models_Arrhythmia.ipynb               # SHAP & LIME on best models
│   └── ML_Results_Arrhythmia.ipynb                          # Results analysis & visualization
│
├── Output/
│   ├── Smote_Arrhythmia.xlsx               # Post-SMOTE cross-validation results
│   ├── all_models.xlsx                     # Per-fold metrics for all 12 models
│   ├── Topsis_Arrhythmia.xlsx             # TOPSIS rankings
│   └── Friedman_Test_Results.xlsx         # Friedman & post-hoc test output
│
├── dataset/
│   ├── INCART 2-lead Arrhythmia Database.csv
│   └── ecg_arrhythmia_kaggle.csv
│
└── README.md
```

---

## 🤖 Models Used

| # | Model | Library | Type |
|---|-------|---------|------|
| 1 | Logistic Regression | scikit-learn | Linear |
| 2 | Decision Tree | scikit-learn | Tree-based |
| 3 | Random Forest | scikit-learn | Ensemble (Bagging) |
| 4 | Bagging Classifier | scikit-learn | Ensemble (Bagging) |
| 5 | Stacking Classifier | scikit-learn | Ensemble (Meta-learning) |
| 6 | Gradient Boosting Machine (GBM) | scikit-learn | Ensemble (Boosting) |
| 7 | XGBoost | xgboost | Ensemble (Boosting) |
| 8 | LightGBM | lightgbm | Ensemble (Boosting) |
| 9 | CatBoost | catboost | Ensemble (Boosting) |
| 10 | Support Vector Machine (SVM) | scikit-learn | Kernel-based (LinearSVC) |
| 11 | K-Nearest Neighbors (KNN) | scikit-learn | Instance-based |
| 12 | MLP Classifier | scikit-learn | Neural Network |

All models undergo hyperparameter optimization via **RandomizedSearchCV** before evaluation.

---

## 📏 Evaluation Metrics

Each model is evaluated per fold and aggregated across all folds:

| Metric | Description |
|--------|-------------|
| ✅ **Accuracy** | Overall correct predictions |
| 🎯 **Precision** | Positive predictive value |
| 🔁 **Recall** | Sensitivity / True Positive Rate |
| ⚖️ **F1-Score** | Harmonic mean of precision and recall |
| 📈 **ROC-AUC** | Area under the ROC curve |
| 🧮 **Matthews Correlation Coefficient (MCC)** | Balanced metric for imbalanced classes |
| 🤝 **Cohen's Kappa Score** | Inter-rater agreement beyond chance |
| 🧮 **Balanced Accuracy** | Average recall per class |
| 🗂️ **Confusion Matrix** | Detailed class-level breakdown |
| ⏱️ **Training Time** | Computational efficiency |

---

## 🔬 Advanced Analysis

### 🔄 SMOTE (Synthetic Minority Over-sampling Technique)
ECG arrhythmia datasets are inherently imbalanced — most beats are normal (class N), while dangerous arrhythmia types are rare. SMOTE generates synthetic minority-class samples to balance the training set before model fitting.

### 📂 Stratified K-Fold Cross-Validation
All models are evaluated using **Stratified K-Fold** to ensure class proportion is preserved across every fold, producing robust, generalisable performance estimates.

### 🏆 TOPSIS (Multi-Criteria Model Ranking)
All 12 models are ranked simultaneously across multiple performance metrics using the **Technique for Order of Preference by Similarity to Ideal Solution** — providing a single composite ranking that reflects overall model quality.

### 📐 Friedman Test (Statistical Comparison)
The **Friedman Test** (non-parametric equivalent of repeated-measures ANOVA) is used to determine whether statistically significant differences exist among the 12 classifiers across all K-Fold runs. Post-hoc **Nemenyi pairwise testing** identifies which specific model pairs differ significantly.

### 🔍 SHAP & LIME (Explainability)
- **SHAP** (SHapley Additive exPlanations): Global feature importance and local per-prediction explanations for the best-performing models
- **LIME** (Local Interpretable Model-agnostic Explanations): Individual instance-level explanations for model predictions

---

## 📋 Results

> Full results are available in the `Output/` directory.

| Model | Accuracy | F1-Score | ROC-AUC | TOPSIS Rank |
|-------|----------|----------|---------|-------------|
| CatBoost | — | — | — | 🥇 |
| XGBoost | — | — | — | 🥈 |
| LightGBM | — | — | — | 🥉 |
| Random Forest | — | — | — | — |
| ... | ... | ... | ... | ... |

> *Fill in exact values from `Output/Smote_Arrhythmia.xlsx` and `Output/Topsis_Arrhythmia.xlsx` after running the notebooks.*

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.8+
- pip or conda
- GPU recommended (T4 or equivalent) for faster training

### Install Required Libraries

```bash
pip install numpy pandas scikit-learn xgboost lightgbm catboost imbalanced-learn \
            shap lime matplotlib seaborn openpyxl xlsxwriter scipy scikit-image \
            specificity ipython-autotime jupyter
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### requirements.txt

```
numpy
pandas
scikit-learn
xgboost
lightgbm
catboost
imbalanced-learn
shap
lime
matplotlib
seaborn
openpyxl
xlsxwriter
scipy
scikit-image
specificity
ipython-autotime
jupyter
```

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended)
1. Upload the notebooks to Google Colab
2. Mount your Google Drive: `drive.mount('/content/drive')`
3. Upload `INCART 2-lead Arrhythmia Database.csv` and the Kaggle dataset to your Drive
4. Run the notebooks in order (see below)

Enable **GPU runtime** in Colab (`Runtime → Change runtime type → T4 GPU`) for faster training of boosting models.

### Option 2: Local Jupyter

```bash
# Clone the repository
git clone https://github.com/<your-username>/Arrhythmia_Classification_Project.git
cd Arrhythmia_Classification_Project

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

**Run notebooks in this order:**

1. `Code/Smote_K_Fold_Arrhythmia.ipynb` — Preprocess data, apply SMOTE, train all 12 models with K-Fold CV
2. `Code/Smote_Friedman_Test_Among_12_Models_Arrhythmia.ipynb` — Statistical comparison (Friedman + Nemenyi)
3. `Code/Topsis_Arrhythmia.ipynb` — Rank models with TOPSIS
4. `Code/Shap_Lime_Best_Models_Arrhythmia.ipynb` — Explain best models with SHAP & LIME
5. `Code/ML_Results_Arrhythmia.ipynb` — Visualise and summarise results

---

## 👥 Contributors

| Name |
|------|
| **Shreya Ghora** |
| **Abhijit Bera** |
| **Sk Akhmal Hossain** | 
| **Purbasa Bhowmick** |
| **Anima Shyamal** |

---

## 👤 Author

**Shreya Ghora**
📧 GitHub: [@shreyaghora](https://github.com/shreyaghora)

---

## 📚 References

- Sadman Sakib. *ECG Arrhythmia Classification Dataset*. Kaggle. [Link](https://www.kaggle.com/datasets/sadmansakib7/ecg-arrhythmia-classification-dataset)
- Taddei A. et al. *INCART 2-lead Arrhythmia Database*. PhysioNet.
- Chawla N. V. et al. *SMOTE: Synthetic Minority Over-sampling Technique*. JAIR, 2002.
- Friedman M. *The Use of Ranks to Avoid the Assumption of Normality Implicit in the Analysis of Variance*. JASA, 1937.
- Lundberg S. M., Lee S-I. *A Unified Approach to Interpreting Model Predictions (SHAP)*. NeurIPS, 2017.
- Ribeiro M. T. et al. *"Why Should I Trust You?": Explaining the Predictions of Any Classifier (LIME)*. KDD, 2016.
