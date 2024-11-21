# High-Impact Lending Decisions

## Project Overview

This project delves into the critical process of loan approval in the banking sector, which involves evaluating multiple customer variables such as credit score, income stability, employment history, and existing debt levels. The objective is to create a robust framework for predicting loan approval decisions using machine learning and statistical methods.

The project involves working with two datasets containing information about over 50,000 customers, their financial behaviors, and credit attributes. The aim is to build predictive models to classify loan approval categories (`P1`, `P2`, `P3`, and `P4`), while addressing challenges such as data imbalance and multicollinearity.

---

## Features and Methodology

1. **Data Preprocessing:**
   - Null value handling using a combination of deletion and imputation strategies.
   - Detection and resolution of multicollinearity using the Variance Inflation Factor (VIF).
   - Label encoding and one-hot encoding for categorical variables.

2. **Exploratory Data Analysis (EDA):**
   - Chi-square tests for categorical features.
   - Analysis of numerical features using descriptive statistics and ANOVA.

3. **Machine Learning Models:**
   - Decision Tree Classifier
   - Random Forest Classifier
   - XGBoost Classifier with hyperparameter tuning using GridSearchCV.

4. **Key Challenges Addressed:**
   - Imbalanced data for the `P3` class.
   - Hyperparameter optimization for improving precision and recall.
   - Future steps to explore clustering algorithms to further enhance model performance.

---

## Dataset

The project uses two datasets with:
- **Dataset 1:** 26 columns.
- **Dataset 2:** 62 columns.
- Final merged dataset: 79 columns after pre-processing.

Features include:
- Customer demographics.
- Financial history.
- Loan-related metrics.

---

## Libraries Required

```bash
pip install xgboost pandas numpy matplotlib scikit-learn
