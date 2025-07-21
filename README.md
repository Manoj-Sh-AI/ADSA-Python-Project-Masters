# House Price Prediction

A complete end‑to‑end pipeline for predicting house sale prices using structured real‑estate data. This project covers data loading, cleaning, exploratory analysis, feature engineering, model comparison, evaluation, and model serialization.

## 🚀 Features

- **Data Cleaning**: Handles missing values, type conversions, and drops irrelevant features.
- **EDA & Visualization**: Quick histograms, skewness analysis, and summary statistics.
- **Preprocessing Pipeline**: Standard scaling for numerical features and one‑hot encoding for categoricals via `ColumnTransformer`.
- **Model Comparison**: Automates hyperparameter tuning with `GridSearchCV` across multiple regressors (Linear, Random Forest, XGBoost, etc.).
- **Evaluation**: Reports RMSE and R² on hold‑out test data; plots top feature importances.
- **Model Persistence**: Saves the trained pipeline using `pickle` and `joblib` for easy deployment.

