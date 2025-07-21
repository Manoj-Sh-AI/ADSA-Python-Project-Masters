# House Price Prediction Web App


A machine learning-powered web application that predicts the price of a house based on user-input features such as location, property type, size, and amenities.


## Table of Contents

- Overview
- Dataset
- Features Used
- Data Preprocessing
- Modeling
- Web App
- How to Run
- Conclusion
- Future Scope


## Overview

This project aims to predict house prices using machine learning techniques. The application allows users to input property details and returns an estimated market price instantly via a web interface.


## Dataset

[https://www.kaggle.com/datasets/polartech/500000-us-homes-data-for-sale-properties]

- **Rows:** 50,000
- **Columns:** 28
- **Target Variable:** **price**
- **Some fields include:** bedrooms, bathrooms, land/living space, postcode, property type, and status.


## Features Used
- bedroom_number
- bathroom_number
- living_space
- land_space
- property_type
- property_status
- postcode
- city
- state


## Data Preprocessing
- **Dropped:** Columns with high null values (e.g., agent info, URLs)
- **Missing Values:**
    - **Categorical ->** Filled with **"Missing"**
    - **Numerical ->** Imputed with median
- **Duplicates:** None found
- **Outliers:** Removed using z-score filtering
- **Encoding:** One-hot encoding for categorical features
- **Scaling:** StandardScaler used for numerical features


## Modeling
- **Models Trained:** Linear Regression, Random Forest, SVR, CatBoost, XGBoost
- **Best Performer:** CatBoost Regressor
- **Evaluation Metrics:** R², MAE, RMSE
- Final model saved as **.pkl** using **pickle**


## Web App
- Built using Streamlit
- Users can input features and receive real-time price estimates
- Simple, fast, and interactive


## How to Run
1. Clone the repo:
`git clone https://github.com/your-username/house-price-predictor.git cd house-price-predictor`

2. Install dependencies:
`pip install -r requirements.txt`

3. Run the Streamlit app:
`streamlit run app.py`


## Conclusion
This project demonstrates an end-to-end machine learning pipeline — from raw data to a deployed web application — capable of accurately predicting house prices based on user input.


## Future Scope
- Integrate live market data and mapping APIs
- Host the app on a cloud platform (e.g., Heroku, Render)
- Expand feature set with geospatial data and neighborhood stat