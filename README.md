# Real Estate Market Analysis and Prediction

## Abstract
This project analyzes real estate trends across the United States using historical housing data sourced from Kaggle. By conducting rigorous preprocessing, exploratory data analysis, feature engineering, modeling, and dashboard development, the goal was to uncover meaningful patterns in property prices, identify key drivers of value, and make real estate data more accessible and useful for decision-making.

Initially, the project's scope included analyzing real estate, stock, and gold markets together, but after consultation with the instructor, the focus was narrowed to **only real estate** to ensure a more detailed and meaningful analysis.

---

## Milestone 1: Data Collection, Preprocessing, and EDA

- **Data Ingestion:**
  - Merged weekly real estate listing CSVs.
  - Parsed `snapshot_date` and `data_gen_date` columns.
  - Sampled 100,000 rows for efficiency.

- **Data Preprocessing:**
  - Removed columns with over 70% missing values.
  - Median imputation for numerical fields; 'Unknown' for categorical fields.
  - Outlier handling (price, living area, year built bounds).
  - Added features: `price_per_sqft`, `lot_size_acres`.

- **Exploratory Data Analysis (EDA):**
  - Correlation heatmap, time series of price trends.
  - Boxplots for city-wise price comparisons.
  - Latitude and longitude distribution analysis.

---

## Milestone 2: Feature Engineering, Feature Selection, and Modeling

- **Feature Engineering:**
  - `property_age`, `total_baths_final`, `lot_size_acres_calc`.
  - Log transformations: `log_price`, `log_living_area`.
  - `price_per_sqft_calc` and `is_pacific` (timezone indicator).

- **Feature Selection:**
  - Focused on most informative variables: beds, property_age, total_baths_final, living_area_sqft, lot_size_acres_calc, price_per_sqft_calc, is_pacific.

- **Modeling:**
  - Trained three models:
    - Linear Regression
    - Random Forest Regressor
    - XGBoost Regressor
  - Evaluation metrics used:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - R² Score
    - ROC AUC Score

- **Key Observations:**
  - Random Forest performed the best.
  - XGBoost provided strong generalization.
  - Linear Regression served as baseline but struggled with complex relationships.

---

## Milestone 3: Model Evaluation, Dashboard Development, and Final Interpretation

- **Detailed Model Evaluation:**
  - Explained MAE vs RMSE trade-off.
  - Residuals analysis showed mild heteroscedasticity for high-end properties.
  - Bias-variance tradeoff discussed for Random Forest vs XGBoost.
  - Model complexity vs interpretability was assessed, highlighting Random Forest's and XGBoost's advantages and challenges.
  - ROC Curve plotted and interpreted to validate model classification capability.

- **Bias and Limitations Discussion:**
  - Discussed geographic sampling bias, missing real-world influencing variables (like crime rates, school ratings, weather factors), and urban/rural representation imbalances.

- **Dashboard Development (Streamlit):**
  - Sidebar filters: State selection, growth rate, future year forecast.
  - KPIs: Average price, lot size, days on market.
  - Visualizations:
    - Histogram of prices
    - Living area vs price scatter plot
    - Top cities by average price (bar chart)
    - Property type distribution (pie chart)
    - Days on market by city (boxplot)
    - Median price by year built (line chart)
    - Interactive property map (Mapbox)
    - Predicted vs actual price plot
    - Residuals distribution (violin plot, heatmap)
    - Residuals vs Predicted Price plot
    - Predicted future prices map with adjustable growth rate and forecast years

- **Prediction Insights Section:**
  - Created an advanced prediction panel showing detailed error distributions.
  - Future forecasted prices mapped geographically to simulate property appreciation over time.

- **Report Generation:**
  - Model evaluation reports automatically generated as PDFs using ReportLab.

- **Future Work Suggestions:**
  - Incorporate external datasets (weather, crime rates, schools, transportation proximity).
  - Explore SHAP-based explainability techniques.
  - Consider building separate models for luxury real estate segments to mitigate heteroscedasticity.

---

## Tech Stack

| Category                 | Technologies / Libraries                                              | Purpose |
|---------------------------|------------------------------------------------------------------------|---------|
| Programming Language      | Python 3.x                                                             | Main language for development |
| Data Manipulation         | pandas, numpy                                                          | Data cleaning, transformation, feature engineering |
| Machine Learning          | scikit-learn, XGBoost                                                  | Model building, training, evaluation |
| Visualization             | matplotlib, seaborn, plotly.express, plotly.graph_objects             | Static and interactive data visualizations |
| Dashboard Development     | Streamlit                                                             | Building the interactive dashboard |
| Report Generation         | ReportLab                                                             | PDF export of model evaluation |
| Utility Libraries         | os, json                                                              | File system and data management |

---