import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def load_data(path):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path, parse_dates=["snapshot_date", "data_gen_date"],nrows=100000)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")

    return df

def feature_engineering(df):
    print("=== Starting Feature Engineering ===")
    df = df.copy()
    df['property_age'] = df['snapshot_date'].dt.year - df['year_built']
    df['total_baths_final'] = (
        df['homeData.bathInfo.computedFullBaths'] +
        0.5 * df['homeData.bathInfo.computedPartialBaths']
    )
    df['lot_size_acres_calc'] = df['lot_size_sqft'] / 43560
    df['log_price'] = np.log1p(df['price'])
    df['log_living_area'] = np.log1p(df['living_area_sqft'])
    df['price_per_sqft_calc'] = df['price'] / df['living_area_sqft']
    df['is_pacific'] = df['timezone'].str.contains('Pacific').astype(int)
    print("Feature engineering completed.\n")
    return df

def train_model(df):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import (
        mean_absolute_error, mean_squared_error, r2_score,
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, roc_curve
    )
    # had to import libraries here as it was giving error
    import json, os
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    features = ['beds', 'property_age', 'total_baths_final',
                'living_area_sqft', 'lot_size_acres_calc',
                'price_per_sqft_calc', 'is_pacific']
    X = df[features]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    y_test_class = (y_test > y.median()).astype(int)
    y_pred_class = (y_pred > y.median()).astype(int)

    results = {
        'RandomForest': {
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred),
            'Accuracy': accuracy_score(y_test_class, y_pred_class),
            'Precision': precision_score(y_test_class, y_pred_class),
            'Recall': recall_score(y_test_class, y_pred_class),
            'F1': f1_score(y_test_class, y_pred_class),
            'ROC AUC': roc_auc_score(y_test_class, y_pred)
        }
    }

    fpr, tpr, _ = roc_curve(y_test_class, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, label=f'RandomForest (AUC = {roc_auc_score(y_test_class, y_pred):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve - Random Forest')
    plt.legend()
    os.makedirs('../Reports/Graphs', exist_ok=True)
    plt.savefig('../Reports/Graphs/RandomForest_roc_curve.png')
    plt.close()

    with open('../Data/model_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Add predictions to a new DataFrame
    test_with_preds = X_test.copy()
    test_with_preds['actual_price'] = y_test.values
    test_with_preds['predicted_price'] = y_pred
    test_with_preds['index'] = y_test.index  # Add this
    test_with_preds.set_index('index', inplace=True)  # And this

    print('Random Forest model training and evaluation complete.')

    return test_with_preds, rf

# this method has several graphs logic which shows insights on the actual data as well as the predicted data. This dashboard is built using streamlit.
def run_dashboard(df, pred_df):
    import plotly.express as px
    import plotly.graph_objects as go
    st.set_page_config(page_title="Real Estate Dashboard", layout="wide")
    st.title("📊 Real Estate Listings Dashboard")

    with st.sidebar:
        st.header("Filters")
        states = df['state'].unique().tolist()
        sel_state = st.selectbox("Select State", ["All States"] + states)

        st.markdown("---")
        st.subheader("Price Forecast Settings")
        growth_rate = st.slider("Growth Rate (%)", 1, 10, 4)
        forecast_years = st.slider("Years into Future", 1, 20, 10)


    if sel_state != "All States":
        df = df[df['state'] == sel_state]

    st.subheader(f"Overview for {'All States' if sel_state == 'All States' else sel_state}")
    
    # Key Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Price", f"${df['price'].mean():,.0f}")
    col2.metric("Median Lot Size (Acres)", f"{df['lot_size_acres_calc'].median():.2f}")
    col3.metric("Average Days on Market", f"{df['days_on_market'].mean():.1f}")

    # 1. Price Distribution
    st.plotly_chart(
        px.histogram(df, x='price', nbins=50, title="1. Price Distribution"),
        use_container_width=True
    )

    # 2. Scatter Plot - Living Area vs Price
    st.plotly_chart(
        px.scatter(df, x='living_area_sqft', y='price', color='city', title="3. Price vs Living Area"),
        use_container_width=True
    )

    # 3. Scatter Plot - Predicted vs Actual Price
    st.markdown("---")
    st.header("🔍 Prediction Insights")

    st.subheader("3. Predicted vs Actual Price")
    import plotly.graph_objects as go

    fig_pred_errorbars = go.Figure()
    fig_pred_errorbars.add_trace(go.Scatter(
        x=pred_df['actual_price'],
        y=pred_df['predicted_price'],
        mode='markers',
        marker=dict(color='green', size=6, opacity=0.5),
        name='Predictions'
    ))
    fig_pred_errorbars.add_trace(go.Scatter(
        x=[pred_df['actual_price'].min(), pred_df['actual_price'].max()],
        y=[pred_df['actual_price'].min(), pred_df['actual_price'].max()],
        mode='lines',
        line=dict(color='yellow', dash='dash'),
        name='Perfect Prediction'
    ))
    fig_pred_errorbars.update_layout(
        title="📉Predicted vs Actual Prices",
        xaxis_title="Actual Price",
        yaxis_title="Predicted Price",
        height=500
    )
    st.plotly_chart(fig_pred_errorbars, use_container_width=True)

    # 4. Prediction Error Distribution two graphs - violin and heatmap
    st.subheader("5. Prediction Error Distribution")

    # Calculate residuals and bins if not already done
    pred_df['residual'] = pred_df['actual_price'] - pred_df['predicted_price']
    pred_df['abs_error'] = np.abs(pred_df['residual'])

    # Create bins
    pred_df['price_bin'] = pd.qcut(pred_df['actual_price'], q=5, duplicates='drop')
    pred_df['size_bin'] = pd.qcut(pred_df['living_area_sqft'], q=5, duplicates='drop')

    # Group for heatmap
    heat_df = pred_df.groupby(['price_bin', 'size_bin'])['abs_error'].mean().reset_index()
    heat_df['price_bin'] = heat_df['price_bin'].astype(str)
    heat_df['size_bin'] = heat_df['size_bin'].astype(str)
    heat_df_pivot = heat_df.pivot(index='price_bin', columns='size_bin', values='abs_error')

    # Layout: left = violin, right = heatmap
    col1, col2 = st.columns(2)

    with col1:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig_violin, ax = plt.subplots(figsize=(6, 4))
        sns.violinplot(x=pred_df['residual'], inner='box', linewidth=1.2, ax=ax)
        ax.axvline(0, color='red', linestyle='--')
        ax.set_title("Violin Plot of Prediction Residuals")
        ax.set_xlabel("Residual (Actual - Predicted)")
        ax.grid(True, axis='x', linestyle='--', alpha=0.3)
        st.pyplot(fig_violin, use_container_width=True)

    with col2:
        import plotly.express as px
        fig_heatmap = px.imshow(
            heat_df_pivot,
            text_auto=".2f",
            labels=dict(x="Living Area (bin)", y="Price (bin)", color="Avg Error"),
            title="Heatmap of Avg Absolute Error"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)


    df_future = pred_df.copy()
    df_future['property_age'] += 10  # simulate property aging 10 more years

    predicted_now = model.predict(df_future[[
        'beds', 'property_age', 'total_baths_final',
        'living_area_sqft', 'lot_size_acres_calc',
        'price_per_sqft_calc', 'is_pacific'
    ]])

    # Apply growth rate adjustment (CAGR)
    df_future['predicted_future_price'] = predicted_now * ((1 + growth_rate / 100) ** forecast_years)


    st.subheader("6. Residuals vs Predicted Prices")
    fig_residuals = go.Figure()
    fig_residuals.add_trace(go.Scatter(
        x=pred_df['predicted_price'],
        y=pred_df['residual'],
        mode='markers',
        marker=dict(color='green', size=6, opacity=0.6),
        name='Residuals'
    ))
    fig_residuals.add_trace(go.Scatter(
        x=[pred_df['predicted_price'].min(), pred_df['predicted_price'].max()],
        y=[0, 0],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Zero Error Line'
    ))
    fig_residuals.update_layout(
        title="Residuals vs Predicted Price",
        xaxis_title="Predicted Price",
        yaxis_title="Residual (Actual - Predicted)",
        height=500
    )
    st.plotly_chart(fig_residuals, use_container_width=True)


    st.subheader("7. Predicted Price Map")
    if 'latitude' in pred_df.columns and 'longitude' in pred_df.columns:
        fig_pred_map = px.scatter_mapbox(
            df_future,  # Use future df instead of pred_df
            lat='latitude',
            lon='longitude',
            color='predicted_future_price',
            size='living_area_sqft',
            hover_data=['actual_price', 'predicted_future_price', 'beds', 'city'],
            title=f"Predicted Prices in {forecast_years} Years (Growth: {growth_rate}%)",
            zoom=3,
            height=600,
            color_continuous_scale='Viridis',
            range_color=[0, 2000000]  # Optional: adjust for visual scaling
        )

        fig_pred_map.update_layout(mapbox_style="open-street-map")
        fig_pred_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig_pred_map, use_container_width=True)

    # 8. Bar Chart - Average Price by City
    avg_price_city = df.groupby('city')['price'].mean().sort_values(ascending=False).head(15).reset_index()
    fig3 = px.bar(avg_price_city, x='city', y='price', title="8. Top 15 Cities by Average Price", labels={'price': 'Average Price'})
    st.plotly_chart(fig3, use_container_width=True)

    # 9. Pie Chart - Property Type Distribution
    prop_type_dist = df['property_type'].value_counts().reset_index()
    prop_type_dist.columns = ['Property Type', 'Count']
    fig4 = px.pie(prop_type_dist, values='Count', names='Property Type', title='9. Property Type Distribution')
    st.plotly_chart(fig4, use_container_width=True)

    # 10. Boxplot - Days on Market by City (top 10)
    top_cities = df['city'].value_counts().nlargest(10).index.tolist()
    dom_df = df[df['city'].isin(top_cities)]
    fig5 = px.box(dom_df, x='city', y='days_on_market', title="10. Days on Market by Top Cities")
    st.plotly_chart(fig5, use_container_width=True)

    # 11. Compare States (if All States selected)
    if sel_state == "All States":
        avg_state_price = df.groupby('state')['price'].mean().sort_values(ascending=False).reset_index()
        fig7 = px.bar(avg_state_price, x='state', y='price', title="11. Average Price by State")
        st.plotly_chart(fig7, use_container_width=True)

        st.write("Tip: Use the dropdown in the sidebar to explore a specific state's trends.")

    # 12. Interactive Map - Property Locations
    st.subheader("12. Property Map of Listings")

    color_options = ['state', 'city', 'price', 'property_type', 'days_on_market']
    selected_color = st.selectbox("Color points by:", color_options, index=color_options.index('state'))

    fig_map = px.scatter_mapbox(
        df,
        lat='latitude',
        lon='longitude',
        color=selected_color,
        hover_data=['price', 'city', 'state', 'living_area_sqft', 'property_type'],
        zoom=3,
        height=600,
        title="Interactive Map of Properties"
    )

    fig_map.update_layout(mapbox_style="open-street-map")
    fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})

    st.plotly_chart(fig_map, use_container_width=True)
    
    
    

DATA_PATH = "../Data/cleaned_real_estate_data.csv"

df = load_data(DATA_PATH)
df_fe = feature_engineering(df)
#df_fe['predicted_future_price'] = df_fe['price'] * ((1 + 0.04) ** 10)
pred_df, model = train_model(df_fe)
pred_df = pred_df.merge(
    df_fe[['latitude', 'longitude', 'city']],
    left_index=True,
    right_index=True,
    how='left'
)

run_dashboard(df_fe, pred_df)
