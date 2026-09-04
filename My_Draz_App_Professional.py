import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Daraz Daily Orders Predictor",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Daraz Daily Orders Prediction")
st.caption(" Machine Learning dashboard using Linear Regression")

st.write(
    "Enter business information in the sidebar to estimate daily orders. "
    "The dashboard also shows model performance, feature impact, dataset statistics, "
    "and a downloadable prediction."
)

@st.cache_data
def load_data():
    return pd.read_csv("daraz_daily_orders_synthetic.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Dataset not found. Keep 'daraz_daily_orders_synthetic.csv' "
        "in the same folder as this Streamlit app."
    )
    st.stop()

feature_columns = [
    "Website_Visitors",
    "Ad_Spend_PKR",
    "Discount_Percent",
    "Weekend"
]
target_column = "Daily_Orders"

required = feature_columns + [target_column]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {', '.join(missing)}")
    st.stop()

X = df[feature_columns]
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.sidebar.header("📊 Business Information")

visitor_min = int(df["Website_Visitors"].min())
visitor_max = int(df["Website_Visitors"].max())
visitor_default = int(df["Website_Visitors"].median())

spend_min = int(df["Ad_Spend_PKR"].min())
spend_max = int(df["Ad_Spend_PKR"].max())
spend_default = int(df["Ad_Spend_PKR"].median())

discount_min = int(df["Discount_Percent"].min())
discount_max = int(df["Discount_Percent"].max())
discount_default = int(df["Discount_Percent"].median())

website_visitors = st.sidebar.slider(
    "Website Visitors",
    visitor_min,
    visitor_max,
    visitor_default,
    max(1, int((visitor_max - visitor_min) / 50))
)

ad_spend = st.sidebar.slider(
    "Ad Spend (PKR)",
    spend_min,
    spend_max,
    spend_default,
    max(1, int((spend_max - spend_min) / 30))
)

discount = st.sidebar.slider(
    "Discount Percent",
    discount_min,
    discount_max,
    discount_default,
    1
)

weekend_choice = st.sidebar.selectbox("Is it a weekend?", ["No", "Yes"])
weekend = 1 if weekend_choice == "Yes" else 0

new_day = pd.DataFrame({
    "Website_Visitors": [website_visitors],
    "Ad_Spend_PKR": [ad_spend],
    "Discount_Percent": [discount],
    "Weekend": [weekend]
})

predicted_orders = max(0, float(model.predict(new_day)[0]))
avg_orders = df[target_column].mean()
difference = predicted_orders - avg_orders
percentage_difference = (difference / avg_orders * 100) if avg_orders else 0

st.subheader("📌 Current Business Scenario")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Website Visitors", f"{website_visitors:,}")
c2.metric("Ad Spend", f"PKR {ad_spend:,}")
c3.metric("Discount", f"{discount}%")
c4.metric("Weekend", weekend_choice)

st.divider()

st.subheader("🎯 Prediction")
p1, p2 = st.columns([2, 1])

with p1:
    st.success(f"### Expected Daily Orders: {round(predicted_orders):,}")
    st.info(
        f"The model estimates approximately **{predicted_orders:.1f} orders** "
        "for the selected scenario."
    )

with p2:
    st.metric(
        "vs Dataset Average",
        f"{difference:+.1f} orders",
        f"{percentage_difference:+.1f}%"
    )

st.subheader("🤖 Model Performance")
m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{mae:.2f}")
m2.metric("RMSE", f"{rmse:.2f}")
m3.metric("R² Score", f"{r2:.3f}")

st.caption(
    "MAE = average absolute prediction error. RMSE gives more weight to large errors. "
    "R² measures the proportion of target variation explained by the model."
)

st.subheader("📈 Feature Impact")
coef_df = pd.DataFrame({
    "Feature": feature_columns,
    "Coefficient": model.coef_
})
coef_df["Absolute Impact"] = coef_df["Coefficient"].abs()
coef_df = coef_df.sort_values("Absolute Impact", ascending=False)

st.bar_chart(coef_df.set_index("Feature")["Coefficient"])

st.write(
    "Positive coefficients indicate an increase in predicted orders when that feature "
    "increases, while negative coefficients indicate the opposite, with other inputs held constant."
)

st.subheader("🔍 Actual vs Predicted Orders")
comparison_df = pd.DataFrame({
    "Actual Orders": y_test.values,
    "Predicted Orders": np.round(y_pred, 1)
}).reset_index(drop=True)
st.line_chart(comparison_df)

st.subheader("📊 Dataset Overview")
d1, d2, d3, d4 = st.columns(4)
d1.metric("Total Records", f"{len(df):,}")
d2.metric("Training Records", f"{len(X_train):,}")
d3.metric("Testing Records", f"{len(X_test):,}")
d4.metric("Average Orders", f"{avg_orders:.1f}")

with st.expander("View Dataset"):
    st.dataframe(df, use_container_width=True)

result_df = new_day.copy()
result_df["Predicted_Daily_Orders"] = round(predicted_orders)

st.download_button(
    "⬇️ Download Prediction as CSV",
    result_df.to_csv(index=False),
    "daily_orders_prediction.csv",
    "text/csv"
)

st.subheader("💡 Business Insight")
recommendations = []

if website_visitors < df["Website_Visitors"].median():
    recommendations.append("Website traffic is below the dataset median; consider improving marketing or SEO.")
if ad_spend < df["Ad_Spend_PKR"].median():
    recommendations.append("Ad spend is below the dataset median; additional advertising may improve demand.")
if discount < df["Discount_Percent"].median():
    recommendations.append("The discount is below the dataset median; test a higher discount while monitoring profit.")
if weekend == 1:
    recommendations.append("The prediction includes the weekend effect learned by the model.")

if recommendations:
    for item in recommendations:
        st.write(f"• {item}")
else:
    st.write("• The selected inputs are around or above the dataset's typical values.")

st.warning(
    "⚠️ This prediction is an estimate, not a guarantee. Real orders can also depend "
    "on seasonality, competition, product availability, platform changes, and other factors "
    "not included in this model."
)
