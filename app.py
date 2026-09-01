import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title("Daily Orders Prediction")

st.write("Enter today's business information and let our Machine Learning Model"
         "predict the expected number of orders.")



df = pd.read_csv("daraz_daily_orders_synthetic.csv")

feature_columns = [
    "Website_Visitors",
    "Ad_Spend_PKR",
    "Discount_Percent",
    "Weekend"
]

X = df[feature_columns]

y = df["Daily_Orders"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42   
)

model = LinearRegression()

model.fit(X_train,y_train)


st.sidebar.header("Enter Business Information")

website_visitors = st.sidebar.slider(
    "Website Visitors",
    min_value=500,
    max_value=7000,
    value=3500,
    step=100
)

ad_spend = st.sidebar.slider(
    "Ad Spend (PKR)",
    min_value = 0,
    max_value=30000,
    value=15000,
    step=1000
)

discount = st.sidebar.slider(
    "Discount Percent",
    min_value=0,
    max_value=40,
    value=15,
    step=1
)

weekend_choice = st.sidebar.selectbox(
    "Is it a weekend?",
    ["No", "Yes"]
)

if weekend_choice == "Yes":
    weekend = 1
else:
    weekend = 0 
 
st.subheader("Your Selected Information")

st.write("Website Visitors :",website_visitors)

st.write("Ad Spend :", ad_spend, "PKR")

st.write("Discount :", discount, "%")

st.write("Weekend :", weekend_choice)


## Prediction Button

if st.button("Predict Daily Orders"):

    new_day = pd.DataFrame({
        "Website_Visitors": [website_visitors],
        "Ad_Spend_PKR" : [ad_spend],
        "Discount_Percent" : [discount],
        "Weekend" : [weekend]
    })

    predicted_orders = model.predict(new_day)[0]

    predicted_orders = round(predicted_orders)

    st.success(
        f"Expected Daily Orders: {predicted_orders}"
    )