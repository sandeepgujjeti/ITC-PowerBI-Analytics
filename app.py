import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

# ✅ Load dataset safely
try:
    df = pd.read_csv("ITC_Products_Sales_Data.csv", parse_dates=["Date"])
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# ✅ Ensure "Date" is correct
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")  # Convert invalid to NaT
df.dropna(subset=["Date"], inplace=True)  # Remove rows with invalid dates
df["Day"] = df["Date"].dt.day_name()  # Convert to weekday name (Monday, Tuesday...)

# ✅ Fill missing values
df["Revenue"].fillna(df["Revenue"].median(), inplace=True)
df["Sales_Quantity"].fillna(df["Sales_Quantity"].median(), inplace=True)
df["Customer_Rating"].fillna(df["Customer_Rating"].mean(), inplace=True)

# Streamlit UI
st.title("📊 ITC Products Sales Dashboard")
st.sidebar.header("🔍 Filters")

st.write("### 📝 Sample Data (with Date & Day)")
st.write(df[["Date", "Day", "Product", "Revenue", "Sales_Quantity", "Customer_Rating"]].head())

# ✅ Sales Trend
st.write("### 📈 Sales Trend Over Time")
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x="Date", y="Revenue", data=df, ci=None, ax=ax ,color="#FFA235")
plt.xticks(rotation=45)
st.pyplot(fig)
plt.close(fig)

# ✅ Revenue by Day of the Week
st.write("### 📅 Revenue by Day of the Week")
day_revenue = df.groupby("Day")["Revenue"].sum().reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=day_revenue.index, y=day_revenue.values, ax=ax , palette=["#333453","#456123","#e4a3d2"])
plt.xticks(rotation=45)
st.pyplot(fig)
plt.close(fig)

# ✅ Top Selling Products
top_products = df.groupby("Product")["Revenue"].sum().sort_values(ascending=False)
st.write("### 🔝 Top Selling ITC Products")
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=top_products.index, y=top_products.values, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)
plt.close(fig)

# ✅ Customer Ratings
st.write("### 🌟 Customer Ratings of ITC Products")
fig, ax = plt.subplots(figsize=(10, 5))
palette = ["#FF9999", "#66B3FF", "#99FF99", "#FFCC99"]
# sns.boxplot(x="Product", y="Customer_Rating", data=df, ax=ax,palette=palette)
sns.boxplot(x="Product", y="Customer_Rating", data=df, ax=ax,palette=["#a1b2c9","#a2b45d"])
plt.xticks(rotation=45)
st.pyplot(fig)
plt.close(fig)

# ✅ Sales Forecasting using LSTM
df["DayOfYear"] = df["Date"].dt.dayofyear
X = np.array(df[["DayOfYear"]]).reshape(-1, 1)
y = np.array(df["Revenue"]).reshape(-1, 1)

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Reshape for LSTM (Fix Input Shape Error)
X_train = X_train.reshape((X_train.shape[0], 1, 1))
X_test = X_test.reshape((X_test.shape[0], 1, 1))

# ✅ Build & Train LSTM Model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(1, 1)),
    LSTM(50, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, verbose=0)

y_pred = model.predict(X_test)

# ✅ Plot Predictions
st.write("### 🤖 Sales Forecasting Using LSTM")
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(X_test[:, 0, 0], y_test, color="blue", label="Actual Revenue")
ax.plot(X_test[:, 0, 0], y_pred, color="red", linewidth=10, label="Predicted Revenue")
plt.xlabel("Day of Year")
plt.ylabel("Revenue")
plt.title("📊 Sales Forecasting Using LSTM")
plt.legend()
st.pyplot(fig)
plt.close(fig)

# ✅ Customer Segmentation using K-Means
df_segment = df[["Sales_Quantity", "Revenue"]].fillna(0)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  # Fix warning
df["Customer_Segment"] = kmeans.fit_predict(df_segment)

st.write("### 🏷️ Customer Segmentation Using K-Means")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=df["Sales_Quantity"], y=df["Revenue"], hue=df["Customer_Segment"], palette="viridis", ax=ax)
plt.xlabel("Sales Quantity")
plt.ylabel("Revenue")
plt.title("🔹 Customer Segmentation Using K-Means")
st.pyplot(fig)
plt.close(fig)
