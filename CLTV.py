# Customer Lifetime Value Analysis using Online Retail Dataset

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings

# Configure pandas display settings
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Load dataset
df_ = pd.read_excel("/Users/erdinc/PycharmProjects/pythonProject3/CRM/online_retail_II.xlsx",
                    sheet_name="Year 2009-2010")

# Copy dataset to preserve original data
df = df_.copy()

# Data preprocessing
# Remove canceled invoices (identified by "C")
df = df[~df["Invoice"].str.contains("C", na=False)]

# Keep only positive quantity transactions
df = df[df["Quantity"] > 0]

# Drop rows with missing values
df.dropna(inplace=True)

# Create TotalPrice column
df["TotalPrice"] = df["Quantity"] * df["Price"]

# Calculate customer-based summary metrics
cltv_df = df.groupby("Customer ID").agg({
    "Invoice": lambda x: x.nunique(),         # Number of unique transactions
    "Quantity": lambda x: x.sum(),            # Total quantity purchased
    "TotalPrice": lambda x: x.sum()           # Total revenue
})

cltv_df.columns = ["total_transactions", "total_units", "total_price"]

# Calculate Average Order Value (AOV)
cltv_df["average_order_value"] = cltv_df["total_price"] / cltv_df["total_transactions"]

# Calculate Purchase Frequency
cltv_df["purchase_frequency"] = cltv_df["total_transactions"] / cltv_df.shape[0]

# Calculate Repeat Rate and Churn Rate
repeat_rate = cltv_df[cltv_df["total_transactions"] > 1].shape[0] / cltv_df.shape[0]
churn_rate = 1 - repeat_rate

# Assume a profit margin of 10%
cltv_df["profit_margin"] = cltv_df["total_price"] * 0.10

# Calculate Customer Value
cltv_df["customer_value"] = cltv_df["average_order_value"] * cltv_df["purchase_frequency"]

# Calculate Customer Lifetime Value (CLTV)
cltv_df["cltv"] = (cltv_df["customer_value"] / churn_rate) * cltv_df["profit_margin"]

# Segment customers into quartiles (A, B, C, D)
cltv_df["segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

# Review segment statistics
segment_summary = cltv_df.groupby("segment").agg(["count", "mean", "sum"])
print(segment_summary)

# Function to automate CLTV calculation
def create_cltv_df(dataframe, profit_margin_rate=0.10):
    # Data preprocessing
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe.dropna(inplace=True)

    # Calculate total price
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]

    # Aggregate data by customer
    cltv_df = dataframe.groupby("Customer ID").agg({
        'Invoice': lambda x: x.nunique(),
        'Quantity': lambda x: x.sum(),
        'TotalPrice': lambda x: x.sum()
    })

    cltv_df.columns = ['total_transactions', 'total_units', 'total_price']

    # Calculate metrics
    cltv_df['average_order_value'] = cltv_df['total_price'] / cltv_df['total_transactions']
    cltv_df['purchase_frequency'] = cltv_df['total_transactions'] / cltv_df.shape[0]

    repeat_rate = cltv_df[cltv_df['total_transactions'] > 1].shape[0] / cltv_df.shape[0]
    churn_rate = 1 - repeat_rate

    cltv_df['profit_margin'] = cltv_df['total_price'] * profit_margin_rate
    cltv_df['customer_value'] = cltv_df['average_order_value'] * cltv_df['purchase_frequency']
    cltv_df['cltv'] = (cltv_df['customer_value'] / churn_rate) * cltv_df['profit_margin']

    cltv_df["segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_df

# Execute the function with original data
df = df_.copy()
cltv_final = create_cltv_df(df)

# Display the resulting CLTV dataframe
print(cltv_final.head())