"""OPTIONAL DASHBOARD(Streamlit)"""
import streamlit as st
import pandas as pd

st.title("Trading Behavior Dashboard")

data = pd.read_csv("merged_data.csv")

st.write("Sample Data")
st.dataframe(data.head())

st.bar_chart(data.groupby('sentiment_label')['pnl'].mean())
"""
