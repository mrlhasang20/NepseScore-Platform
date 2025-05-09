import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Portfolio Tracker: Company Scoring")

# Load scored data
data = pd.read_csv("../data/scored_company_data.csv")

# Sector filter
sectors = data['Sector'].unique()
selected_sector = st.selectbox("Select Sector", ["All"] + list(sectors))

# Filter data
if selected_sector != "All":
    filtered_data = data[data['Sector'] == selected_sector]
else:
    filtered_data = data

# Display data
st.write("### Company Scores")
st.dataframe(filtered_data[['Company_Name', 'Sector', 'Score']])

# Score distribution
fig = px.histogram(filtered_data, x='Score', nbins=20, title='Score Distribution')
st.plotly_chart(fig)

# Top companies
st.write("### Top 5 Companies")
top_companies = filtered_data.nlargest(5, 'Score')[['Company_Name', 'Sector', 'Score']]
st.table(top_companies)