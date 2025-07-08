import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline import main as run_pipeline

script_dir = os.path.dirname(os.path.abspath(__file__))

# Cache data loading
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Run pipeline on startup if output files are missing
if not (os.path.exists(os.path.join(script_dir, "../data/scored_company_data.csv")) and 
        os.path.exists(os.path.join(script_dir, "../data/predicted_metrics.csv")) and 
        os.path.exists(os.path.join(script_dir, "../data/final_scores.csv"))):
    st.write("Running pipeline to generate data...")
    run_pipeline()

st.title("NepseScore Platform")

#load data
scored_data = load_data(os.path.join(script_dir, "../data/scored_company_data.csv"))
predicted_data = load_data(os.path.join(script_dir, "../data/predicted_metrics.csv"))  
final_scores = load_data(os.path.join(script_dir, "../data/final_scores.csv"))

# # Load data
# scored_data = pd.read_csv("data/scored_company_data.csv")
# predicted_data = pd.read_csv("data/predicted_metrics.csv")
# final_scores = pd.read_csv("data/final_scores.csv")

# Tabs for scoring, predictions, and final scores
tab1, tab2, tab3 = st.tabs(["Current Scores", "Future Predictions", "Final Scores"])

with tab1:
    st.write("### Current Company Scores")
    sectors = scored_data['Sector'].unique()
    selected_sector = st.selectbox("Select Sector", ["All"] + list(sectors), key="current_sector")
    
    if selected_sector != "All":
        filtered_data = scored_data[scored_data['Sector'] == selected_sector]
    else:
        filtered_data = scored_data
    
    st.dataframe(filtered_data[['Company_Name', 'Sector', 'Score']])
    
    fig = px.histogram(filtered_data, x='Score', nbins=20, title='Current Score Distribution')
    st.plotly_chart(fig)
    
    st.write("### Top 5 Companies (Current)")
    top_companies = filtered_data.nlargest(5, 'Score')[['Company_Name', 'Sector', 'Score']]
    st.table(top_companies)

with tab2:
    st.write("### Future Predictions")
    companies = predicted_data['Company_Name'].unique()
    selected_company = st.selectbox("Select Company", companies, key="prediction_company")
    
    company_preds = predicted_data[predicted_data['Company_Name'] == selected_company]
    
    fig = go.Figure()
    for metric in ['EPS', 'P_E', 'Dividend_Yield']:
        fig.add_trace(go.Scatter(x=company_preds['Period'], y=company_preds[metric], name=metric))
    
    fig.update_layout(title=f'Predicted Metrics for {selected_company}', xaxis_title='Period (Months)', yaxis_title='Value')
    st.plotly_chart(fig)
    
    st.write("### Predicted Values")
    st.dataframe(company_preds)

with tab3:
    st.write("### Final Company Scores")
    final_sectors = final_scores['Sector'].unique()
    final_selected_sector = st.selectbox("Select Sector", ["All"] + list(final_sectors), key="final_sector")
    
    if final_selected_sector != "All":
        final_filtered_data = final_scores[final_scores['Sector'] == final_selected_sector]
    else:
        final_filtered_data = final_scores
    
    st.dataframe(final_filtered_data[['Company_Name', 'Sector', 'Score', 'Predicted_Score', 'Final_Score']])
    
    fig = px.histogram(final_filtered_data, x='Final_Score', nbins=20, title='Final Score Distribution')
    st.plotly_chart(fig)
    
    st.write("### Top 5 Companies (Final)")
    top_final_companies = final_filtered_data.nlargest(5, 'Final_Score')[['Company_Name', 'Sector', 'Final_Score']]
    st.table(top_final_companies)
