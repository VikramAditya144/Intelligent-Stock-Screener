import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from dotenv import load_dotenv

# Set page config first - this must be the first Streamlit command
st.set_page_config(page_title="Branch Analytics", layout="wide", page_icon="📊")

# Load environment variables
load_dotenv()
GEMINI_API_KEY = "AIzaSyAGtLtFk7XcqWZQ5jKX3jJBO7EpN1W6_Do"

# Custom CSS for Branch International branding
st.markdown("""
<style>
    .main {background-color: #F5F9FC;}
    h1 {color: #003D7D;}
    h2 {color: #00A8E0;}
    .st-bb {background-color: white;}
    .st-at {background-color: #003D7D;}
    .css-1aumxhk {color: #003D7D;}
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class LoanAnalytics:
    def __init__(self):
        self.metric_mappings = {
            "Loan Amount": "Loan_Amount",
            "Interest Rate": "Interest_Rate",
            "Credit Score": "Credit_Score",
            "Monthly Income": "Monthly_Income",
            "Debt-to-Income Ratio": "Debt_to_Income_Ratio",
            "Employment Length": "Employment_Length_Years",
            "Risk Score": "Risk_Score",
            "Days Past Due": "Days_Past_Due",
            "Previous Loans": "Previous_Loans",
            "Previous Defaults": "Previous_Defaults"
        }
        
        self.categorical_filters = {
            "Country": ["India", "Nigeria", "USA"],
            "Loan_Type": ["Personal", "Business", "Education", "Emergency"],
            "Loan_Status": ["Active", "Completed", "Defaulted", "Late"],
            "Current_Employment_Status": ["Employed", "Self-Employed", "Business Owner"]
        }

    def parse_natural_language(self, query: str) -> dict:
        """Convert natural language query to filters using AI"""
        prompt = f"""
Convert this natural language query into precise loan data filters. Return ONLY JSON format:
{{
    "numerical_conditions": list of "column operator value",
    "categorical_filters": {{"column": [exact_values]}}
}}

Key Columns and Values:
- Loan Types (Loan_Type): ["Personal", "Business", "Education", "Emergency"]
- Countries (Country): ["India", "Nigeria", "USA"]
- Loan Status (Loan_Status): ["Active", "Completed", "Defaulted", "Late"]
- Employment Status (Current_Employment_Status): ["Employed", "Self-Employed", "Business Owner"]
- Numerical Columns: Loan_Amount, Interest_Rate, Credit_Score, Monthly_Income, Debt_to_Income_Ratio, 
  Employment_Length_Years, Risk_Score, Days_Past_Due, Previous_Loans, Previous_Defaults

Rules:
1. Map location references (city/region) to Country column
2. Use exact column names and categorical values as listed
3. Convert relative terms to numerical values:
   - "high" -> > 75th percentile
   - "low" -> < 25th percentile
   - "recent" = last 30 days
4. Support multiple conditions with AND logic

Examples:
1. "Emergency loans in India under $1000" → 
   {{
    "numerical_conditions": ["Loan_Amount < 1000"],
    "categorical_filters": {{"Loan_Type": ["Emergency"], "Country": ["India"]}}
   }}

2. "High-risk personal loans from USA" →
   {{
    "numerical_conditions": ["Risk_Score > 75"],
    "categorical_filters": {{"Loan_Type": ["Personal"], "Country": ["USA"]}}
   }}

3. "Active business loans greater than 1000 dollars in Nigeria" →
   {{
    "numerical_conditions": ["Loan_Amount > 1000"],
    "categorical_filters": {{"Loan_Type": ["Business"], "Loan_Status": ["Active"] , "Country": ["Nigeria"]}}
   }}

Current Query: {query}
"""
        
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return eval(response.text)

    def create_geo_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create geographical distribution map"""
        country_data = df.groupby('Country').agg({
            'Customer_ID': 'count',
            'Loan_Amount': 'sum'
        }).reset_index()
        
        country_data.columns = ['Country', 'Total_Loans', 'Total_Amount']
        
        fig = px.choropleth(country_data,
                           locations="Country",
                           locationmode="country names",
                           color="Total_Amount",
                           hover_name="Country",
                           hover_data=["Total_Loans"],
                           color_continuous_scale="Blues",
                           title="Loan Distribution by Country")
        fig.update_layout(geo=dict(showframe=False))
        return fig

    def create_loan_health(self, df: pd.DataFrame) -> go.Figure:
        """Create loan health dashboard"""
        fig = go.Figure()
        
        # Add risk score distribution
        fig.add_trace(go.Histogram(
            x=df['Risk_Score'],
            name='Risk Distribution',
            opacity=0.7,
            marker_color='#003D7D'
        ))
        
        # Add repayment status
        status_counts = df['Loan_Status'].value_counts()
        fig.add_trace(go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            name="Loan Status",
            hole=0.4,
            domain={'x': [0.7, 1], 'y': [0, 0.5]}
        ))
        
        fig.update_layout(
            title="Loan Portfolio Health Dashboard",
            barmode='overlay',
            height=600
        )
        return fig

    def generate_insights(self, df: pd.DataFrame) -> str:
        """Generate automated insights using AI"""
        prompt = f"""
        Analyze this loan portfolio data and provide key insights in markdown format:
        - Portfolio composition by country and loan type
        - Risk distribution patterns
        - Default rate analysis
        - 3 actionable recommendations
        
        Data Summary:
        {df.describe().to_string()}
        
        Format with ## headings and bullet points. Use professional financial terminology.
        """
        
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text

def main():
    st.title("📈 Branch International Loan Analytics")
    st.markdown("### Data-Driven Financial Inclusion Insights")
    
    # Configure Gemini API
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Initialize analytics engine
    if 'analytics' not in st.session_state:
        st.session_state.analytics = LoanAnalytics()
    
    # Sidebar for data upload
    with st.sidebar:
        st.header("Data Management")
        uploaded_file = st.file_uploader("Upload Loan Portfolio CSV", type=["csv"])
        st.markdown("### AI Command Examples:")
        st.code("Show high-risk loans from Nigeria\nFilter loans over $5000 with late payments\nDisplay education loans in India")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        
        # Create main tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🌍 Overview", "🔍 Data Explorer", "🤖 AI Assistant", "📋 Raw Data"])
        
        with tab1:  # Overview Tab
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Portfolio Snapshot")
                st.markdown(f"""
                <div class="metric-card">
                    <h3>${df['Loan_Amount'].sum():,.2f}</h3>
                    <p>Total Loan Portfolio</p>
                </div>
                """, unsafe_allow_html=True)
                
                cols = st.columns(3)
                metrics = [
                    ("Active Loans", len(df[df['Loan_Status'] == 'Active'])),
                    ("Default Rate", f"{len(df[df['Loan_Status'] == 'Defaulted'])/len(df)*100:.1f}%"),
                    ("Avg Risk Score", f"{df['Risk_Score'].mean():.0f}")
                ]
                for i, (label, value) in enumerate(metrics):
                    cols[i].metric(label, value)
                
            with col2:
                st.subheader("Geographical Distribution")
                st.plotly_chart(st.session_state.analytics.create_geo_distribution(df), use_container_width=True)
            
            st.subheader("Portfolio Health Dashboard")
            st.plotly_chart(st.session_state.analytics.create_loan_health(df), use_container_width=True)
        
        with tab2:  # Data Explorer
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Quick Filters")
                selected_metric = st.selectbox("Select Metric", list(st.session_state.analytics.metric_mappings.keys()))
                mapped_metric = st.session_state.analytics.metric_mappings[selected_metric]
                threshold = st.slider("Threshold Value", float(df[mapped_metric].min()), float(df[mapped_metric].max()))
                filtered_df = df[df[mapped_metric] > threshold]
            
            with col2:
                st.subheader("Metric Distribution")
                fig = px.histogram(filtered_df, x=mapped_metric, 
                                 color='Country', nbins=50,
                                 title=f"{selected_metric} Distribution",
                                 color_discrete_sequence=['#003D7D', '#00A8E0', '#7FD6FF'])
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Dynamic Cross-Analysis")
            x_metric = st.selectbox("X-Axis", list(st.session_state.analytics.metric_mappings.keys()), key="x_axis")
            y_metric = st.selectbox("Y-Axis", list(st.session_state.analytics.metric_mappings.keys()), key="y_axis")
            x_axis = st.session_state.analytics.metric_mappings[x_metric]
            y_axis = st.session_state.analytics.metric_mappings[y_metric]
            
            fig = px.scatter(df, x=x_axis, y=y_axis, color='Loan_Status',
                           hover_name='Customer_ID', 
                           color_discrete_sequence=['#003D7D', '#00A8E0', '#FF4B4B'])
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:  # AI Assistant
            col1, col2 = st.columns([2, 3])
            with col1:
                st.subheader("Natural Language Query")
                query = st.text_area("Ask about your data:", height=150,
                                   placeholder="Example: Show business loans in Nigeria with Risk Score > 150")
                
                if st.button("Execute Query"):
                    with st.spinner("Processing with AI..."):
                        try:
                            filters = st.session_state.analytics.parse_natural_language(query)
                            filtered_df = df.copy()
                            
                            # Apply categorical filters
                            for col, values in filters['categorical_filters'].items():
                                if values:
                                    filtered_df = filtered_df[filtered_df[col].isin(values)]
                            
                            # Apply numerical conditions
                            for condition in filters['numerical_conditions']:
                                col, op, val = condition.split()
                                filtered_df = filtered_df.query(f"{col} {op} {val}")
                            
                            st.session_state.filtered_df = filtered_df
                        except Exception as e:
                            st.error(f"Error processing query: {str(e)}")
            
            with col2:
                st.subheader("Query Results")
                if 'filtered_df' in st.session_state:
                    st.dataframe(st.session_state.filtered_df, use_container_width=True)
                    
                    cols = st.columns(3)
                    cols[0].metric("Matched Loans", len(st.session_state.filtered_df))
                    cols[1].metric("Average Risk", f"{st.session_state.filtered_df['Risk_Score'].mean():.1f}")
                    cols[2].metric("Default Rate", 
                                 f"{len(st.session_state.filtered_df[st.session_state.filtered_df['Loan_Status'] == 'Defaulted'])/len(st.session_state.filtered_df)*100:.1f}%")
                    
                    st.plotly_chart(px.box(st.session_state.filtered_df, y='Interest_Rate', 
                                         color='Country', title="Interest Rate Distribution",
                                         color_discrete_sequence=['#003D7D', '#00A8E0', '#7FD6FF']), 
                                  use_container_width=True)
            
            st.subheader("AI-Powered Insights")
            if st.button("Generate Portfolio Insights"):
                with st.spinner("Analyzing..."):
                    insights = st.session_state.analytics.generate_insights(df)
                    st.markdown(insights)
        
        with tab4:  # Raw Data
            st.subheader("Full Loan Portfolio Data")
            st.dataframe(df, use_container_width=True)
            st.download_button("Download Filtered Data", 
                             df.to_csv(index=False).encode('utf-8'),
                             "branch_loans.csv",
                             "text/csv")

if __name__ == "__main__":
    main()