import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Union, Optional
from scipy import stats
import google.generativeai as genai

# Load environment variables
load_dotenv()
GEMINI_API_KEY = "AIzaSyCId9px4yRNrG9bsPEuENCIFo8t-XEERws"
genai.configure(api_key=GEMINI_API_KEY)

class StockScreener:
    def __init__(self):
        self.metric_mappings = {
            "Market Capitalization": "Market_Capitalization_B",
            "P/E Ratio": "P_E_Ratio",
            "ROE": "ROE_percent",
            "Debt-to-Equity Ratio": "Debt_to_Equity_Ratio",
            "Dividend Yield": "Dividend_Yield_percent",
            "Revenue Growth": "Revenue_Growth_percent",
            "EPS Growth": "EPS_Growth_percent",
            "Current Ratio": "Current_Ratio",
            "Gross Margin": "Gross_Margin_percent"
        }

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean column names for consistency"""
        df_cleaned = df.copy()
        column_mappings = {
            "Market Capitalization (B)": "Market_Capitalization_B",
            "P/E Ratio": "P_E_Ratio",
            "ROE (%)": "ROE_percent",
            "Debt-to-Equity Ratio": "Debt_to_Equity_Ratio",
            "Dividend Yield (%)": "Dividend_Yield_percent",
            "Revenue Growth (%)": "Revenue_Growth_percent",
            "EPS Growth (%)": "EPS_Growth_percent",
            "Current Ratio": "Current_Ratio",
            "Gross Margin (%)": "Gross_Margin_percent"
        }
        df_cleaned.rename(columns=column_mappings, inplace=True)
        return df_cleaned

    def parse_single_condition(self, condition: str) -> tuple:
        """Parse a single condition into column, operator, and value"""
        condition = condition.strip()
        for display_name, internal_name in self.metric_mappings.items():
            condition = condition.replace(display_name, internal_name)
        
        parts = condition.split()
        if len(parts) != 3:
            raise ValueError(f"Invalid condition format: {condition}")
        
        column, operator, value = parts
        return column, operator, float(value)

    def apply_conditions(self, df: pd.DataFrame, conditions_text: str) -> pd.DataFrame:
        """Apply multiple conditions joined by AND"""
        filtered_df = df.copy()
        
        conditions = [cond.strip() for cond in conditions_text.split("AND") if cond.strip()]
        
        for condition in conditions:
            try:
                column, operator, value = self.parse_single_condition(condition)
                
                if operator == ">":
                    filtered_df = filtered_df[filtered_df[column] > value]
                elif operator == "<":
                    filtered_df = filtered_df[filtered_df[column] < value]
                elif operator == ">=":
                    filtered_df = filtered_df[filtered_df[column] >= value]
                elif operator == "<=":
                    filtered_df = filtered_df[filtered_df[column] <= value]
                elif operator == "=":
                    filtered_df = filtered_df[filtered_df[column] == value]
                else:
                    st.error(f"Unsupported operator: {operator}")
                    return pd.DataFrame()
                
                if filtered_df.empty:
                    st.warning(f"No stocks match after applying condition: {condition}")
                    break
                    
            except Exception as e:
                st.error(f"Error in condition '{condition}': {str(e)}")
                return pd.DataFrame()
                
        return filtered_df

    def create_correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap for numerical columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=numeric_cols,
            y=numeric_cols,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        
        fig.update_layout(
            title="Correlation Heatmap",
            height=600,
            width=800
        )
        
        return fig

    def create_distribution_plot(self, df: pd.DataFrame, column: str) -> go.Figure:
        """Create distribution plot for a specific metric"""
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df[column],
            name="Distribution",
            nbinsx=30,
            opacity=0.75
        ))
        
        # Add KDE curve
        kde_x = np.linspace(df[column].min(), df[column].max(), 100)
        kde = stats.gaussian_kde(df[column].dropna())
        fig.add_trace(go.Scatter(
            x=kde_x,
            y=kde(kde_x) * len(df[column]) * (df[column].max() - df[column].min()) / 30,
            name="KDE",
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title=f"Distribution of {column}",
            xaxis_title=column,
            yaxis_title="Frequency",
            height=400
        )
        
        return fig

    def analyze_with_gemini(self, df: pd.DataFrame, query: str) -> str:
        """Analyze stock data using Gemini"""
        try:
            data_str = df.to_string()
            
            prompt = f"""
            You are a financial data analyst. Analyze the following stock market data based on this query:
            
            Query: {query}

            Stock Data:
            {data_str}

            Provide a professional financial analysis including:
            1. Key Market Findings
            2. Statistical Patterns
            3. Investment Insights
            4. Relevant Recommendations
            
            Format your response with clear headings and financial terminology where appropriate.
            Focus on market-relevant information and patterns in the data.
            """
            
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(prompt)
            
            if hasattr(response.candidates[0].content, 'parts'):
                return response.candidates[0].content.parts[0].text
            return "Analysis could not be generated."
            
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            return "Unable to analyze the stock data. Please refine your query."

def display_filtered_data_analysis(filtered_df: pd.DataFrame, screener: StockScreener):
    """Display all analysis for filtered data"""
    st.subheader(f"Screening Results ({len(filtered_df)} stocks)")
    st.dataframe(filtered_df, use_container_width=True)
    
    # Summary Statistics
    st.subheader("Summary Statistics")
    summary_stats = filtered_df.describe()
    st.dataframe(summary_stats, use_container_width=True)
    
    # Data Visualization Section
    st.header("Data Visualization")
    
    # Distribution Analysis
    st.subheader("Distribution Analysis")
    
    # Store the selected metric in session state if not already present
    if 'selected_metric' not in st.session_state:
        st.session_state.selected_metric = list(screener.metric_mappings.values())[0]
    
    # Update the selected metric based on selectbox
    selected_metric = st.selectbox(
        "Select metric for distribution analysis:",
        list(screener.metric_mappings.values()),
        index=list(screener.metric_mappings.values()).index(st.session_state.selected_metric)
    )
    st.session_state.selected_metric = selected_metric
    
    if selected_metric:
        dist_fig = screener.create_distribution_plot(filtered_df, selected_metric)
        st.plotly_chart(dist_fig, use_container_width=True)
    
    # Correlation Analysis
    st.subheader("Correlation Analysis")
    corr_fig = screener.create_correlation_heatmap(filtered_df)
    st.plotly_chart(corr_fig, use_container_width=True)

def main():
    st.set_page_config(page_title="Stock Screening Tool", layout="wide")
    
    st.title("Stock Screening Tool")
    st.markdown("### Advanced Stock Screening and Analysis Platform")
    
    if not GEMINI_API_KEY:
        st.error("API Key not found. Please check your environment configuration.")
        return
    
    # Initialize screener in session state
    if 'screener' not in st.session_state:
        st.session_state.screener = StockScreener()
    
    # Sidebar for file upload and metrics guide
    with st.sidebar:
        st.header("Data Upload")
        uploaded_file = st.file_uploader("Upload Stock Data CSV", type="csv")
        
        st.header("Available Metrics")
        st.markdown("""
        - Market Capitalization
        - P/E Ratio
        - ROE
        - Debt-to-Equity Ratio
        - Dividend Yield
        - Revenue Growth
        - EPS Growth
        - Current Ratio
        - Gross Margin
        """)
        
        st.header("Example Queries")
        st.markdown("""
        **Query 1: Large Cap Growth Stocks**
        ```
        Market Capitalization > 100 AND
        ROE > 15 AND
        EPS Growth > 10
        ```
        
        **Query 2: Dividend Value Stocks**
        ```
        Dividend Yield > 2 AND
        P/E Ratio < 20 AND
        Debt-to-Equity Ratio < 1
        ```
        """)
    
    # Main content area
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            df = st.session_state.screener.clean_column_names(df)
            st.session_state.df = df
            
            # Data preview
            st.header("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Create tabs for different functionalities
            tab1, tab2 = st.tabs(["Stock Screening", "Talk with CSV"])
            
            # Stock Screening Tab
            with tab1:
                st.header("Stock Screening")
                
                # Store conditions in session state if not present
                if 'conditions' not in st.session_state:
                    st.session_state.conditions = ""
                
                # Get conditions from text area
                conditions = st.text_area(
                    "Enter screening conditions (separate with AND):",
                    value=st.session_state.conditions,
                    height=100,
                    placeholder="""Market Capitalization > 100 AND
ROE > 15 AND
EPS Growth > 10"""
                )
                
                # Update conditions in session state
                st.session_state.conditions = conditions
                
                if st.button("Screen Stocks", type="primary"):
                    if conditions:
                        with st.spinner("Screening stocks..."):
                            filtered_df = st.session_state.screener.apply_conditions(df, conditions)
                            
                            if not filtered_df.empty:
                                st.session_state.filtered_df = filtered_df
                                display_filtered_data_analysis(filtered_df, st.session_state.screener)
                            else:
                                st.warning("No stocks match all specified criteria.")
                    else:
                        st.warning("Please enter screening conditions.")
                
                # Display previous results if they exist
                elif 'filtered_df' in st.session_state:
                    display_filtered_data_analysis(st.session_state.filtered_df, st.session_state.screener)
            
            # Talk with CSV Tab
            with tab2:
                st.header("Talk with Your Stock Data")
                query = st.text_area(
                    "Ask questions about your stock data:",
                    height=100,
                    placeholder="Example: What are the top 5 stocks by market cap? What's the relationship between ROE and EPS growth?"
                )
                
                if st.button("Analyze", key="analyze_button"):
                    if query:
                        with st.spinner("Analyzing data..."):
                            analysis = st.session_state.screener.analyze_with_gemini(df, query)
                            st.markdown(analysis)
                    else:
                        st.warning("Please enter a question about your data.")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
