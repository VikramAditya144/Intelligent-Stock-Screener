# Intelligent Stock Screener

A web-based stock screening tool inspired by Screener.in. It allows you to filter and analyze a dataset of stocks based on various financial parameters.

## Features

- User-friendly interface with query-like filtering similar to Screener.in
- Support for various conditions (>, <, =) on metrics like Market Capitalization, P/E Ratio, ROE, etc.
- AND-only logic for filtering (all conditions must be met)
- Display of screened stocks in a table format
- Sorting functionality on each column
- Pagination for displaying large result sets (more than 10 stocks)
- Optional integration with Google Generative AI (requires API key) for analyzing screened data (beta feature)
- Responsive design for smooth operation on both desktops and mobile devices

## Getting Started

### Clone the repository:
```bash
git clone https://github.com/your-username/stock-sleuth.git
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

### (Optional) Set up Google Generative AI (beta feature):
1. Create a Google Cloud Project and enable the Generative AI API.
2. Obtain an API key and store it in a .env file named `.env.local` with the following line:
   ```
   GEMINI_API_KEY=your_api_key
   ```

### Run the application:
```bash
streamlit run main.py
```

## Usage

1. Access the application in your web browser at http://localhost:8501.
2. Upload your stock data CSV file in the sidebar.
3. Enter your filtering conditions in the "Stock Screening" tab using the query format (e.g., Market Capitalization > 100 AND ROE > 15).
4. Click the "Screen Stocks" button to apply the filters.
5. The filtered stocks will be displayed in a table along with summary statistics and data visualization options.
6. (Optional) In the "Talk with CSV" tab, you can ask questions about your data (e.g., "What are the top 5 stocks by market cap?") and the Generative AI model (if enabled) will attempt to provide an analysis.

## Note

- The Generative AI analysis is a beta feature and may not always produce accurate results.
- Ensure your stock data CSV file includes the following columns:
  - Market Capitalization (Billion)
  - P/E Ratio
  - ROE (%)
  - Debt-to-Equity Ratio
  - Dividend Yield (%)
  - Revenue Growth (%)
  - EPS Growth (%)
  - Current Ratio
  - Gross Margin (%)

## Disclaimer

This project is for educational purposes only and should not be taken as financial advice. Always conduct your own research before making any investment decisions.

## Future Improvements

- Implement additional filtering options (e.g., by industry, sector)
- Integrate with financial data APIs for real-time data access
- Enhance the Generative AI analysis capabilities
- Will be implementing Voice Models so that you can talk with the data
