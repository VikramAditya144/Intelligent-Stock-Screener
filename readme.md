# Intelligent Stock Screener

A powerful web-based stock screening tool built with Streamlit and Python, inspired by Screener.in. This application allows users to filter and analyze stocks based on various financial metrics, visualize data distributions, and get AI-powered insights using Google's Gemini Pro.

🔗 Live Demo: [Stock Screener Application](https://vikramaditya144-intelligent-stock-screener-app-qpvfp6.streamlit.app/)

https://www.loom.com/share/6f99e373b53f4cf1b48207c25a277897

## Features

### 1. Advanced Stock Screening
- Filter stocks using multiple conditions with AND logic
- Support for various comparison operators (>, <, >=, <=, =)
- Real-time filtering and result updates
- Comprehensive financial metrics support:
  - Market Capitalization
  - P/E Ratio
  - ROE (Return on Equity)
  - Debt-to-Equity Ratio
  - Dividend Yield
  - Revenue Growth
  - EPS Growth
  - Current Ratio
  - Gross Margin

### 2. Data Visualization
- Interactive distribution plots for each metric
- Correlation heatmap for understanding relationships between metrics
- Summary statistics for filtered results
- Responsive and interactive charts using Plotly

### 3. AI-Powered Analysis
- Integration with Google's Gemini Pro API
- Natural language queries about your stock data
- Detailed financial analysis and insights
- Professional recommendations based on data patterns

### 4. User Interface
- Clean and intuitive design
- Sidebar with helpful examples and metric guides
- Tabbed interface for different functionalities
- Responsive layout that works on both desktop and mobile

## Technical Stack

- **Frontend Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly Express, Plotly Graph Objects
- **Statistical Analysis**: SciPy
- **AI Integration**: Google Generative AI (Gemini Pro)
- **Environment Management**: python-dotenv

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory and add:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

5. Run the application:
```bash
streamlit run app.py
```

## Usage Guide

### Basic Stock Screening

1. **Upload Data**:
   - Use the sidebar to upload your CSV file containing stock data
   - Ensure your CSV has the required columns matching the available metrics

2. **Enter Screening Conditions**:
   ```
   Market Capitalization > 100 AND
   ROE > 15 AND
   EPS Growth > 10
   ```

3. **View Results**:
   - Filtered stocks table
   - Summary statistics
   - Distribution analysis
   - Correlation heatmap

### AI Analysis

1. Navigate to the "Talk with CSV" tab
2. Enter your question about the data
3. Click "Analyze" to get AI-powered insights

## Example Queries

### Large Cap Growth Stocks
```
Market Capitalization > 100 AND
ROE > 15 AND
EPS Growth > 10
```

### Dividend Value Stocks
```
Dividend Yield > 2 AND
P/E Ratio < 20 AND
Debt-to-Equity Ratio < 1
```

## Data Format Requirements

Your CSV file should include the following columns:
- Market Capitalization (B)
- P/E Ratio
- ROE (%)
- Debt-to-Equity Ratio
- Dividend Yield (%)
- Revenue Growth (%)
- EPS Growth (%)
- Current Ratio
- Gross Margin (%)

## Implementation Details

### Key Classes

#### StockScreener
- Handles data processing and filtering
- Manages metric mappings and condition parsing
- Creates visualizations and statistical analysis
- Integrates with Gemini Pro for AI analysis

### Data Processing
- Column name cleaning and standardization
- Robust condition parsing and validation
- Error handling for invalid inputs
- Support for multiple numerical operators

### Visualization
- Distribution plots with KDE curves
- Correlation analysis using heatmaps
- Interactive Plotly charts
- Responsive design for different screen sizes

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by [Screener.in](https://www.screener.in/)
- Built with [Streamlit](https://streamlit.io/)
- Powered by [Google Gemini Pro](https://ai.google.dev/)

## Support

For support, please open an issue in the repository or contact [vikramaditya1533@gmail.com]
