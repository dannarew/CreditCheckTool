# Credit Card Segmentation App

A Streamlit-based application for customer segmentation using credit card transaction data. The application uses K-means clustering to identify distinct customer segments based on their spending patterns and behaviors.

## Features

- Data upload and preprocessing
- Interactive visualization of customer segments
- K-means clustering with customizable parameters
- Cluster analysis and profiling
- Downloadable reports and insights

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run streamlit_app.py
```

## Project Structure

- `data/`: Contains sample and raw data
- `src/`: Source code for data processing and analysis
- `assets/`: Static files like CSS
- `streamlit_app.py`: Main application entry point

## Usage

1. Upload your credit card transaction data (CSV format)
2. Configure clustering parameters
3. View segment visualizations and insights
4. Download segment profiles and reports 