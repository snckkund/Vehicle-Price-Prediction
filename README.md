# Vehicle Price Prediction

A machine learning application that predicts vehicle prices based on various features using advanced ML algorithms. Built with Streamlit, this interactive web application provides both prediction capabilities and insightful data visualizations.

## Project Structure
```
VehiclePricePrediction/
│
├── app.py                 # Main Streamlit application file
├── requirements.txt       # Project dependencies
│
├── dataset/              # Data directory
│   └── vehicle_data.csv  # Vehicle dataset
│
├── utils/                # Utility modules
│   ├── data_processor.py # Data preprocessing functions
│   ├── model_trainer.py  # ML model training utilities
│   └── visualizer.py     # Data visualization functions
│
└── .streamlit/           # Streamlit configuration
```

## Features
- Interactive web interface using Streamlit
- Advanced data preprocessing and feature engineering
- Multiple machine learning models for price prediction
- Comprehensive data visualizations and insights
- Real-time price predictions

## Requirements
- Python 3.12
- Dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/snckkund/Vehicle-Price-Prediction.git
cd Vehicle-Price-Prediction
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:5000)

3. Use the interactive interface to:
   - Upload vehicle data
   - View data visualizations
   - Get price predictions
   - Explore feature importance

## Project Components

### Data Processing (`utils/data_processor.py`)
- Handles data cleaning and preprocessing
- Feature engineering and transformation
- Data validation and quality checks

### Model Training (`utils/model_trainer.py`)
- Implements machine learning models
- Model training and evaluation
- Hyperparameter optimization

### Data Visualization (`utils/visualizer.py`)
- Creates interactive plots and charts
- Feature importance visualization
- Model performance metrics
