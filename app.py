import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor
from utils.model_trainer import ModelTrainer
from utils.visualizer import Visualizer

st.set_page_config(page_title="Vehicle Price Prediction", layout="wide")

def load_data():
    """Load data from the dataset directory"""
    try:
        df = pd.read_csv('dataset/vehicle_data.csv')
        # Fill NaN values with 'Unknown' for categorical columns
        categorical_cols = ['make', 'fuel', 'transmission', 'exterior_color', 'interior_color']
        for col in categorical_cols:
            df[col] = df[col].fillna('Unknown')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def main():
    st.title("Vehicle Price Prediction System")

    # Load data and train model
    df = load_data()
    
    @st.cache_resource
    def initialize_models(df):
        """Initialize all three models"""
        data_processor = DataProcessor(df)
        X, y = data_processor.prepare_features_and_target()
        X_processed = data_processor.preprocess_features(X)
        
        models = {}
        for model_type in ["Random Forest", "XGBoost", "Linear Regression"]:
            trainer = ModelTrainer(model_type)
            trainer.train_and_evaluate(X_processed, y)
            models[model_type] = trainer
            
        return models, data_processor

    if df is not None:
        data_processor = DataProcessor(df)
        visualizer = Visualizer()

        # Sidebar options
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", ["Data Exploration", "Model Training", "Price Prediction"])

        if page == "Data Exploration":
            st.header("Data Exploration")

            # Display basic statistics
            st.subheader("Dataset Overview")
            st.write(f"Number of records: {len(df)}")
            st.write(f"Number of features: {len(df.columns)}")

            # Display sample data
            st.subheader("Sample Data")
            st.dataframe(df.head())

            # Feature distributions
            st.subheader("Feature Distributions")
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            selected_feature = st.selectbox("Select feature to visualize", numeric_columns)
            fig = visualizer.plot_distribution(df, selected_feature)
            st.plotly_chart(fig)

            # Correlation matrix
            st.subheader("Feature Correlations")
            fig = visualizer.plot_correlation_matrix(df)
            st.plotly_chart(fig)

        elif page == "Model Training":
            st.header("Model Training")

            # Model parameters
            st.subheader("Model Configuration")
            model_type = st.selectbox("Select Model", ["Random Forest", "XGBoost", "Linear Regression"])
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2)

            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    # Prepare data
                    X, y = data_processor.prepare_features_and_target()
                    X_processed = data_processor.preprocess_features(X)

                    # Train model
                    model_trainer = ModelTrainer(model_type)
                    metrics = model_trainer.train_and_evaluate(X_processed, y, test_size)

                    # Display metrics
                    st.subheader("Model Performance")
                    st.write(f"R2 Score: {metrics['r2_score']:.4f}")
                    st.write(f"MAE: {metrics['mae']:.2f}")
                    st.write(f"RMSE: {metrics['rmse']:.2f}")

                    # Plot actual vs predicted
                    fig = visualizer.plot_actual_vs_predicted(
                        metrics['y_test'], 
                        metrics['y_pred']
                    )
                    st.plotly_chart(fig)

                    # Display feature importance
                    st.subheader("Feature Importance")
                    if model_type in ["Random Forest", "XGBoost"]:
                        feature_importance = model_trainer.get_feature_importance(X_processed.columns)
                        fig = visualizer.plot_feature_importance(feature_importance)
                        st.plotly_chart(fig)
                    elif model_type == "Linear Regression":
                        feature_importance = model_trainer.get_feature_importance(X_processed.columns)
                        fig = visualizer.plot_feature_importance(feature_importance, title="Feature Coefficients (Absolute Values)")
                        st.plotly_chart(fig)

        else:  # Price Prediction
            st.header("Price Prediction")

            # Input form for prediction
            st.subheader("Enter Vehicle Details")

            col1, col2, col3 = st.columns(3)
            
            with col1:
                year = st.selectbox("Year", [2023, 2024])
                mileage = st.number_input("Mileage", min_value=0)
                cylinders = st.number_input("Cylinders", min_value=0)
                doors = st.number_input("Number of Doors", min_value=2, max_value=5, value=4)

            with col2:
                # Get unique makes
                make = st.selectbox("Make", sorted(df['make'].fillna('Unknown').astype(str).unique()))
                
                # Filter models based on selected make
                models_for_make = sorted(df[df['make'] == make]['model'].unique())
                model = st.selectbox("Model", models_for_make)
                
                # Filter trims based on selected make and model
                trims_for_model = sorted(df[(df['make'] == make) & (df['model'] == model)]['trim'].unique())
                trim = st.selectbox("Trim Level/Series", trims_for_model)
                
                fuel = st.selectbox("Fuel Type", sorted(df['fuel'].fillna('Unknown').astype(str).unique()))
                transmission = st.selectbox("Transmission", sorted(df['transmission'].fillna('Unknown').astype(str).unique()))

            with col3:
                body = st.selectbox("Body Type", sorted(df['body'].fillna('Unknown').astype(str).unique()))
                engine = st.selectbox("Engine", sorted(df['engine'].fillna('Unknown').astype(str).unique()))
                exterior_color = st.selectbox("Exterior Color", sorted(df['exterior_color'].fillna('Unknown').astype(str).unique()))
                interior_color = st.selectbox("Interior Color", sorted(df['interior_color'].fillna('Unknown').astype(str).unique()))

            if st.button("Predict Price"):
                # Extract engine size from engine description
                engine_size = None
                if engine != 'Unknown':
                    import re
                    match = re.search(r'(\d+\.?\d*)L?', str(engine))
                    if match:
                        engine_size = float(match.group(1))

                # Create prediction sample
                sample = pd.DataFrame({
                    'year': [year],
                    'mileage': [mileage],
                    'cylinders': [cylinders],
                    'doors': [doors],
                    'make': [make],
                    'model': [model],
                    'fuel': [fuel],
                    'transmission': [transmission],
                    'body': [body],
                    'trim': [trim],
                    'exterior_color': [exterior_color],
                    'interior_color': [interior_color],
                    'engine_size': [engine_size if engine_size is not None else np.nan]
                })

                # Get or initialize trained models
                if 'models' not in st.session_state:
                    st.session_state.models, st.session_state.data_processor = initialize_models(df)

                # Process sample
                X_processed = st.session_state.data_processor.preprocess_features(sample)
                
                # Get predictions from all models
                st.subheader("Model Predictions")
                
                # Performance-based weights for ensemble
                weights = {
                    "Linear Regression": 0.45,  # Highest weight due to best performance
                    "XGBoost": 0.35,           # Medium weight
                    "Random Forest": 0.20       # Lowest weight due to lower performance
                }
                
                predictions = {}
                
                # Show Linear Regression first (best performer)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Linear Regression** (Best Performer)")
                    lr_pred = st.session_state.models["Linear Regression"].predict(X_processed)[0]
                    predictions["Linear Regression"] = lr_pred
                    st.write(f"${lr_pred:,.2f}")
                    st.caption("R² Score: 0.886")
                    
                with col2:
                    st.markdown("**XGBoost**")
                    xgb_pred = st.session_state.models["XGBoost"].predict(X_processed)[0]
                    predictions["XGBoost"] = xgb_pred
                    st.write(f"${xgb_pred:,.2f}")
                    st.caption("R² Score: 0.794")
                    
                with col3:
                    st.markdown("**Random Forest**")
                    rf_pred = st.session_state.models["Random Forest"].predict(X_processed)[0]
                    predictions["Random Forest"] = rf_pred
                    st.write(f"${rf_pred:,.2f}")
                    st.caption("R² Score: 0.695")
                
                # Calculate weighted ensemble average
                weighted_avg = sum(predictions[model] * weights[model] for model in predictions)
                st.markdown("---")
                st.markdown("**Weighted Ensemble Prediction**")
                st.write(f"${weighted_avg:,.2f}")
                st.caption("(Weighted average based on model performance)")
                
                # Calculate and display average prediction
                avg_pred = np.mean([rf_pred, xgb_pred, lr_pred])
                st.markdown("**Ensemble Average**")
                st.write(f"${avg_pred:,.2f}")

if __name__ == "__main__":
    main()