from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

class ModelTrainer:
    def __init__(self, model_type='Random Forest'):
        self.model_type = model_type
        self.model = self._get_model()
        
    def _get_model(self):
        """Return the specified model instance with optimized hyperparameters."""
        if self.model_type == "Random Forest":
            return RandomForestRegressor(
                n_estimators=500,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "XGBoost":
            return XGBRegressor(
                n_estimators=500,
                max_depth=7,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=1,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=42,
                n_jobs=-1
            )
        else:
            return LinearRegression(n_jobs=-1)
    
    def _inverse_transform_predictions(self, y_pred):
        """Convert log predictions back to original scale"""
        return np.expm1(y_pred)
    
    def train_and_evaluate(self, X, y, test_size=0.2):
        """Train the model with cross-validation and return evaluation metrics."""
        # Ensure y is numpy array
        if isinstance(y, pd.Series):
            y = y.to_numpy()
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Make predictions (still in log scale)
        y_pred_log = self.model.predict(X_test)
        
        # Calculate metrics in original scale
        y_pred = self._inverse_transform_predictions(y_pred_log)
        y_test_orig = self._inverse_transform_predictions(y_test)
        
        metrics = {
            'r2_score': r2_score(y_test_orig, y_pred),
            'mae': mean_absolute_error(y_test_orig, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred)),
            'y_test': y_test_orig,
            'y_pred': y_pred
        }
        
        # Add cross-validation score (on log scale)
        cv_scores = cross_val_score(
            self.model, X, y, 
            cv=5, 
            scoring='r2'
        )
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()
        
        return metrics
    
    def predict(self, X):
        """Make predictions using the trained model."""
        y_pred_log = self.model.predict(X)
        return self._inverse_transform_predictions(y_pred_log)

    def get_feature_importance(self, feature_names):
        """Get feature importance scores for the trained model."""
        if self.model_type == "Random Forest":
            importance = self.model.feature_importances_
        elif self.model_type == "XGBoost":
            importance = self.model.feature_importances_
        else:  # Linear Regression
            importance = np.abs(self.model.coef_)
            
        # Create initial dictionary of feature names and their importance scores
        feature_importance = dict(zip(feature_names, importance))
        
        # Define basic features to show
        basic_features = {
            'year': 'Year',
            'mileage': 'Mileage',
            'cylinders': 'Cylinders',
            'doors': 'Doors',
            'engine_size': 'Engine Size',
            'make': 'Make',
            'fuel': 'Fuel Type',
            'transmission': 'Transmission',
            'body': 'Body Type',
            'trim': 'Trim'
        }
        
        # Aggregate importance for basic features
        aggregated_importance = {}
        for feature in feature_importance:
            # Check if feature is a basic feature
            if feature in basic_features:
                aggregated_importance[basic_features[feature]] = feature_importance[feature]
            else:
                # For one-hot encoded features, aggregate back to original feature
                for basic_feature, display_name in basic_features.items():
                    if feature.startswith(basic_feature + '_'):
                        if display_name not in aggregated_importance:
                            aggregated_importance[display_name] = 0
                        aggregated_importance[display_name] += feature_importance[feature]
        
        # Sort by importance score in descending order
        aggregated_importance = dict(sorted(aggregated_importance.items(), key=lambda x: x[1], reverse=True))
        return aggregated_importance
