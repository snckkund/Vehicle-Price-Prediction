import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import re

class DataProcessor:
    def __init__(self, df):
        df = df.copy()
        
        # Extract engine size from engine description
        df['engine_size'] = df['engine'].fillna('').apply(self._extract_engine_size)
        
        # Convert categorical columns to string type
        self.categorical_features = ['make', 'model', 'fuel', 'transmission', 'body', 'trim', 'exterior_color', 'interior_color']
        for col in self.categorical_features:
            df[col] = df[col].astype(str)
        
        self.df = df
        self.column_transformer = None
        self.numerical_features = ['year', 'mileage', 'cylinders', 'engine_size', 'doors']
        
    def _extract_engine_size(self, engine_desc):
        """Extract engine size in liters from engine description"""
        if pd.isna(engine_desc):
            return np.nan
        
        # Try to find L or l followed by a number
        match = re.search(r'(\d+\.?\d*)L?', str(engine_desc))
        if match:
            return float(match.group(1))
        return np.nan

    def _handle_price_outliers(self, df, price_col='price'):
        """Remove extreme price outliers using IQR method"""
        Q1 = df[price_col].quantile(0.25)
        Q3 = df[price_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[price_col] >= lower_bound) & (df[price_col] <= upper_bound)]

    def prepare_features_and_target(self, target_column='price'):
        """Prepare features and target variable."""
        if target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")

        # Remove rows with null values in target and handle price outliers
        df_clean = self.df.dropna(subset=[target_column])
        df_clean = self._handle_price_outliers(df_clean, target_column)

        # Log transform the target variable
        y = np.log1p(df_clean[target_column])
        
        # Select features
        features = self.numerical_features + self.categorical_features
        X = df_clean[features]

        return X, y

    def preprocess_features(self, X):
        """Preprocess features for model training."""
        if self.column_transformer is None:
            # Numerical pipeline with robust scaling
            numerical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Categorical pipeline with one-hot encoding
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            ])

            # Column transformer
            self.column_transformer = ColumnTransformer(
                transformers=[
                    ('num', numerical_pipeline, self.numerical_features),
                    ('cat', categorical_pipeline, self.categorical_features)
                ]
            )

            # Fit and transform
            X_processed = self.column_transformer.fit_transform(X)
            
            # Get feature names
            numeric_features = self.numerical_features
            categorical_features = []
            if hasattr(self.column_transformer.named_transformers_['cat'], 'get_feature_names_out'):
                categorical_features = self.column_transformer.named_transformers_['cat'].get_feature_names_out(self.categorical_features)
            
            # Clean feature names to be compatible with XGBoost
            def clean_feature_name(name):
                return str(name).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_')
            
            # Combine all feature names
            feature_names = [clean_feature_name(f) for f in numeric_features]
            feature_names.extend([clean_feature_name(f) for f in categorical_features])
            
            # Convert to DataFrame
            return pd.DataFrame(X_processed, columns=feature_names)

        # Transform new data
        X_processed = self.column_transformer.transform(X)
        
        # Get feature names
        numeric_features = self.numerical_features
        categorical_features = []
        if hasattr(self.column_transformer.named_transformers_['cat'], 'get_feature_names_out'):
            categorical_features = self.column_transformer.named_transformers_['cat'].get_feature_names_out(self.categorical_features)
        
        # Clean feature names to be compatible with XGBoost
        def clean_feature_name(name):
            return str(name).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_')
        
        # Combine all feature names
        feature_names = [clean_feature_name(f) for f in numeric_features]
        feature_names.extend([clean_feature_name(f) for f in categorical_features])
        
        # Convert to DataFrame
        return pd.DataFrame(X_processed, columns=feature_names)