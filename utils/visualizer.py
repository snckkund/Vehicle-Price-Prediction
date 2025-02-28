import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

class Visualizer:
    def plot_distribution(self, df, feature):
        """Plot distribution of a feature."""
        fig = px.histogram(
            df, 
            x=feature,
            title=f'Distribution of {feature}',
            template='plotly_white'
        )
        fig.update_layout(
            showlegend=False,
            xaxis_title=feature,
            yaxis_title="Count"
        )
        return fig
    
    def plot_correlation_matrix(self, df):
        """Plot correlation matrix of numerical features."""
        # Calculate correlation matrix
        corr_matrix = df.select_dtypes(include=[np.number]).corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        
        fig.update_layout(
            title='Feature Correlation Matrix',
            template='plotly_white'
        )
        return fig
    
    def plot_actual_vs_predicted(self, y_true, y_pred):
        """Plot actual vs predicted values."""
        fig = px.scatter(
            x=y_true,
            y=y_pred,
            labels={'x': 'Actual Price', 'y': 'Predicted Price'},
            title='Actual vs Predicted Prices'
        )
        
        # Add perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            )
        )
        
        fig.update_layout(template='plotly_white')
        return fig

    def plot_feature_importance(self, feature_importance, title="Feature Importance"):
        """Plot feature importance scores."""
        # Convert dictionary to dataframe
        df = pd.DataFrame(list(feature_importance.items()), columns=['Feature', 'Importance'])
        
        # Create bar plot
        fig = px.bar(
            df,
            x='Importance',
            y='Feature',
            orientation='h',
            title=title
        )
        
        fig.update_layout(
            template='plotly_white',
            yaxis={'categoryorder': 'total ascending'},  # Sort bars by value
            xaxis_title="Importance Score",
            yaxis_title="Feature"
        )
        
        return fig
