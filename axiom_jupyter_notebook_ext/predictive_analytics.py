import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from anthropic import Anthropic
import os
import json
from IPython.core.magic import register_line_magic
from IPython.display import display, Markdown
from .jupyter_axiom_extension import axiom_extension


class PredictiveAnalytics:
    """Predictive analytics tools for Axiom data using Claude"""
    
    def __init__(self):
        self.anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    def detect_anomalies(self, df, value_cols=None, threshold=2):
        """Detect simple anomalies using z-score method"""
        # Find numeric columns if not specified
        if value_cols is None:
            value_cols = df.select_dtypes(include=np.number).columns.tolist()
            if '_time' in value_cols:
                value_cols.remove('_time')
                
        if not value_cols:
            return None, "No numeric columns found for anomaly detection"
            
        try:
            # Create result dataframe
            result_df = df.copy()
            result_df['is_anomaly'] = False
            
            # Check each column for anomalies
            for col in value_cols:
                # Calculate z-scores
                mean = df[col].mean()
                std = df[col].std()
                if std == 0:  # Avoid division by zero
                    continue
                z_scores = (df[col] - mean) / std
                
                # Mark anomalies
                anomalies = abs(z_scores) > threshold
                result_df['is_anomaly'] = result_df['is_anomaly'] | anomalies
            
            # Count anomalies
            anomaly_count = result_df['is_anomaly'].sum()
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # If we have timestamp column, use it for x-axis
            if '_time' in df.columns:
                time_col = pd.to_datetime(df['_time'])
                
                # Plot normal points
                ax.scatter(
                    time_col[~result_df['is_anomaly']], 
                    df.loc[~result_df['is_anomaly'], value_cols[0]],
                    c='blue', label='Normal'
                )
                
                # Plot anomalies
                ax.scatter(
                    time_col[result_df['is_anomaly']], 
                    df.loc[result_df['is_anomaly'], value_cols[0]],
                    c='red', label='Anomaly'
                )
                
                ax.set_xlabel('Time')
                
            else:
                # If no time column, use index and display points
                ax.scatter(
                    range(sum(~result_df['is_anomaly'])),
                    df.loc[~result_df['is_anomaly'], value_cols[0]],
                    c='blue', label='Normal'
                )
                
                ax.scatter(
                    range(sum(result_df['is_anomaly'])),
                    df.loc[result_df['is_anomaly'], value_cols[0]],
                    c='red', label='Anomaly'
                )
                
                ax.set_xlabel('Index')
                
            ax.set_ylabel(value_cols[0])
            ax.set_title(f'Anomaly Detection ({anomaly_count} anomalies found)')
            ax.legend()
            
            # Get Claude's explanation of anomalies
            self.get_anomaly_explanation(result_df, value_cols, df)
            
            return result_df, fig
            
        except Exception as e:
            return None, f"Error detecting anomalies: {str(e)}"
            
    def get_anomaly_explanation(self, result_df, value_cols, original_df):
        """Get Claude's explanation of detected anomalies"""
        if sum(result_df['is_anomaly']) == 0:
            return "No anomalies detected."
            
        # Prepare data for Claude
        anomaly_df = original_df[result_df['is_anomaly']].copy()
        
        # If there are too many anomalies, just take a sample
        if len(anomaly_df) > 20:
            anomaly_df = anomaly_df.sample(20)
            
        anomaly_json = anomaly_df.to_json(orient='records', date_format='iso')
        
        # Get statistics about normal vs anomalous data
        normal_stats = {}
        anomaly_stats = {}
        
        for col in value_cols:
            normal_stats[col] = {
                'mean': float(original_df.loc[~result_df['is_anomaly'], col].mean()),
                'std': float(original_df.loc[~result_df['is_anomaly'], col].std()),
                'min': float(original_df.loc[~result_df['is_anomaly'], col].min()),
                'max': float(original_df.loc[~result_df['is_anomaly'], col].max())
            }
            
            anomaly_stats[col] = {
                'mean': float(original_df.loc[result_df['is_anomaly'], col].mean()),
                'std': float(original_df.loc[result_df['is_anomaly'], col].std()),
                'min': float(original_df.loc[result_df['is_anomaly'], col].min()),
                'max': float(original_df.loc[result_df['is_anomaly'], col].max())
            }
        
        # Ask Claude to explain
        system_prompt = """You are an expert data analyst specializing in anomaly detection and explanation. 
Your task is to explain anomalies detected in a dataset, focusing on why they might have occurred and what they might indicate.

Be specific about:
1. The characteristics that make these data points anomalous
2. Potential causes for these anomalies in a monitoring or observability context
3. Whether the anomalies represent potential issues or expected behavior
4. Recommendations for further investigation

Keep your response concise, informative, and actionable."""

        user_message = f"""Anomalies have been detected in a dataset using z-score threshold detection.

Here are statistics about the normal data points:
{json.dumps(normal_stats, indent=2)}

Here are statistics about the anomalous data points:
{json.dumps(anomaly_stats, indent=2)}

Here's a sample of the anomalous data points:
{anomaly_json}

Please explain why these might be considered anomalies, what they might indicate in a monitoring system, and what actions might be worth taking."""

        try:
            response = self.anthropic.messages.create(
                model="claude-3-sonnet-20240229",
                system=system_prompt,
                max_tokens=1500,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )
            
            explanation = response.content[0].text
            display(Markdown("## Anomaly Analysis\n" + explanation))
            
        except Exception as e:
            print(f"Error generating anomaly explanation: {str(e)}")

@register_line_magic
def axiom_anomalies(line):
    """Detect anomalies in the latest query results"""
    ip = get_ipython()
    
    # Get the df from the user namespace
    df = ip.user_ns.get('df')
    
    if df is None:
        print("No dataframe found. Run a query first.")
        return
        
    # Initialize predictive analytics if not already
    if not hasattr(axiom_extension, 'predictive_analytics'):
        axiom_extension.predictive_analytics = PredictiveAnalytics()
    
    # Parse arguments
    args = line.split()
    value_cols = None
    threshold = 2
    
    for arg in args:
        if '=' in arg:
            key, value = arg.split('=')
            if key == 'value_cols':
                value_cols = value.split(',')
            elif key == 'threshold':
                threshold = float(value)
    
    # Detect anomalies
    print("Detecting anomalies...")
    result_df, result = axiom_extension.predictive_analytics.detect_anomalies(
        df, value_cols, threshold
    )
    
    if isinstance(result, plt.Figure):
        display(result)
    else:
        print(result)  # Error message
    
    # Store result in user namespace
    if result_df is not None:
        ip.user_ns['anomalies_df'] = result_df
        print("Anomaly detection results saved to 'anomalies_df' variable")


@register_line_magic
def axiom_forecast(line):
    """Generate natural language forecast for the latest query results"""
    ip = get_ipython()
    
    # Get the df from the user namespace
    df = ip.user_ns.get('df')
    
    if df is None:
        print("No dataframe found. Run a query first.")
        return
        
    # Initialize predictive analytics if not already
    if not hasattr(axiom_extension, 'predictive_analytics'):
        axiom_extension.predictive_analytics = PredictiveAnalytics()
    
    # Convert dataframe to JSON for Claude
    df_json = df.head(50).to_json(orient='records', date_format='iso')
    
    # Call Claude for forecasting insights
    system_prompt = """You are an expert data scientist specializing in time series forecasting.
Your task is to analyze temporal data and provide insights about likely future trends.

Based on the data provided:
1. Identify patterns, trends, and seasonality in the data
2. Make predictions about how these patterns are likely to continue
3. Note any factors that might influence future values
4. Provide confidence levels for your predictions

Keep your response focused on the forecast rather than just describing the data."""

    user_message = f"""I have time series data from a monitoring system and need forecasting insights.

Here's a sample of the data:
{df_json}

Based on this data, please provide:
1. A forecast of how these metrics are likely to trend in the near future
2. Any patterns that suggest potential future issues
3. Confidence in these predictions and what might change the outcome"""

    try:
        client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model="claude-3-opus-20240229",  # Using Opus for better forecasting
            system=system_prompt,
            max_tokens=1500,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        
        forecast = response.content[0].text
        display(Markdown("## Forecast Analysis\n" + forecast))
        
    except Exception as e:
        print(f"Error generating forecast: {str(e)}")

# Add to extension loading
def load_predictive_analytics(ipython):
    ipython.register_magic_function(axiom_anomalies, 'line', 'axiom_anomalies')
    ipython.register_magic_function(axiom_forecast, 'line', 'axiom_forecast')