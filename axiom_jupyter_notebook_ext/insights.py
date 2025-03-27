import os
import pandas as pd
import json
from anthropic import Anthropic
from IPython.display import Markdown, display
from IPython.core.magic import register_line_magic

from .jupyter_axiom_extension import axiom_extension


class DataInsightsGenerator:
    """Generates insights about Axiom query results using Anthropic Claude"""
    
    def __init__(self):
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        
    def analyze_dataframe(self, df, query=None, max_rows=30):
        """Generate insights for a dataframe"""
        if self.client is None:
            return "Anthropic API key not set. Please set ANTHROPIC_API_KEY environment variable."
            
        if df is None or len(df) == 0:
            return "No data to analyze."
            
        # Prepare data sample for the model
        df_sample = df.head(max_rows)
        df_json = df_sample.to_json(orient='records', date_format='iso')
        df_columns = df.columns.tolist()
        
        # Get basic statistics - handle numeric columns only
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        stats_df = df[numeric_cols].describe() if numeric_cols else pd.DataFrame()
        stats_json = stats_df.to_json() if not stats_df.empty else "{}"
        
        # Prepare the system prompt
        system_prompt = """You are a data analyst expert specializing in logs and metrics analysis. 
Your task is to provide valuable insights based on the data you receive.

For the given dataset:
1. Identify key patterns, anomalies, or trends
2. Highlight important statistics and their implications
3. Suggest potential areas for further investigation
4. Look for correlations between fields when relevant
5. For time series data, note any temporal patterns
6. Format your response in Markdown for readability with appropriate headings and bullet points

Be concise but thorough. Focus on actionable insights rather than just describing the data.
Your insights should help the user understand what's happening in their system based on this data."""

        # Prepare user message with dataset info and original query
        user_message = f"Analyze this dataset with columns: {df_columns}\n\n"
        if query:
            user_message += f"This data was retrieved using the query: {query}\n\n"
        user_message += f"Dataset sample (first {max_rows} rows): {df_json}\n\n"
        
        if not stats_df.empty:
            user_message += f"Summary statistics for numeric columns: {stats_json}\n\n"
            
        user_message += "What insights can you provide about this data? Focus on patterns, anomalies, or interesting observations that would be valuable to someone monitoring systems or analyzing performance."
        
        try:
            response = self.client.messages.create(
                model="claude-3-opus-20240229",  # Using Opus for deeper analysis
                system=system_prompt,
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )
            
            insights = response.content[0].text
            return insights
            
        except Exception as e:
            return f"Error generating insights: {str(e)}"

@register_line_magic
def axiom_insights(line):
    """Generate AI insights for the latest query results"""
    ip = get_ipython()
    
    # Get the df from the user namespace (set by axiom_query)
    df = ip.user_ns.get('df')
    
    if df is None:
        print("No dataframe found. Run a query first.")
        return
        
    # Initialize insights generator if not already
    if not hasattr(axiom_extension, 'insights_generator'):
        axiom_extension.insights_generator = DataInsightsGenerator()
    
    # Get query if available
    query = None
    if hasattr(axiom_extension, 'last_query'):
        query = axiom_extension.last_query
    
    # Generate insights
    print("Generating insights, please wait...")
    insights = axiom_extension.insights_generator.analyze_dataframe(df, query)
    
    # Display as markdown
    display(Markdown(insights))

# Add to extension loading
def load_insights_magic(ipython):
    ipython.register_magic_function(axiom_insights, 'line', 'axiom_insights')