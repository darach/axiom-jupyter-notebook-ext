import os
from anthropic import Anthropic
from IPython.core.magic import register_line_cell_magic
from .jupyter_axiom_extension import axiom_extension

class NLQHandler:
    """Handles natural language queries and converts them to Axiom queries"""
    
    def __init__(self, client=None):
        # Initialize Anthropic client with API key
        self.anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.axiom_client = client
        self.datasets = []
        self.recent_prompts = []
        self.recent_responses = []
        
    def update_client(self, client):
        """Update the Axiom client reference"""
        self.axiom_client = client
        # Refresh dataset list for context
        if client:
            try:
                self.datasets = [ds.name for ds in client.datasets.get_list()]
            except:
                pass
    
    def get_dataset_context(self):
        """Get dataset information to provide context to the LLM"""
        context = ""
        if not self.axiom_client or not self.datasets:
            return context
            
        # For each dataset, get fields info if available
        for ds_name in self.datasets[:5]:  # Limit to avoid token explosion
            try:
                fields_info = self.axiom_client.datasets.get_fields(ds_name)
                field_names = [field.name for field in fields_info]
                context += f"Dataset '{ds_name}' has these fields: {', '.join(field_names[:10])}\n"
            except:
                context += f"Dataset '{ds_name}' exists but field info isn't available\n"
                
        return context
    
    def generate_query(self, natural_query):
        """Convert natural language to Axiom query language"""
        # Build prompt with context about available datasets
        system_prompt = """You are a helpful expert in Axiom Query Language (AQL). 
Your task is to convert a natural language question into a valid AQL query.

AQL uses a piped syntax similar to LogQL:

Examples:
- "Count all events from last hour": `dataset_name | where _time > ago(1h) | count`
- "Show CPU usage by host": `system_metrics | where _measurement == "cpu" | summarize avg = avg(usage) by host`
- "Find error events": `logs | where level == "error" | limit 100`

Follow these rules:
1. Use only datasets mentioned in the context or query.
2. If no dataset is specified, choose the most appropriate one from context.
3. Include time ranges using ago() syntax when time is mentioned.
4. For aggregations, use summarize with appropriate functions.
5. Format the query properly with pipes.
6. Return ONLY the query, nothing else - no explanations, no markdown."""

        # Get available datasets for context
        context = self.get_dataset_context()
        if context:
            system_prompt += f"\n\nAvailable datasets and fields:\n{context}"
            
        # Generate query via Claude
        try:
            response = self.anthropic.messages.create(
                model="claude-3-sonnet-20240229",
                system=system_prompt,
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": f"Convert this to an Axiom query: {natural_query}"}
                ]
            )
            
            query = response.content[0].text.strip()
            
            # Save for history
            self.recent_prompts.append(natural_query)
            self.recent_responses.append(query)
            
            return query
            
        except Exception as e:
            return f"Error generating query: {str(e)}"

# Register as a Jupyter magic
@register_line_cell_magic
def axiom_nlq(line, cell=None):
    """Magic for natural language querying"""
    if not hasattr(axiom_extension, 'nlq_handler'):
        axiom_extension.nlq_handler = NLQHandler()
    
    # Update client reference if needed
    try:
        axiom_extension.nlq_handler.update_client(axiom_extension.client)
    except:
        print("Warning: Could not update client in NLQ handler")
    
    # Use the cell content if available, otherwise the line
    query_text = cell if cell is not None else line
    
    if not query_text.strip():
        print("Please provide a natural language query")
        return
        
    # Convert to AQL
    aql_query = axiom_extension.nlq_handler.generate_query(query_text)
    
    # Replace application-logs with vercel
    aql_query = aql_query.replace("application-logs", "vercel")
    
    print(f"Generated Axiom Query:\n{aql_query}\n")
    print("Running query...")
    
    # Run it through the existing axiom_query magic
    get_ipython().run_cell_magic("axiom_query", "", aql_query)

# Add to extension loading
def load_nlq_magic(ipython):
    # Make sure to explicitly register as both line and cell magic
    ipython.register_magic_function(axiom_nlq, 'cell', 'axiom_nlq')
    ipython.register_magic_function(axiom_nlq, 'line', 'axiom_nlq')