import os
import json
import time
import uuid
import asyncio
import aiohttp
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from anthropic import Anthropic
from datetime import datetime, timedelta

from .jupyter_axiom_extension import axiom_extension


class MCPAgent:
    """An AI agent using Mission Control Protocol to perform autonomous data analysis"""
    
    def __init__(self, axiom_client=None):
        self.anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.axiom_client = axiom_client
        self.mcp_server_url = os.environ.get("MCP_SERVER_URL", "http://localhost:3333")
        self.agent_id = str(uuid.uuid4())
        self.conversation_id = str(uuid.uuid4())
        self.mission_id = None
        self.session = None
        self.datasets = []
        self.mission_status = "idle"
        self.mission_results = {}
        
    def update_client(self, client):
        """Update the Axiom client reference"""
        self.axiom_client = client
        # Refresh dataset list
        if client:
            try:
                self.datasets = [ds.name for ds in client.datasets.get_list()]
            except:
                pass
    
    async def initialize_session(self):
        """Initialize an aiohttp session for MCP server communication"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close_session(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def register_mission(self, mission_objective):
        """Register a new mission with the MCP server"""
        session = await self.initialize_session()
        
        mission_data = {
            "agentId": self.agent_id,
            "conversationId": self.conversation_id,
            "objective": mission_objective,
            "metadata": {
                "datasets": self.datasets,
                "client": "axiom_jupyter",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        try:
            async with session.post(
                f"{self.mcp_server_url}/missions", 
                json=mission_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    self.mission_id = result.get("missionId")
                    self.mission_status = "registered"
                    return self.mission_id
                else:
                    error = await response.text()
                    return f"Error registering mission: {error}"
        except Exception as e:
            return f"Connection error: {str(e)}"
    
    async def execute_mission(self, mission_steps=None):
        """Execute the mission steps through the MCP server"""
        if not self.mission_id:
            return "No mission registered"
            
        if not mission_steps:
            # Generate mission steps based on objective
            mission_steps = await self.generate_mission_steps()
        
        session = await self.initialize_session()
        
        execution_data = {
            "missionId": self.mission_id,
            "steps": mission_steps
        }
        
        try:
            async with session.post(
                f"{self.mcp_server_url}/missions/{self.mission_id}/execute", 
                json=execution_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    self.mission_status = "executing"
                    return result
                else:
                    error = await response.text()
                    return f"Error executing mission: {error}"
        except Exception as e:
            return f"Connection error: {str(e)}"
    
    async def check_mission_status(self):
        """Check the status of a running mission"""
        if not self.mission_id:
            return "No mission registered"
            
        session = await self.initialize_session()
        
        try:
            async with session.get(
                f"{self.mcp_server_url}/missions/{self.mission_id}/status"
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    self.mission_status = result.get("status", "unknown")
                    return result
                else:
                    error = await response.text()
                    return f"Error checking mission status: {error}"
        except Exception as e:
            return f"Connection error: {str(e)}"
    
    async def get_mission_results(self):
        """Get the results of a completed mission"""
        if not self.mission_id:
            return "No mission registered"
            
        session = await self.initialize_session()
        
        try:
            async with session.get(
                f"{self.mcp_server_url}/missions/{self.mission_id}/results"
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    self.mission_results = result
                    return result
                else:
                    error = await response.text()
                    return f"Error getting mission results: {error}"
        except Exception as e:
            return f"Connection error: {str(e)}"
    
    async def generate_mission_steps(self):
        """Generate mission steps using Claude"""
        if not self.mission_id:
            return []
            
        # Define the system prompt
        system_prompt = """You are a data analysis agent that determines the necessary steps to complete a data analysis mission.

For each mission, you need to provide a sequence of well-defined steps that will solve the objective.
Each step should be concrete and actionable. The steps will be executed by specialized tools.

Available tools:
1. axiom_query - Executes Axiom queries and returns results
2. data_transform - Performs data transformations like filtering, grouping, etc.
3. visualization - Creates visualizations of data
4. forecasting - Generates predictions and forecasts
5. anomaly_detection - Identifies anomalies in data
6. insights_generator - Provides analysis and insights
7. alert_creator - Creates alerts based on conditions

For each step, provide:
- tool: which tool to use
- parameters: what parameters to pass
- description: brief description of what this step does
- expected_output: what output format is expected"""

        # Add dataset context if available
        if self.datasets:
            system_prompt += f"\n\nAvailable datasets: {', '.join(self.datasets)}"
        
        try:
            # Get mission objective
            async with self.session.get(
                f"{self.mcp_server_url}/missions/{self.mission_id}"
            ) as response:
                if response.status == 200:
                    mission = await response.json()
                    objective = mission.get("objective", "")
                else:
                    return []
            
            # Generate steps for the objective
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                system=system_prompt,
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": f"Generate a sequence of steps to complete this data analysis mission: '{objective}'. Return the steps as a JSON array where each step has tool, parameters, description, and expected_output fields."}
                ]
            )
            
            # Extract and parse JSON
            steps_text = response.content[0].text
            
            # Find JSON in the response
            start_index = steps_text.find("[")
            end_index = steps_text.rfind("]") + 1
            
            if start_index != -1 and end_index != -1:
                json_str = steps_text[start_index:end_index]
                steps = json.loads(json_str)
                return steps
            else:
                # Fallback if JSON not found
                return []
                
        except Exception as e:
            print(f"Error generating mission steps: {str(e)}")
            return []
            
    async def run_mission(self, objective, timeout=300):
        """Run a full mission workflow"""
        # Register mission
        mission_id = await self.register_mission(objective)
        if isinstance(mission_id, str) and mission_id.startswith("Error"):
            return mission_id
            
        # Execute mission
        execution = await self.execute_mission()
        if isinstance(execution, str) and execution.startswith("Error"):
            return execution
            
        # Poll for completion
        start_time = time.time()
        status = "executing"
        
        # Create progress widget
        progress = widgets.FloatProgress(
            value=0,
            min=0,
            max=1.0,
            description='Progress:',
            bar_style='info',
            style={'description_width': 'initial'}
        )
        
        status_label = widgets.HTML(value="<b>Status:</b> Starting mission...")
        display(status_label)
        display(progress)
        
        while status in ["executing", "registered"] and (time.time() - start_time) < timeout:
            status_response = await self.check_mission_status()
            
            if isinstance(status_response, dict):
                status = status_response.get("status", "unknown")
                progress_value = status_response.get("progress", 0)
                current_step = status_response.get("currentStep", "")
                
                # Update widgets
                progress.value = progress_value
                status_label.value = f"<b>Status:</b> {status} - {current_step}"
                
                if status == "completed":
                    break
                    
            await asyncio.sleep(2)
            
        if status != "completed":
            status_label.value = "<b>Status:</b> Timeout or error occurred"
            return "Mission did not complete in time"
            
        # Get results
        results = await self.get_mission_results()
        
        # Clear progress UI
        clear_output(wait=True)
        
        # Display final results
        if isinstance(results, dict):
            # Process and format results
            html_output = "<h2>Mission Results</h2>"
            
            # Add mission details
            html_output += f"<p><b>Objective:</b> {objective}</p>"
            
            # Add insights if available
            if "insights" in results:
                html_output += f"<h3>Insights</h3>{results['insights']}"
                
            # Add visualizations if available
            if "visualizations" in results and results["visualizations"]:
                html_output += "<h3>Visualizations</h3>"
                for viz in results["visualizations"]:
                    if "image_data" in viz:
                        html_output += f"<div><h4>{viz.get('title', 'Visualization')}</h4>"
                        html_output += f"<img src='data:image/png;base64,{viz['image_data']}' />"
                        html_output += f"<p>{viz.get('description', '')}</p></div>"
            
            # Add data summary if available
            if "data_summary" in results:
                html_output += f"<h3>Data Summary</h3>{results['data_summary']}"
                
            # Display
            display(HTML(html_output))
            
            # Store results in mission_results
            self.mission_results = results
            return results
        else:
            return results

# Command-line interface for MCP agent
class MCPAgentCLI:
    """Command-line interface for the MCP Agent"""
    
    def __init__(self, agent):
        self.agent = agent
        self.loop = asyncio.get_event_loop()
        
    def run(self, objective):
        """Run a mission with the given objective"""
        return self.loop.run_until_complete(self.agent.run_mission(objective))

# Register as Jupyter magic
from IPython.core.magic import register_line_cell_magic

@register_line_cell_magic
def axiom_mission(line, cell=None):
    """Magic for running AI missions"""
    if not hasattr(axiom_extension, 'mcp_agent'):
        axiom_extension.mcp_agent = MCPAgent()
    
    # Update client reference if needed
    try:
        client = axiom_extension.get_client()
        axiom_extension.mcp_agent.update_client(client)
    except:
        print("Warning: Could not update client in MCP agent")
    
    # Use the cell content if available, otherwise the line
    objective = cell if cell is not None else line
    
    if not objective.strip():
        print("Please provide a mission objective")
        return
    
    # Create and run event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        results = loop.run_until_complete(axiom_extension.mcp_agent.run_mission(objective))
        return results
    finally:
        loop.close()

# Add to extension loading
def load_mcp_agent(ipython):
    ipython.register_magic_function(axiom_mission, 'line_cell', 'axiom_mission')