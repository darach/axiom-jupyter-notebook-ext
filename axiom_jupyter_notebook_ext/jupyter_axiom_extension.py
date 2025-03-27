"""
jupyter_axiom_extension.py

A jupyter extension that wraps the axiom_py python SDK with jupyter magic
to enable ease of:
 - querying Axiom datasets
 - ingesting data into Axiom datasets
 - inspecting Axiom datasets, events, and fields
 - converting Axiom tabular data to pandas DataFrames and vice versa
 - viewing Axiom tabular data in jupyter notebooks

 TODO Use click for magick cmd line args validation and usage
 TODO Identify common global arguments across all magics
 TODO Split out pandas dataframe ingest/query as an axiom_py extension
 TODO Split out magic and command line interface for same interactions with/without jupyter integration
 TODO Add comprehensive unit, functional and integration/acceptance tests
 TODO Add documentation
 TODO Add examples
 TODO Add a command line interface
 TODO Add a CLI only mode
 TODO Add a Jupyter notebook mode
 TODO Add a CLI and notebook mode flag
 TODO Prepare for oss release and push to pypi
"""

import pandas as pd
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from IPython.core.magic import (
    Magics,
    magics_class,
    line_magic,
    cell_magic,
)
from IPython.display import HTML
from dataclasses import dataclass, asdict

try:
    import axiom_py
    from axiom_py.client import (
        Client as AxiomClient,
        AplOptions,
        AplResultFormat,
    )

    HAS_AXIOM = True
except ImportError:
    HAS_AXIOM = False
import itables

import numpy as np


class AuthProvider:
    """Base class for authentication providers"""

    def get_credentials(self) -> Dict[str, str]:
        """Return credentials needed for axiom_py client"""
        raise NotImplementedError("Subclasses must implement get_credentials")

    def is_authenticated(self) -> bool:
        """Check if credentials are available and valid"""
        raise NotImplementedError("Subclasses must implement is_authenticated")


class EnvVarAuthProvider(AuthProvider):
    """Authentication provider using environment variables"""

    # IDEA Axiom SDK should support a profile argument, which is a dict in the axiom.toml file
    # and can specify org_id, token, and url via the profile name. This reduces friction when
    # using multiple axiom accounts in a single notebook and env based auth.
    def __init__(self, profile_var="AXIOM_PROFILE", org_id_var="AXIOM_ORG_ID", token_var="AXIOM_TOKEN", url_var="AXIOM_URL"):
        self.profile_var = profile_var
        self.org_id_var = org_id_var
        self.token_var = token_var
        self.url_var = url_var

    def get_credentials(self) -> Dict[str, str]:
        profile = os.environ.get(self.profile_var)
        org_id = os.environ.get(self.org_id_var)
        token = os.environ.get(self.token_var)
        url = os.environ.get(self.url_var)

        # If we have a profile, and it exists in the axiom.toml file, use it
        # Otherwise, expect the orgid, token and optional url to be set 
        if profile:
            import toml
            config = toml.load(Path.home() / ".axiom.toml")
            if profile in config["deployments"]:
                org_id = config["deployments"][profile]["org_id"]
                token = config["deployments"][profile]["token"]
                url = config["deployments"][profile]["url"]

        if not org_id or not token:
            raise ValueError(
                f"Missing environment variables: {self.org_id_var} and/or {self.token_var}"
            )
        
        if url is None:
            url = "https://api.axiom.co"

        return {"org_id": org_id, "token": token, "url": url}

    def is_authenticated(self) -> bool:
        return self.org_id_var in os.environ and self.token_var in os.environ


class AxiomClientAuthProvider(AuthProvider):
    """Authentication provider using a local config file produced and maintained by the axiom CLI"""

    def __init__(self, profile: Optional[str] = "default"):
        self.config_path = Path.home() / ".axiom.toml"
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at {self.config_path}")

        with open(self.config_path, "r") as f:
            import toml

            config = toml.load(f)

        if "active_deployment" not in config and profile is None:
            raise ValueError(
                "Config file must contain 'active_deployment' field or specify a profile explicitly"
            )

        if profile is None:
            profile = config["active_deployment"]

        if profile not in config["deployments"]:
            raise ValueError(f"Profile {profile} not found in config file")

        self.config = config["deployments"][profile]
        self.client = None

    def get_credentials(self) -> Dict[str, str]:
        return {"org_id": self.config["org_id"], "token": self.config["token"], "url": self.config["url"]}

    def is_authenticated(self) -> bool:
        if not self.config:
            return False
        return True

def keyvalueobject_to_dataframe(data_objects, line):
    """
    Convert a dataclass of key value pairs to a dataframe. Possible nesting should be flattened.
    """
    # Convert dataclass to dictionary
    data_dict = asdict(data_objects)
    print(data_dict)
    
    # Row per key value pair
    df = pd.DataFrame(data_dict.items(), columns=['Key', 'Value'])

    return df

# Define a helper function to convert dataclass instances to HTML tables
def dataclass_to_dataframe(data_objects, line):
    """
    Convert a list of dataclass objects to an HTML table for display in IPython/Jupyter

    Parameters:
    data_objects (list): List of dataclass instances

    Returns:
    IPython.display.HTML: HTML representation of the dataclass data
    """

    args = {}
    if line is not None:
        args = line.strip().split()

    opts = {}
    for arg in args:
        # split on = to get key and value
        key, value = arg.split("=")
        opts[key] = value

    # Handle a single instance by converting it to a list
    if not isinstance(data_objects, list):
        data_objects = [data_objects]

    if not data_objects:
        return HTML("<p>Empty dataset</p>")

    cells = [list(vars(obj).values()) for obj in data_objects]
    cols = list(vars(data_objects[0]).keys())
    excludes = opts.get("excludes", None)
    if excludes is not None:
        excludes = excludes.split(",")
        if len(cells) > 0:
            # remove cells for excluded fields
            cells = [
                [cell for i, cell in enumerate(row) if cols[i] not in excludes]
                for row in cells
            ]
            # ditto field names
            cols = [col for col in cols if col not in excludes]

    df = pd.DataFrame(cells, columns=cols)

    return df


# TODO Separate tabular <-> dataframe into python SDK
def tabular_to_dataframe(data, excludes=None):
    print("I hate you")
    # Extract cells and fields
    cells = data.columns
    fields = data.fields

    # Extract field names to use as column names
    # Assuming fields is a list of objects with a 'name' property
    column_names = [field.name for field in fields]
    # column_types = [axiom_to_pandas_type(field.type) for field in fields]

    if excludes is not None:
        column_names = [col for col in column_names if col not in excludes]
        column_types = [field.type for field in fields if field.name not in excludes]
        cells = [list(filter(lambda x: x[0] not in excludes, row)) for row in cells]

    # Create DataFrame using cells as data and field names as column names
    df = pd.DataFrame(
        list(map(list, zip(*cells))), columns=column_names, #dtype=column_types
    )

    return df

class AxiomExtension:
    """Jupyter extension for axiom_py SDK with pluggable authentication"""

    def __init__(self):
        self.client = None
        self.auth_provider = None
        self._auth_providers = {
            "env": EnvVarAuthProvider,
            "file": AxiomClientAuthProvider,
            "cli": AxiomClientAuthProvider,
        }

    def register_auth_provider(self, name: str, provider_class: type):
        """Register a custom authentication provider"""
        if not issubclass(provider_class, AuthProvider):
            raise TypeError("Provider class must be a subclass of AuthProvider")

        self._auth_providers[name] = provider_class

    def use_auth_provider(self, provider_type: str, **kwargs):
        """Set the authentication provider to use"""
        if provider_type not in self._auth_providers:
            raise ValueError(
                f"Unknown auth provider: {provider_type}. Available providers: {list(self._auth_providers.keys())}"
            )

        self.auth_provider = self._auth_providers[provider_type](**kwargs)

        # Reset client to force re-authentication
        self.client = None

    def get_client(self) -> AxiomClient:
        """Get or create an authenticated AxiomClient"""
        if not HAS_AXIOM:
            raise ImportError(
                "axiom_py is not installed. Install it with: pip install axiom_py"
            )

        if self.auth_provider is None:
            raise ValueError(
                "No authentication provider set. Call use_auth_provider first."
            )

        if not self.auth_provider.is_authenticated():
            raise ValueError(
                "Not authenticated. Check your authentication provider configuration."
            )

        if self.client is None:
            credentials = self.auth_provider.get_credentials()
            self.client = AxiomClient(
                org_id=credentials["org_id"], token=credentials["token"]
            )

        return self.client

    def query(self, client, query_string: str, **kwargs):
        """Run a query against Axiom"""

        opts = AplOptions(
            format=AplResultFormat.Tabular,
        )

        result = client.query(query_string, opts=opts)

        return result


# Create a singleton instance
axiom_extension = AxiomExtension()

@magics_class
class AxiomMagics(Magics):
    """IPython magics for Axiom integration"""

    def setvar(self, name, value):
        """Set a variable's value.

        Parameters:
        value (str): The value to set

        """

        # Try to evaluate the value as a Python expression
        ip = self.shell
        try:
            ip.user_ns[name] = value
            # print(f"Set '{name}'")
            return value
        except Exception as e:
            # If evaluation fails, set it as a string
            return value

    def set_df(self, value):
        return self.setvar("df", value)

    @line_magic
    def axiom_auth(self, line):
        """Configure authentication for axiom_py"""
        args = line.strip().split()
        if not args:
            print("Usage: %axiom_auth <provider_type> [provider_args]")
            print(
                f"Available providers: {list(axiom_extension._auth_providers.keys())}"
            )
            return

        provider_type = args[0]
        provider_args = {}

        # Parse additional arguments if any
        for arg in args[1:]:
            if "=" in arg:
                key, value = arg.split("=", 1)
                provider_args[key] = value

        try:
            axiom_extension.use_auth_provider(provider_type, **provider_args)
        except Exception as e:
            pass # do nothing, as axiom_info covers the success and failure cases

        self.axiom_info("")

    @line_magic
    def axiom_info(self, line):
        """Display information about the current Axiom configuration"""
        if axiom_extension.auth_provider is None:
            print("No authentication provider configured")
            return

        # Create a key-value dataframe with one row per credential (excluding token)
        # Keys are in upper camel case, values are in lower case
        credentials = axiom_extension.auth_provider.get_credentials()
        data = []
        for k, v in credentials.items():
            if k != 'token':
                # Convert key to upper camel case (e.g. org_id -> OrgId)
                key = ''.join(word.capitalize() for word in k.split('_'))
                # Convert value to lower case
                value = str(v).lower()
                data.append({'Key': key, 'Value': value})
        credentials = axiom_extension.auth_provider.get_credentials()
        data = []
        for k, v in credentials.items():
            if k != 'token':
                data.append({'Key': k, 'Value': v})
        
        # Add provider type and auth status
        data.append({'Key': 'Provider', 'Value': axiom_extension.auth_provider.__class__.__name__})
        data.append({'Key': 'Is authenticated?', 'Value': axiom_extension.auth_provider.is_authenticated()})
        
        # Create dataframe from all rows
        df = pd.DataFrame(data)

        # Test connection
        try:
            self.client = axiom_extension.get_client()
            # Append connect4ion test row to df
            df = pd.concat([df, pd.DataFrame([{"Key": "connection", "Value": "Success" if self.client is not None else "Failed"}])])
        except Exception as e:
            df = pd.concat([df, pd.DataFrame([{"Connection": f"Failed: {e}"}])])

        # Capture frame
        self.set_df(df)

        # Show properties
        return itables.show(df)

    @line_magic
    def axiom_datasets(self, line):
        """List datasets in the Axiom organization"""
        try:
            client = axiom_extension.get_client()
            datasets = client.datasets
            datasets = datasets.get_list()
            return itables.show(self.set_df(dataclass_to_dataframe(datasets, line)))
        except Exception as e:
            print(f"Error listing datasets: {e}")

    @cell_magic
    def axiom_query(self, line, cell):
        """Run an Axiom query from a cell"""
        try:
            print(cell.strip())
            client = axiom_extension.get_client()
            result = axiom_extension.query(client, cell)
            status = keyvalueobject_to_dataframe(result.status, line)
        


            if (
                len(result.tables[0].columns) == 1
                and len(result.tables[0].columns[0]) == 1
            ):
                df = pd.DataFrame([result.tables[0].columns[0][0]], columns=["Count"])
            else:
                df = tabular_to_dataframe(result.tables[0], None)

            # capture frame
            self.set_df(df)
            # show df and status in two separate tables
            return itables.show(df) #, itables.show(status)
        
        except Exception as e:
            print(f"Error executing query: {e}")
            status = dataclass_to_dataframe(e, line)
            # colorize http status codes for errors in the error frame for col 'status' int
            status['status'] = status['status'].apply(lambda x: f'<span style="color: {"orange" if x >= 300 else "red" if x >= 400 else "green"}">{x}</span>')
            return itables.show(status)

    @line_magic
    def axiom_ingest(self, line):
        """Ingest data into Axiom"""

        args = line.strip().split()
        if not args:
            print("Usage: %axiom_ingest <dataset_id> <df>")
            return

        dataset = args[0]
        df = self.shell.user_ns.get(args[1])
        print("Dataset: ", dataset)
        print("Dataframe: ", df)

        # if df has no _time column
        if "_time" not in df.columns:
            df["_time"] = pd.Timestamp.now(tz="UTC").strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ000"
            )

        ndjson = df.to_dict(orient="records")

        try:
            client = axiom_extension.get_client()
            result = client.ingest_events(
                dataset,
                ndjson,
            )

            print(result)
            itables.show(dataclass_to_dataframe(result, None))
        except Exception as e:
            print(f"Error ingesting data: {e}")


def load_ipython_extension(ipython):
    """Load the extension in IPython"""
    if not HAS_AXIOM:
        print("Warning: axiom_py not installed. Install with: pip install axiom_py")

    # Register magics
    ipython.register_magics(AxiomMagics)

    # Add axiom to user namespace for direct access
    ipython.push({"axiom": axiom_extension})
    
    # Load AI extensions if API keys are available
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            # Import and load NLQ magic
            from .nlq import load_nlq_magic
            load_nlq_magic(ipython)
            
            # Import and load insights magic
            from .insights import load_insights_magic 
            load_insights_magic(ipython)
            
            # Import and load predictive analytics
            from .predictive_analytics import load_predictive_analytics
            load_predictive_analytics(ipython)
            
            # Import and load MCP agent if MCP_SERVER_URL is set
            if os.environ.get("MCP_SERVER_URL"):
                from .mcp_agent import load_mcp_agent
                load_mcp_agent(ipython)
                print("All Claude AI features loaded (NLQ, Insights, Predictive Analytics, MCP Agent)")
            else:
                print("Claude AI features loaded (NLQ, Insights, Predictive Analytics) - MCP Agent requires MCP_SERVER_URL")
        except ImportError as e:
            print(f"Warning: AI extensions not fully loaded: {e}")
    else:
        print("AI features not loaded - set ANTHROPIC_API_KEY environment variable to enable")
    
    print("Axiom Extension loaded. Use %axiom_auth to configure authentication.")
