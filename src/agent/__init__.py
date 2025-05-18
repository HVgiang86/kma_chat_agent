"""
KMA Agent - Multi-agent system for KMA student queries.

This package implements a supervisor agent using LangGraph and tool-based agents
for handling student queries about regulations, student information, and scores.
"""

from agent.supervisor_agent import ReActGraph
from agent.state import MyAgentState

__version__ = "0.1.0"
__all__ = [
    "ReActGraph",
    "MyAgentState",
]

# Create and export an instance of the agent
react_graph = ReActGraph()

