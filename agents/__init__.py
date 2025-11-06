"""
Agents package for the MIND medical documentation system.

This package provides agent implementations for retrieval and information
synthesis tasks. Agents use LangChain's agent framework to orchestrate
multi-step reasoning and tool usage.

Key Modules:
    base_agent: Abstract base class for all agents
    exceptions: Custom exceptions for agent-related errors
    retrieval_agent: Multi-tool retrieval agent implementation

Key Classes:
    BaseAgent: Abstract base class defining agent interface
    RetrievalAgent: Agent for multi-source information retrieval

Key Functions:
    create_retrieval_agent: Factory function to create retrieval agent
    invoke_retrieval_agent: Convenience function to invoke retrieval agent

Exceptions:
    AgentError: Base exception for agent errors
    AgentExecutionError: Agent execution failed
    AgentConfigurationError: Invalid agent configuration
    AgentTimeoutError: Agent exceeded time limit
    ToolExecutionError: Tool execution failed
    InvalidAgentStateError: Agent in invalid state

Example:
    >>> from agents import invoke_retrieval_agent
    >>> response = invoke_retrieval_agent("Hvad er diabetes?")
    >>> print(response)

    >>> # Using custom configuration
    >>> from agents import create_retrieval_agent
    >>> agent = create_retrieval_agent(verbose=False, max_iterations=20)
    >>> response = agent.invoke({"input": "Patient information?"})
"""

# Base agent class
from .base_agent import BaseAgent

# Exceptions
from .exceptions import (
    AgentError,
    AgentExecutionError,
    AgentConfigurationError,
    AgentTimeoutError,
    ToolExecutionError,
    InvalidAgentStateError,
)

# Retrieval agent
from .retrieval_agent import (
    create_retrieval_agent,
    invoke_retrieval_agent,
)


__all__ = [
    # Base classes
    'BaseAgent',

    # Exceptions
    'AgentError',
    'AgentExecutionError',
    'AgentConfigurationError',
    'AgentTimeoutError',
    'ToolExecutionError',
    'InvalidAgentStateError',

    # Retrieval agent
    'create_retrieval_agent',
    'invoke_retrieval_agent',
]
