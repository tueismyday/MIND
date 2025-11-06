"""
Base agent interface for the MIND medical documentation system.

This module defines the abstract base class that all agents should inherit from,
providing a common interface for agent creation, execution, and state management.

Key Classes:
    BaseAgent: Abstract base class for all agents in the system

Design Principles:
    - Agents encapsulate specific task logic (retrieval, generation, etc.)
    - Agents maintain their own state and configuration
    - Agents provide a consistent interface for execution and reset
    - Agents support dependency injection for testability

Example:
    >>> class MyAgent(BaseAgent):
    ...     def run(self, input_data: str) -> str:
    ...         return f"Processing: {input_data}"
    ...     def reset(self) -> None:
    ...         pass
    >>> agent = MyAgent("my_agent", {"verbose": True})
    >>> result = agent.run("test input")
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging


logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the MIND system.

    All agents should inherit from this class and implement the required
    abstract methods. This ensures a consistent interface across all agent
    implementations and facilitates testing and composition.

    Attributes:
        name: Unique identifier for this agent instance
        config: Configuration dictionary containing agent-specific settings
        logger: Logger instance for this agent
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base agent.

        Args:
            name: Unique identifier for this agent instance
            config: Optional configuration dictionary. If None, uses empty dict.
                   Common config keys:
                   - verbose (bool): Enable verbose logging
                   - max_iterations (int): Maximum iterations for agent loops
                   - timeout (float): Timeout in seconds for agent execution

        Example:
            >>> agent = MyAgent(
            ...     name="retrieval_agent",
            ...     config={"verbose": True, "max_iterations": 10}
            ... )
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")

        # Log agent initialization
        self.logger.info(
            f"Initializing agent '{name}'",
            extra={"config": self.config}
        )

    @abstractmethod
    def run(self, input_data: Any) -> Any:
        """
        Execute the agent's main logic.

        This method should be implemented by all agent subclasses to define
        the agent's behavior. The method receives input data, processes it
        according to the agent's logic, and returns the result.

        Args:
            input_data: Input data to process. Type depends on agent implementation.

        Returns:
            Processed result. Type depends on agent implementation.

        Raises:
            AgentExecutionError: If agent execution fails

        Example:
            >>> result = agent.run("What is diabetes?")
            >>> print(result)
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the agent's internal state.

        This method should clear any internal state, cached data, or
        conversation history maintained by the agent. Useful for starting
        fresh conversations or recovering from errors.

        Example:
            >>> agent.reset()  # Clear conversation history and state
        """
        pass

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value with an optional default.

        Args:
            key: Configuration key to retrieve
            default: Default value if key is not found

        Returns:
            Configuration value or default if not found

        Example:
            >>> max_iter = agent.get_config_value("max_iterations", 10)
            >>> verbose = agent.get_config_value("verbose", False)
        """
        return self.config.get(key, default)

    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update agent configuration with new values.

        Args:
            updates: Dictionary of configuration updates to apply

        Example:
            >>> agent.update_config({"verbose": True, "max_iterations": 20})
        """
        self.config.update(updates)
        self.logger.debug(
            f"Updated configuration for agent '{self.name}'",
            extra={"updates": updates}
        )

    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(name='{self.name}')"
