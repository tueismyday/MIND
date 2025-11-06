"""
Custom exceptions for the agents module.

This module defines domain-specific exceptions for agent-related errors,
providing clear error handling and meaningful error messages for agent
operations.

Exception Hierarchy:
    AgentError (base)
    ├── AgentExecutionError
    ├── AgentConfigurationError
    ├── AgentTimeoutError
    ├── ToolExecutionError
    └── InvalidAgentStateError

Example:
    >>> try:
    ...     agent.run(invalid_input)
    ... except AgentExecutionError as e:
    ...     logger.error(f"Agent failed: {e}")
    ...     # Handle error appropriately
"""


class AgentError(Exception):
    """
    Base exception for all agent-related errors.

    All agent-specific exceptions should inherit from this class to allow
    for consistent error handling and catching of agent-related errors.

    Example:
        >>> try:
        ...     agent.run(data)
        ... except AgentError as e:
        ...     # Catch any agent-related error
        ...     handle_agent_error(e)
    """
    pass


class AgentExecutionError(AgentError):
    """
    Raised when an agent fails during execution.

    This exception indicates that the agent encountered an error while
    processing input or executing its main logic. The error may be due to
    invalid input, tool failures, LLM errors, or other runtime issues.

    Attributes:
        message: Error description
        agent_name: Name of the agent that failed
        query: The query or input that caused the failure
        cause: Optional underlying exception that caused this error

    Example:
        >>> raise AgentExecutionError(
        ...     "Failed to retrieve patient information",
        ...     agent_name="retrieval_agent",
        ...     query="patient records for ID 12345"
        ... )
    """

    def __init__(
        self,
        message: str,
        agent_name: str = None,
        query: str = None,
        cause: Exception = None
    ):
        """
        Initialize agent execution error.

        Args:
            message: Error description
            agent_name: Name of the agent that failed
            query: The query or input that caused the failure
            cause: Optional underlying exception
        """
        self.message = message
        self.agent_name = agent_name
        self.query = query
        self.cause = cause

        # Build detailed error message
        full_message = message
        if agent_name:
            full_message = f"[{agent_name}] {full_message}"
        if query:
            full_message = f"{full_message} (Query: {query[:100]}...)" if len(query) > 100 else f"{full_message} (Query: {query})"
        if cause:
            full_message = f"{full_message}\nCaused by: {str(cause)}"

        super().__init__(full_message)


class AgentConfigurationError(AgentError):
    """
    Raised when an agent's configuration is invalid.

    This exception indicates that the agent was initialized or configured
    with invalid parameters, missing required settings, or conflicting
    options.

    Example:
        >>> raise AgentConfigurationError(
        ...     "Missing required configuration: 'llm_model'",
        ...     agent_name="retrieval_agent",
        ...     invalid_config={"max_iterations": -1}
        ... )
    """

    def __init__(
        self,
        message: str,
        agent_name: str = None,
        invalid_config: dict = None
    ):
        """
        Initialize agent configuration error.

        Args:
            message: Error description
            agent_name: Name of the agent with invalid config
            invalid_config: The invalid configuration that caused the error
        """
        self.message = message
        self.agent_name = agent_name
        self.invalid_config = invalid_config

        # Build detailed error message
        full_message = message
        if agent_name:
            full_message = f"[{agent_name}] {full_message}"
        if invalid_config:
            full_message = f"{full_message}\nInvalid configuration: {invalid_config}"

        super().__init__(full_message)


class AgentTimeoutError(AgentError):
    """
    Raised when an agent exceeds its execution time limit.

    This exception indicates that the agent took too long to complete its
    task and was terminated to prevent infinite loops or excessive resource
    consumption.

    Example:
        >>> raise AgentTimeoutError(
        ...     "Agent exceeded maximum execution time",
        ...     agent_name="retrieval_agent",
        ...     timeout_seconds=60.0,
        ...     elapsed_seconds=75.3
        ... )
    """

    def __init__(
        self,
        message: str,
        agent_name: str = None,
        timeout_seconds: float = None,
        elapsed_seconds: float = None
    ):
        """
        Initialize agent timeout error.

        Args:
            message: Error description
            agent_name: Name of the agent that timed out
            timeout_seconds: The timeout limit in seconds
            elapsed_seconds: Actual time elapsed before timeout
        """
        self.message = message
        self.agent_name = agent_name
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds

        # Build detailed error message
        full_message = message
        if agent_name:
            full_message = f"[{agent_name}] {full_message}"
        if timeout_seconds and elapsed_seconds:
            full_message = f"{full_message} (Timeout: {timeout_seconds}s, Elapsed: {elapsed_seconds:.2f}s)"

        super().__init__(full_message)


class ToolExecutionError(AgentError):
    """
    Raised when a tool used by an agent fails to execute.

    This exception indicates that one of the tools (functions) called by
    the agent encountered an error during execution. This could be due to
    invalid tool inputs, database connection issues, or other tool-specific
    problems.

    Example:
        >>> raise ToolExecutionError(
        ...     "Failed to retrieve guideline knowledge",
        ...     tool_name="retrieve_guideline_knowledge",
        ...     tool_input="diabetes treatment",
        ...     cause=original_exception
        ... )
    """

    def __init__(
        self,
        message: str,
        tool_name: str = None,
        tool_input: str = None,
        cause: Exception = None
    ):
        """
        Initialize tool execution error.

        Args:
            message: Error description
            tool_name: Name of the tool that failed
            tool_input: Input that was passed to the tool
            cause: Optional underlying exception
        """
        self.message = message
        self.tool_name = tool_name
        self.tool_input = tool_input
        self.cause = cause

        # Build detailed error message
        full_message = message
        if tool_name:
            full_message = f"Tool '{tool_name}': {full_message}"
        if tool_input:
            input_str = tool_input[:100] + "..." if len(tool_input) > 100 else tool_input
            full_message = f"{full_message} (Input: {input_str})"
        if cause:
            full_message = f"{full_message}\nCaused by: {str(cause)}"

        super().__init__(full_message)


class InvalidAgentStateError(AgentError):
    """
    Raised when an agent is in an invalid state for the requested operation.

    This exception indicates that the agent's internal state is inconsistent
    or invalid for the operation being attempted. For example, trying to
    execute an agent before it's been properly initialized.

    Example:
        >>> raise InvalidAgentStateError(
        ...     "Agent must be initialized before execution",
        ...     agent_name="retrieval_agent",
        ...     current_state="uninitialized",
        ...     required_state="initialized"
        ... )
    """

    def __init__(
        self,
        message: str,
        agent_name: str = None,
        current_state: str = None,
        required_state: str = None
    ):
        """
        Initialize invalid agent state error.

        Args:
            message: Error description
            agent_name: Name of the agent with invalid state
            current_state: Description of current state
            required_state: Description of required state
        """
        self.message = message
        self.agent_name = agent_name
        self.current_state = current_state
        self.required_state = required_state

        # Build detailed error message
        full_message = message
        if agent_name:
            full_message = f"[{agent_name}] {full_message}"
        if current_state and required_state:
            full_message = f"{full_message} (Current: {current_state}, Required: {required_state})"

        super().__init__(full_message)
