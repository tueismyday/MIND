"""
Retrieval agent implementation for the MIND medical documentation system.

This module implements a multi-tool retrieval agent that uses LangChain's
structured chat agent framework to retrieve and synthesize information from
multiple sources including patient records, clinical guidelines, and generated
documents.

The agent uses conversational memory to maintain context across interactions
and can perform multi-step reasoning to answer complex medical queries.

Key Classes:
    RetrievalAgent: Main retrieval agent implementation

Key Functions:
    create_retrieval_agent: Factory function to create configured agent
    invoke_retrieval_agent: Convenience function to invoke agent with query

Dependencies:
    - LangChain: Agent framework and prompting
    - config.llm_config: LLM model configuration
    - config.settings: Agent behavior settings
    - core.memory: Conversational memory management
    - tools.*: Specialized retrieval tools

Example:
    >>> from agents.retrieval_agent import invoke_retrieval_agent
    >>> response = invoke_retrieval_agent("Hvad er diabetes type 2?")
    >>> print(response)

Architecture:
    The retrieval agent follows a structured chat pattern:
    1. Receives user query
    2. Analyzes query and plans retrieval strategy
    3. Executes tools (retrieve_patient_info, retrieve_guideline_knowledge, etc.)
    4. Synthesizes information from multiple sources
    5. Returns comprehensive answer in Danish
"""

from typing import Any, Dict, List, Optional
import logging

from langchain_classic.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

from config.llm_config import llm_config
from config.settings import (
    MAX_AGENT_ITERATIONS,
    AGENT_EARLY_STOPPING_METHOD,
    CHAT_HISTORY_LIMIT
)
from core.memory import memory_manager
from tools.patient_tools import retrieve_patient_info
from tools.guideline_tools import retrieve_guideline_knowledge
from tools.document_tools import retrieve_generated_document_info, start_document_generation
from utils.profiling import profile
from agents.exceptions import AgentExecutionError, AgentConfigurationError


logger = logging.getLogger(__name__)


# System prompt template for the retrieval agent
RETRIEVAL_AGENT_SYSTEM_PROMPT = """You are an intelligent multi-step retrieval (RAG) agent working in a hospital.
You are expected to think aloud, reason step-by-step, and use multiple tools if needed to fully understand and answer the user's question.

Recent conversation history:
{formatted_history}

Please refer to this conversation history when answering questions that reference previous interactions.
Your goal is to simulate how an expert clinician-researcher would retrieve and synthesize information from multiple sources to give a clear, accurate answer – in Danish.

Rules:
    - Use the 'start_document_generation' tool if the user asks to create a document
    - Never repeat the same tool with the same input more than once.
    - Stop when you have enough information.
    - Always answer in Danish.

Available tools:
    - retrieve_patient_info: Access patient records, input your search criterias for the vectordatabase in danish
    - retrieve_generated_document_info: Get information about finalized documents, input your search criterias for the vectordatabase in danish
    - start_document_generation: Launch creation of a medical document. Use this if the user asks you to create a document.
    - retrieve_guideline_knowledge: Get information about medical guidelines, input your search criterias for the vectordatabase in danish

Think step-by-step:
    - First, analyze the user's question in detail.
    - Then hypothesize what information you may need to answer it.
    - Then decide which tool is best for retrieving that information.
    - After each tool, reflect: did this bring you closer to the answer? Should you expand, follow up, or rerank?
    - You are allowed to perform multiple tool actions in sequence before finalizing your answer, especially when the question requires combining patient-specific and guideline-based knowledge.
    - Before you finalize, make sure to check if the information you have is sufficient to answer the question. If any information is redundant or irrelevant, you should remove it.

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {{tool_names}}

Provide only ONE action per $JSON_BLOB, as shown:

```
{{{{
 "action": $TOOL_NAME,
 "action_input": $INPUT
}}}}
```
Where $INPUT is your query for the RAG.

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{{{
 "action": "Final Answer",
 "action_input": "Final response to human"
}}}}
```"""


def _format_chat_history(chat_history: List[Any], max_messages: int = None) -> str:
    """
    Format chat history for inclusion in agent prompt.

    Converts LangChain message objects into a human-readable string format
    suitable for inclusion in the agent's system prompt. Applies optional
    message limit to prevent prompt overflow.

    Args:
        chat_history: List of LangChain message objects
        max_messages: Maximum number of recent messages to include (None = all)

    Returns:
        Formatted string with chat history

    Example:
        >>> messages = [HumanMessage(content="Hej"), AIMessage(content="Hej!")]
        >>> formatted = _format_chat_history(messages, max_messages=10)
        >>> print(formatted)
        Human: Hej
        Assistant: Hej!
    """
    if not chat_history:
        return ""

    # Limit to most recent messages if specified
    if max_messages is not None:
        chat_history = chat_history[-max_messages:]

    formatted_lines = []
    for message in chat_history:
        if hasattr(message, "content") and hasattr(message, "type"):
            role = "Human" if message.type == "human" else "Assistant"
            formatted_lines.append(f"{role}: {message.content}")

    return "\n".join(formatted_lines)


def _get_default_tools() -> List[BaseTool]:
    """
    Get the default set of tools for the retrieval agent.

    Returns:
        List of tool instances available to the agent

    Note:
        This function provides a centralized location for tool configuration,
        making it easy to add, remove, or modify available tools.
    """
    return [
        retrieve_guideline_knowledge,
        retrieve_patient_info,
        retrieve_generated_document_info,
        start_document_generation
    ]


def _create_agent_prompt(
    formatted_history: str,
    tools: List[BaseTool]
) -> ChatPromptTemplate:
    """
    Create the prompt template for the retrieval agent.

    Constructs a ChatPromptTemplate with system instructions, user input,
    and agent scratchpad. The prompt is partially filled with tool names.

    Args:
        formatted_history: Pre-formatted chat history string
        tools: List of tools available to the agent

    Returns:
        Configured ChatPromptTemplate instance

    Note:
        The prompt follows LangChain's structured chat agent format with
        JSON-based action specification.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", RETRIEVAL_AGENT_SYSTEM_PROMPT),
        ("user", "{input}"),
        ("assistant", "{agent_scratchpad}"),
    ])

    # Partial fill with formatted history and tool names
    tool_names = ", ".join([tool.name for tool in tools])
    prompt = prompt.partial(
        formatted_history=formatted_history,
        tool_names=tool_names
    )

    return prompt


def create_retrieval_agent(
    llm: Optional[Any] = None,
    tools: Optional[List[BaseTool]] = None,
    memory: Optional[Any] = None,
    max_iterations: Optional[int] = None,
    early_stopping_method: Optional[str] = None,
    verbose: bool = True
) -> AgentExecutor:
    """
    Create a retrieval agent with appropriate tools, memory, and configuration.

    This factory function creates a fully configured AgentExecutor with
    conversational memory and specialized retrieval tools. The agent uses
    a structured chat pattern for multi-step reasoning.

    Args:
        llm: Language model to use. If None, uses llm_config.llm_retrieve
        tools: List of tools for the agent. If None, uses default tools
        memory: Memory instance. If None, uses memory_manager.retrieval_memory
        max_iterations: Maximum agent iterations. If None, uses MAX_AGENT_ITERATIONS
        early_stopping_method: Early stopping strategy. If None, uses AGENT_EARLY_STOPPING_METHOD
        verbose: Enable verbose output for debugging

    Returns:
        Configured AgentExecutor ready to process queries

    Raises:
        AgentConfigurationError: If required configuration is invalid or missing

    Example:
        >>> agent = create_retrieval_agent(verbose=True)
        >>> response = agent.invoke({"input": "Hvad er diabetes?"})
        >>> print(response['output'])

    Note:
        This function supports dependency injection for all major components,
        making it easy to test with mocks or use custom configurations.
    """
    logger.info("Creating retrieval agent")

    # Use defaults if not provided (dependency injection)
    if llm is None:
        try:
            llm = llm_config.llm_retrieve
        except Exception as e:
            raise AgentConfigurationError(
                "Failed to load LLM from llm_config",
                agent_name="retrieval_agent",
                invalid_config={"llm": None}
            ) from e

    if tools is None:
        tools = _get_default_tools()

    if memory is None:
        memory = memory_manager.retrieval_memory

    if max_iterations is None:
        max_iterations = MAX_AGENT_ITERATIONS

    if early_stopping_method is None:
        early_stopping_method = AGENT_EARLY_STOPPING_METHOD

    # Validate configuration
    if not tools:
        raise AgentConfigurationError(
            "Agent must have at least one tool",
            agent_name="retrieval_agent",
            invalid_config={"tools": tools}
        )

    if max_iterations < 1:
        raise AgentConfigurationError(
            "max_iterations must be at least 1",
            agent_name="retrieval_agent",
            invalid_config={"max_iterations": max_iterations}
        )

    logger.debug(
        "Agent configuration",
        extra={
            "num_tools": len(tools),
            "tool_names": [tool.name for tool in tools],
            "max_iterations": max_iterations,
            "early_stopping_method": early_stopping_method,
            "verbose": verbose
        }
    )

    # Load and format chat history from memory
    try:
        memory_variables = memory.load_memory_variables({})
        chat_history = memory_variables.get("chat_history", [])
        chat_history = chat_history[-CHAT_HISTORY_LIMIT:]
        formatted_history = _format_chat_history(chat_history)

        logger.debug(
            "Loaded chat history",
            extra={
                "num_messages": len(chat_history),
                "limit": CHAT_HISTORY_LIMIT
            }
        )
    except Exception as e:
        logger.warning(f"Failed to load chat history: {e}")
        formatted_history = ""

    # Create agent prompt
    prompt = _create_agent_prompt(formatted_history, tools)

    # Create structured chat agent
    try:
        agent = create_structured_chat_agent(
            llm=llm,
            tools=tools,
            prompt=prompt
        )
    except Exception as e:
        raise AgentConfigurationError(
            "Failed to create structured chat agent",
            agent_name="retrieval_agent"
        ) from e

    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=max_iterations,
        early_stopping_method=early_stopping_method
    )

    logger.info(
        "Retrieval agent created successfully",
        extra={
            "num_tools": len(tools),
            "max_iterations": max_iterations
        }
    )

    return agent_executor


@profile
def invoke_retrieval_agent(
    query: str,
    agent_executor: Optional[AgentExecutor] = None
) -> str:
    """
    Invoke the retrieval agent to process a query.

    This is a convenience function that creates an agent (if not provided)
    and invokes it with the given query. The function handles response
    parsing and error handling, returning a clean string response.

    The function is decorated with @profile for performance monitoring,
    allowing tracking of agent invocation time and resource usage.

    Args:
        query: User query to process. Should be in Danish for best results.
        agent_executor: Optional pre-configured agent. If None, creates new agent.

    Returns:
        Agent's response as a string

    Raises:
        AgentExecutionError: If agent invocation fails

    Example:
        >>> response = invoke_retrieval_agent("Hvad er symptomerne på diabetes?")
        >>> print(response)

        >>> # With custom agent
        >>> custom_agent = create_retrieval_agent(verbose=False)
        >>> response = invoke_retrieval_agent("Patientjournal?", custom_agent)

    Note:
        Each invocation creates a new agent by default. For multiple queries
        in the same session, consider creating an agent once and reusing it
        for better performance and memory continuity.
    """
    logger.info(
        "Invoking retrieval agent",
        extra={"query_length": len(query)}
    )
    logger.debug(f"Query: {query}")

    # Create agent if not provided
    if agent_executor is None:
        try:
            agent_executor = create_retrieval_agent()
        except AgentConfigurationError as e:
            logger.error(f"Failed to create agent: {e}")
            raise AgentExecutionError(
                "Failed to create retrieval agent",
                agent_name="retrieval_agent",
                query=query,
                cause=e
            ) from e

    # Invoke agent
    try:
        logger.debug("Executing agent with query")
        response = agent_executor.invoke(
            input={"input": query}
        )
        logger.debug("Agent execution completed successfully")
    except Exception as e:
        logger.error(
            f"Agent execution failed: {e}",
            exc_info=True,
            extra={"query": query}
        )
        raise AgentExecutionError(
            f"Agent execution failed: {str(e)}",
            agent_name="retrieval_agent",
            query=query,
            cause=e
        ) from e

    # Parse response
    if isinstance(response, dict) and 'output' in response:
        result = response['output']
        logger.info(
            "Agent invocation successful",
            extra={
                "response_length": len(result),
                "response_type": "dict"
            }
        )
        return result
    elif isinstance(response, str):
        logger.info(
            "Agent invocation successful",
            extra={
                "response_length": len(response),
                "response_type": "str"
            }
        )
        return response
    else:
        logger.warning(
            f"Unexpected agent response format: {type(response)}",
            extra={"response_type": str(type(response))}
        )
        return "Unexpected agent response format."
