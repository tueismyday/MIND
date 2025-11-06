"""
Memory management for conversation context in the MIND system.

This module handles conversation memory and context management for the
retrieval system, maintaining chat history and guideline cache for
efficient document generation.

Architecture:
    The MemoryManager class provides centralized management of:
    - Conversation memory: Token-buffered chat history
    - Retrieved guidelines: Cached guideline content

    The conversation memory uses a token buffer to limit memory usage
    while maintaining relevant context for multi-turn interactions.

Key Classes:
    MemoryManager: Singleton manager for memory and context

Dependencies:
    - langchain_classic.memory: Conversation memory implementation
    - config.llm_config: LLM configuration for token counting
    - config.settings: Memory configuration (token limits)

Singleton Pattern:
    The module provides a global memory_manager instance for consistent
    memory state across the application.

Example:
    >>> from core.memory import memory_manager
    >>> # Access retrieval memory
    >>> memory = memory_manager.retrieval_memory
    >>> memory.save_context({"input": "query"}, {"output": "response"})
    >>>
    >>> # Store retrieved guidelines
    >>> memory_manager.retrieved_guidelines = {"section": "content"}
    >>> if memory_manager.has_retrieved_guidelines():
    ...     guidelines = memory_manager.retrieved_guidelines
"""

import logging
from typing import Dict, Any, Optional

from langchain_classic.memory import ConversationTokenBufferMemory

from config.llm_config import llm_config
from config.settings import MEMORY_MAX_TOKEN_LIMIT

# Configure logger for this module
logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manages conversation memory and context for the MIND system.

    This class provides centralized memory management for conversation
    history and guideline caching. The retrieval memory uses token
    buffering to maintain relevant context while preventing excessive
    memory usage.

    Attributes:
        _retrieval_memory: Cached conversation memory instance
        _retrieved_guidelines: Cached guideline content

    Properties:
        retrieval_memory: Get or create retrieval memory instance
        retrieved_guidelines: Get cached guidelines
        has_retrieved_guidelines: Check if guidelines are cached

    Example:
        >>> manager = MemoryManager()
        >>> memory = manager.retrieval_memory
        >>> manager.retrieved_guidelines = {"intro": "guidelines..."}
    """

    def __init__(self):
        """
        Initialize the memory manager.

        Sets up internal state but uses lazy initialization for the
        actual memory instance to minimize startup overhead.
        """
        logger.debug("Initializing MemoryManager")
        self._retrieval_memory: Optional[ConversationTokenBufferMemory] = None
        self._retrieved_guidelines: Optional[Dict[str, Any]] = None
        logger.debug("MemoryManager initialized")

    @property
    def retrieval_memory(self) -> ConversationTokenBufferMemory:
        """
        Get the retrieval conversation memory instance.

        Lazy loads the memory instance on first access. The memory uses
        token buffering to maintain relevant context within the configured
        token limit.

        Returns:
            ConversationTokenBufferMemory instance configured for retrieval

        Example:
            >>> memory = memory_manager.retrieval_memory
            >>> memory.save_context(
            ...     {"input": "What is diabetes?"},
            ...     {"output": "Diabetes is..."}
            ... )
        """
        if self._retrieval_memory is None:
            logger.debug(
                f"Creating retrieval memory with max token limit: {MEMORY_MAX_TOKEN_LIMIT}"
            )
            self._retrieval_memory = ConversationTokenBufferMemory(
                llm=llm_config.llm_retrieve,
                memory_key="chat_history",
                max_token_limit=MEMORY_MAX_TOKEN_LIMIT,
                return_messages=True
            )
            logger.info("Retrieval memory created successfully")
        return self._retrieval_memory

    @property
    def retrieved_guidelines(self) -> Optional[Dict[str, Any]]:
        """
        Get the currently cached retrieved guidelines.

        Returns:
            Dictionary containing guideline sections and content, or None
            if no guidelines have been retrieved yet

        Example:
            >>> guidelines = memory_manager.retrieved_guidelines
            >>> if guidelines:
            ...     print(f"Sections: {list(guidelines.keys())}")
        """
        return self._retrieved_guidelines

    @retrieved_guidelines.setter
    def retrieved_guidelines(self, guidelines: Dict[str, Any]) -> None:
        """
        Set the retrieved guidelines cache.

        Args:
            guidelines: Dictionary containing guideline sections and content

        Example:
            >>> memory_manager.retrieved_guidelines = {
            ...     "introduction": "Medical guidelines for...",
            ...     "diagnosis": "Diagnostic criteria..."
            ... }
        """
        logger.debug(
            f"Caching retrieved guidelines "
            f"({len(guidelines)} sections)" if guidelines else "(clearing cache)"
        )
        self._retrieved_guidelines = guidelines

    def clear_retrieved_guidelines(self) -> None:
        """
        Clear the cached retrieved guidelines.

        This should be called when starting a new document generation
        task or when guidelines need to be refreshed.

        Example:
            >>> memory_manager.clear_retrieved_guidelines()
            >>> assert not memory_manager.has_retrieved_guidelines()
        """
        logger.debug("Clearing retrieved guidelines cache")
        self._retrieved_guidelines = None

    def has_retrieved_guidelines(self) -> bool:
        """
        Check if guidelines have been retrieved and cached.

        Returns:
            True if guidelines are cached, False otherwise

        Example:
            >>> if not memory_manager.has_retrieved_guidelines():
            ...     # Retrieve guidelines
            ...     guidelines = retrieve_guidelines()
            ...     memory_manager.retrieved_guidelines = guidelines
        """
        return self._retrieved_guidelines is not None

    def clear_all(self) -> None:
        """
        Clear all cached memory and guidelines.

        This completely resets the memory manager state, clearing both
        conversation history and guideline cache. Useful for starting
        fresh or cleaning up after completing a task.

        Example:
            >>> memory_manager.clear_all()
            >>> assert not memory_manager.has_retrieved_guidelines()
        """
        logger.info("Clearing all memory and caches")

        # Clear retrieval memory if it exists
        if self._retrieval_memory is not None:
            logger.debug("Clearing retrieval memory")
            self._retrieval_memory.clear()

        # Clear guidelines
        self.clear_retrieved_guidelines()

        logger.info("All memory and caches cleared")


# Global memory manager instance (singleton pattern)
# Note: This is not thread-safe. For multi-threaded applications,
# consider using thread-local storage or separate instances per thread.
memory_manager = MemoryManager()
