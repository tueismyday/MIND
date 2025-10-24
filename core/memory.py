"""
Memory management for the Agentic RAG Medical Documentation System.
Handles conversation memory and context management.
"""

from langchain_classic.memory import ConversationTokenBufferMemory
from config.llm_config import llm_config
from config.settings import MEMORY_MAX_TOKEN_LIMIT

class MemoryManager:
    """Manages conversation memory for the system."""
    
    def __init__(self):
        self._retrieval_memory = None
        self._retrieved_guidelines = None
    
    @property
    def retrieval_memory(self) -> ConversationTokenBufferMemory:
        """Get the retrieval memory instance."""
        if self._retrieval_memory is None:
            self._retrieval_memory = ConversationTokenBufferMemory(
                llm=llm_config.llm_retrieve, 
                memory_key="chat_history",
                max_token_limit=MEMORY_MAX_TOKEN_LIMIT,
                return_messages=True
            )
        return self._retrieval_memory
    
    @property
    def retrieved_guidelines(self) -> dict:
        """Get the currently retrieved guidelines."""
        return self._retrieved_guidelines
    
    @retrieved_guidelines.setter
    def retrieved_guidelines(self, guidelines: dict):
        """Set the retrieved guidelines."""
        self._retrieved_guidelines = guidelines
    
    def clear_retrieved_guidelines(self):
        """Clear the retrieved guidelines."""
        self._retrieved_guidelines = None
    
    def has_retrieved_guidelines(self) -> bool:
        """Check if guidelines have been retrieved."""
        return self._retrieved_guidelines is not None

# Global memory manager instance
memory_manager = MemoryManager()