"""
Robust error handling for RAG and LLM operations.
Provides retry logic and graceful degradation.
"""

import time
from typing import Callable, Any, Optional, List, Dict, Tuple
from functools import wraps


class GenerationError(Exception):
    """Base exception for document generation errors"""
    pass


class RAGError(GenerationError):
    """RAG search failures"""
    pass


class LLMError(GenerationError):
    """LLM invocation failures"""
    pass


class ParseError(GenerationError):
    """Parsing/extraction failures"""
    pass


def retry_on_failure(max_retries: int = 3, 
                     delay: float = 2.0,
                     exponential_backoff: bool = True):
    """
    Decorator to retry function on failure with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        exponential_backoff: Whether to exponentially increase delay
    """
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        print(f"[RETRY] {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}")
                        print(f"[RETRY] Retrying in {current_delay:.1f} seconds...")
                        time.sleep(current_delay)
                        
                        if exponential_backoff:
                            current_delay *= 2
                    else:
                        print(f"[ERROR] {func.__name__} failed after {max_retries + 1} attempts")
            
            # All retries exhausted
            raise last_exception
        
        return wrapper
    return decorator


def safe_rag_search(query: str, 
                   max_references: int = 3,
                   note_types: List[str] = None,
                   fallback_result: Optional[Any] = None) -> Any:
    """
    Safely perform RAG search with error handling.
    
    Args:
        query: Search query
        max_references: Max sources to retrieve
        fallback_result: Result to return on complete failure
        
    Returns:
        Search result or fallback
    """
    
    from tools.patient_tools import get_patient_info_with_sources, PatientInfoResult
    
    try:
        result = get_patient_info_with_sources(query, max_references=max_references, note_types=note_types)
        return result
        
    except Exception as e:
        print(f"[ERROR] RAG search failed for query '{query[:50]}...': {str(e)}")
        
        # Return empty fallback result
        if fallback_result is None:
            fallback_result = PatientInfoResult(
                content="Ingen information kunne hentes fra patientjournal (teknisk fejl)",
                sources=[],
                max_references=0,
                search_method="failed"
            )
        
        return fallback_result


def safe_llm_invoke(prompt: str, 
                   llm_instance: Any,
                   max_retries: int = 3,
                   fallback_response: str = "LLM kunne ikke generere svar") -> str:
    """
    Safely invoke LLM with retry logic.
    
    Args:
        prompt: LLM prompt
        llm_instance: LLM instance to use
        max_retries: Max retry attempts
        fallback_response: Response on complete failure
        
    Returns:
        LLM response or fallback
    """
        
    last_error = None
    
    for attempt in range(max_retries):
        try:
            response = llm_instance.invoke(
                prompt,
                    tags=["safe_invoke", f"attempt_{attempt + 1}"]
            )
            return response
            
        except Exception as e:
            last_error = e
            print(f"[ERROR] LLM invocation failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
    
    # All retries failed
    print(f"[CRITICAL] LLM invocation failed after {max_retries} attempts: {last_error}")
    return fallback_response