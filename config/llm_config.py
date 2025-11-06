"""
LLM configuration and client management for the MIND medical documentation system.

This module manages vLLM client instances for both server mode (external API)
and local mode (in-Python instance). Provides a unified interface for LLM
operations with automatic token tracking.
"""

import logging
from typing import Dict, Any, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class VLLMClient:
    """
    vLLM client wrapper for server mode (OpenAI-compatible API).

    Connects to an external vLLM server running with OpenAI-compatible API.
    Supports temperature, top_p, and presence_penalty parameters.

    Attributes:
        model_name: HuggingFace model identifier
        temperature: Default sampling temperature
        base_url: Base URL of the vLLM server
        last_usage: Token usage from last invocation

    Example:
        >>> client = VLLMClient(
        ...     model_name="Qwen/Qwen3-30B",
        ...     temperature=0.2,
        ...     base_url="http://localhost:8000"
        ... )
        >>> response = client.invoke("Hello, world!")
        >>> print(client.last_usage)
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.2,
        base_url: str = "http://localhost:8000"
    ):
        """
        Initialize vLLM client for server mode.

        Args:
            model_name: HuggingFace model identifier
            temperature: Default sampling temperature
            base_url: Base URL of the vLLM server
        """
        self.model_name = model_name
        self.temperature = temperature
        self.base_url = base_url.rstrip('/')
        self.last_usage: Optional[Dict[str, int]] = None

        self.client = OpenAI(
            api_key="EMPTY",  # vLLM doesn't require API key
            base_url=f"{self.base_url}/v1",
        )

        logger.debug(
            f"Initialized VLLMClient: model={model_name}, "
            f"base_url={self.base_url}, temperature={temperature}"
        )

    def invoke(self, prompt: str, **kwargs) -> str:
        """
        Invoke the language model with the given prompt.

        Maintains compatibility with langchain interface.
        Captures token usage information for tracking purposes.

        Note: Server mode (OpenAI API) only supports:
            temperature, top_p, presence_penalty, frequency_penalty
        The top_k and min_p parameters are NOT supported in server mode.

        Args:
            prompt: Input prompt text
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)

        Returns:
            Generated text response

        Raises:
            RuntimeError: If vLLM generation fails or returns None

        Example:
            >>> response = client.invoke(
            ...     "What is diabetes?",
            ...     temperature=0.5,
            ...     max_tokens=1000
            ... )
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', 6056),
                top_p=kwargs.get('top_p', 0.95),
                presence_penalty=kwargs.get('presence_penalty', 1.5),
            )

            # Capture token usage information if available
            if hasattr(response, 'usage') and response.usage:
                self.last_usage = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
                logger.debug(
                    f"Token usage: prompt={self.last_usage['prompt_tokens']}, "
                    f"completion={self.last_usage['completion_tokens']}, "
                    f"total={self.last_usage['total_tokens']}"
                )
            else:
                self.last_usage = None

            # Safety check for None response
            content = response.choices[0].message.content

            if content is None:
                logger.error("vLLM returned None response")
                logger.debug(f"Prompt length: {len(prompt)} characters")
                raise RuntimeError(
                    "vLLM returned empty response - prompt may be too long or server issue"
                )

            return content.strip()

        except AttributeError as e:
            # Specific error for None.strip()
            logger.error(f"vLLM response was None: {e}", exc_info=True)
            raise RuntimeError(
                "vLLM generation failed: Response was None "
                "(likely context overflow or server error)"
            ) from e

        except Exception as e:
            logger.error(f"vLLM generation error: {e}", exc_info=True)
            raise RuntimeError(f"vLLM generation failed: {e}") from e


class InPythonVLLMClient:
    """
    vLLM client for local mode (in-Python model loading).

    Loads the vLLM model directly in the Python process for local inference.
    Supports all sampling parameters including top_k and min_p.

    Attributes:
        model_name: HuggingFace model identifier
        temperature: Default sampling temperature
        llm: vLLM LLM instance
        last_usage: Token usage from last invocation

    Example:
        >>> client = InPythonVLLMClient(
        ...     model_name="Qwen/Qwen3-30B",
        ...     temperature=0.2,
        ...     gpu_memory_utilization=0.75,
        ...     max_model_len=14000
        ... )
        >>> response = client.invoke("Hello, world!")
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.2,
        gpu_memory_utilization: float = 0.75,
        max_model_len: int = 14000,
        max_num_seqs: int = 1
    ):
        """
        Initialize in-Python vLLM client for local mode.

        Args:
            model_name: HuggingFace model identifier
            temperature: Default sampling temperature
            gpu_memory_utilization: GPU memory fraction (0.0-1.0)
            max_model_len: Maximum context length
            max_num_seqs: Maximum number of sequences to process in parallel

        Raises:
            ImportError: If vLLM is not installed
        """
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "vLLM is not installed. Install it with: pip install vllm"
            ) from e

        self.model_name = model_name
        self.temperature = temperature
        self.last_usage: Optional[Dict[str, int]] = None
        self.SamplingParams = SamplingParams

        logger.info("Initializing in-Python vLLM instance...")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  GPU Memory Utilization: {gpu_memory_utilization}")
        logger.info(f"  Max Model Length: {max_model_len}")
        logger.info(f"  Max Num Seqs: {max_num_seqs}")

        # Initialize vLLM with local model
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            trust_remote_code=True  # Required for some models like Qwen
        )

        logger.info("vLLM model loaded successfully!")

    def invoke(self, prompt: str, **kwargs) -> str:
        """
        Invoke the language model with the given prompt.

        Maintains compatibility with langchain interface.
        Captures token usage information for tracking purposes.

        Note: Local mode (vLLM native API) supports ALL parameters:
            temperature, top_p, top_k, min_p, presence_penalty

        Args:
            prompt: Input prompt text
            **kwargs: Additional generation parameters

        Returns:
            Generated text response

        Raises:
            RuntimeError: If vLLM generation fails or returns None

        Example:
            >>> response = client.invoke(
            ...     "What is diabetes?",
            ...     temperature=0.5,
            ...     top_k=20,
            ...     max_tokens=1000
            ... )
        """
        try:
            # Create sampling parameters with all generation settings
            sampling_params = self.SamplingParams(
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', 6056),
                top_p=kwargs.get('top_p', 0.95),
                top_k=kwargs.get('top_k', 20),
                min_p=kwargs.get('min_p', 0.0),
                presence_penalty=kwargs.get('presence_penalty', 1.5),
            )

            # Generate response
            outputs = self.llm.generate([prompt], sampling_params)

            if not outputs or len(outputs) == 0:
                logger.error("vLLM returned no outputs")
                raise RuntimeError("vLLM returned empty response")

            output = outputs[0]

            # Capture token usage information
            if hasattr(output, 'metrics') and output.metrics:
                self.last_usage = {
                    'prompt_tokens': len(output.prompt_token_ids),
                    'completion_tokens': sum(len(o.token_ids) for o in output.outputs),
                    'total_tokens': (
                        len(output.prompt_token_ids) +
                        sum(len(o.token_ids) for o in output.outputs)
                    )
                }
                logger.debug(
                    f"Token usage: prompt={self.last_usage['prompt_tokens']}, "
                    f"completion={self.last_usage['completion_tokens']}, "
                    f"total={self.last_usage['total_tokens']}"
                )
            else:
                self.last_usage = None

            # Extract generated text
            generated_text = output.outputs[0].text

            if generated_text is None:
                logger.error("vLLM returned None response")
                logger.debug(f"Prompt length: {len(prompt)} characters")
                raise RuntimeError(
                    "vLLM returned empty response - prompt may be too long"
                )

            return generated_text.strip()

        except Exception as e:
            logger.error(f"vLLM local generation error: {e}", exc_info=True)
            raise RuntimeError(f"vLLM local generation failed: {e}") from e


class LLMConfig:
    """
    Configuration manager for vLLM instances in server or local mode.

    Manages the creation and caching of vLLM client instances
    for different purposes (retrieve, generate, critique).
    Uses lazy initialization to avoid loading models until needed.

    Attributes:
        mode: Operating mode ("server" or "local")
        base_url: Server URL (server mode only)
        model_name: HuggingFace model identifier
        gpu_memory_utilization: GPU memory fraction (local mode only)
        max_model_len: Max context length (local mode only)
        max_num_seqs: Max parallel sequences (local mode only)

    Example:
        >>> from config.vllm import VLLMConfig as VLLMSettings
        >>> from config.generation import GenerationConfig
        >>> vllm_settings = VLLMSettings()
        >>> gen_settings = GenerationConfig()
        >>> llm_config = LLMConfig(
        ...     mode=vllm_settings.vllm_mode,
        ...     model_name=vllm_settings.vllm_model_name,
        ...     base_url=vllm_settings.vllm_server_url,
        ...     temperature=gen_settings.temperature
        ... )
        >>> llm = llm_config.llm_generate
        >>> response = llm.invoke("Hello, world!")
    """

    def __init__(
        self,
        mode: str = "server",
        base_url: str = "http://localhost:8000",
        model_name: str = "cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit",
        temperature: float = 0.2,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 14000,
        max_num_seqs: int = 1
    ):
        """
        Initialize LLM configuration manager.

        Args:
            mode: "server" or "local"
            base_url: Server URL (server mode)
            model_name: HuggingFace model identifier
            temperature: Default sampling temperature
            gpu_memory_utilization: GPU memory fraction (local mode)
            max_model_len: Max context length (local mode)
            max_num_seqs: Max parallel sequences (local mode)

        Raises:
            ValueError: If mode is invalid
        """
        # Validate mode
        if mode not in ["server", "local"]:
            raise ValueError(
                f"Invalid vLLM mode: {mode}. Must be 'server' or 'local'"
            )

        self.mode = mode
        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs

        logger.info(f"LLM Configuration Mode: {self.mode}")
        logger.info(f"LLM Model: {self.model_name}")

        if self.mode == "server":
            logger.info(f"LLM Server URL: {self.base_url}")
        else:
            logger.info(f"LLM GPU Memory Utilization: {self.gpu_memory_utilization}")
            logger.info(f"LLM Max Model Length: {self.max_model_len}")

        # Lazy initialization of clients
        self._llm_retrieve: Optional[Any] = None
        self._llm_generate: Optional[Any] = None
        self._llm_critique: Optional[Any] = None

    def _create_client(self, temperature: float) -> Any:
        """
        Create a client instance based on the current mode.

        Args:
            temperature: Sampling temperature for this client

        Returns:
            VLLMClient or InPythonVLLMClient instance
        """
        if self.mode == "server":
            return VLLMClient(
                model_name=self.model_name,
                temperature=temperature,
                base_url=self.base_url
            )
        else:  # local mode
            return InPythonVLLMClient(
                model_name=self.model_name,
                temperature=temperature,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                max_num_seqs=self.max_num_seqs
            )

    @property
    def llm_retrieve(self) -> Any:
        """
        Get the retrieval LLM instance (lazy initialization).

        Returns:
            VLLMClient or InPythonVLLMClient for retrieval tasks
        """
        if self._llm_retrieve is None:
            logger.debug("Initializing LLM for retrieval")
            self._llm_retrieve = self._create_client(self.temperature)
        return self._llm_retrieve

    @property
    def llm_generate(self) -> Any:
        """
        Get the generation LLM instance (lazy initialization).

        Returns:
            VLLMClient or InPythonVLLMClient for generation tasks
        """
        if self._llm_generate is None:
            logger.debug("Initializing LLM for generation")
            self._llm_generate = self._create_client(self.temperature)
        return self._llm_generate

    @property
    def llm_critique(self) -> Any:
        """
        Get the critique LLM instance (lazy initialization).

        Returns:
            VLLMClient or InPythonVLLMClient for critique tasks
        """
        if self._llm_critique is None:
            logger.debug("Initializing LLM for critique")
            self._llm_critique = self._create_client(self.temperature)
        return self._llm_critique


# Global LLM configuration instance
# Note: This will be initialized with default settings from settings.py
# when first imported
llm_config: Optional[LLMConfig] = None


def get_llm_config() -> LLMConfig:
    """
    Get the global LLM configuration instance.

    Creates the instance using settings from config.settings if not already created.
    This provides backward compatibility with existing code.

    Returns:
        Global LLMConfig instance

    Example:
        >>> llm_config = get_llm_config()
        >>> llm = llm_config.llm_generate
    """
    global llm_config

    if llm_config is None:
        # Import here to avoid circular dependency
        from .vllm import VLLMConfig as VLLMSettings
        from .generation import GenerationConfig

        vllm_settings = VLLMSettings()
        gen_settings = GenerationConfig()

        logger.info("Creating global LLMConfig instance")

        llm_config = LLMConfig(
            mode=vllm_settings.vllm_mode,
            base_url=vllm_settings.vllm_server_url,
            model_name=vllm_settings.vllm_model_name,
            temperature=gen_settings.temperature,
            gpu_memory_utilization=vllm_settings.vllm_gpu_memory_utilization,
            max_model_len=vllm_settings.vllm_max_model_len,
            max_num_seqs=vllm_settings.vllm_max_num_seqs
        )

    return llm_config


# Initialize global instance
llm_config = get_llm_config()
