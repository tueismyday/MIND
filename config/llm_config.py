"""
LLM configuration for the Agentic RAG Medical Documentation System.
Manages the setup and configuration of different LLM instances using vLLM.
"""
from openai import OpenAI
from .settings import (
    TEMPERATURE,
    TOP_P,
    TOP_K,
    MIN_P,
    PRESENCE_PENALTY,
    VLLM_MODE,
    VLLM_SERVER_URL,
    VLLM_MODEL_NAME,
    VLLM_GPU_MEMORY_UTILIZATION,
    VLLM_MAX_MODEL_LEN,
    VLLM_MAX_NUM_SEQS
)

class VLLMClient:
    """vLLM client wrapper compatible with your existing interface."""

    def __init__(self, model_name, temperature=None, base_url="http://localhost:8000"):
        self.model_name = model_name
        self.temperature = temperature if temperature is not None else TEMPERATURE
        self.base_url = base_url.rstrip('/')
        self.last_usage = None  # Store token usage from last invocation

        self.client = OpenAI(
            api_key="EMPTY",  # vLLM doesn't require API key
            base_url=f"{self.base_url}/v1",
        )
    
    def invoke(self, prompt, **kwargs):
        """
        Invoke method to maintain compatibility with langchain interface.
        Captures token usage information for tracking purposes.
        Uses hard defaults from settings.py for generation parameters.
        
        Note: Server mode (OpenAI API) only supports: temperature, top_p, presence_penalty, frequency_penalty
        top_k and min_p are NOT supported in server mode (only in local mode).
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', 6056),
                top_p=kwargs.get('top_p', TOP_P),
                presence_penalty=kwargs.get('presence_penalty', PRESENCE_PENALTY),
            )

            # Capture token usage information if available
            if hasattr(response, 'usage') and response.usage:
                self.last_usage = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            else:
                self.last_usage = None

            # Safety check for None response
            content = response.choices[0].message.content

            if content is None:
                print(f"[ERROR] vLLM returned None response")
                print(f"[DEBUG] Prompt length: {len(prompt)} chars")
                raise RuntimeError("vLLM returned empty response - prompt may be too long or server issue")

            return content.strip()
            
        except AttributeError as e:
            # Specific error for None.strip()
            print(f"[ERROR] vLLM response was None")
            raise RuntimeError(f"vLLM generation failed: Response was None (likely context overflow or server error)")
        except Exception as e:
            print(f"[ERROR] vLLM error: {str(e)}")
            raise RuntimeError(f"vLLM generation failed: {e}")

class InPythonVLLMClient:
    """vLLM client for in-process model loading (local mode)."""

    def __init__(self, model_name, temperature=None,
                 gpu_memory_utilization=0.75,
                 max_model_len=14000,
                 max_num_seqs=1):
        """
        Initialize in-Python vLLM client.

        Args:
            model_name: HuggingFace model name
            temperature: Default sampling temperature (uses TEMPERATURE from settings if None)
            gpu_memory_utilization: GPU memory fraction for model (0.0-1.0)
            max_model_len: Maximum context length
            max_num_seqs: Maximum number of sequences to process in parallel
        """
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM is not installed. Install it with: pip install vllm"
            )

        self.model_name = model_name
        self.temperature = temperature if temperature is not None else TEMPERATURE
        self.last_usage = None
        self.SamplingParams = SamplingParams

        print(f"[vLLM LOCAL] Initializing in-Python vLLM instance...")
        print(f"[vLLM LOCAL] Model: {model_name}")
        print(f"[vLLM LOCAL] GPU Memory Utilization: {gpu_memory_utilization}")
        print(f"[vLLM LOCAL] Max Model Length: {max_model_len}")
        print(f"[vLLM LOCAL] Max Num Seqs: {max_num_seqs}")

        # Initialize vLLM with local model
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            trust_remote_code=True  # Required for some models like Qwen
        )

        print(f"[vLLM LOCAL] Model loaded successfully!")

    def invoke(self, prompt, **kwargs):
        """
        Invoke method to maintain compatibility with langchain interface.
        Captures token usage information for tracking purposes.
        Uses hard defaults from settings.py for all generation parameters.
        
        Note: Local mode (vLLM native API) supports ALL parameters including:
        temperature, top_p, top_k, min_p, presence_penalty
        """
        try:
            # Create sampling parameters with all generation settings
            sampling_params = self.SamplingParams(
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', 6056),
                top_p=kwargs.get('top_p', TOP_P),
                top_k=kwargs.get('top_k', TOP_K),
                min_p=kwargs.get('min_p', MIN_P),
                presence_penalty=kwargs.get('presence_penalty', PRESENCE_PENALTY),
            )

            # Generate response
            outputs = self.llm.generate([prompt], sampling_params)

            if not outputs or len(outputs) == 0:
                print(f"[ERROR] vLLM returned no outputs")
                raise RuntimeError("vLLM returned empty response")

            output = outputs[0]

            # Capture token usage information
            if hasattr(output, 'metrics') and output.metrics:
                metrics = output.metrics
                self.last_usage = {
                    'prompt_tokens': len(output.prompt_token_ids),
                    'completion_tokens': sum(len(o.token_ids) for o in output.outputs),
                    'total_tokens': len(output.prompt_token_ids) + sum(len(o.token_ids) for o in output.outputs)
                }
            else:
                self.last_usage = None

            # Extract generated text
            generated_text = output.outputs[0].text

            if generated_text is None:
                print(f"[ERROR] vLLM returned None response")
                print(f"[DEBUG] Prompt length: {len(prompt)} chars")
                raise RuntimeError("vLLM returned empty response - prompt may be too long")

            return generated_text.strip()

        except Exception as e:
            print(f"[ERROR] vLLM local generation error: {str(e)}")
            raise RuntimeError(f"vLLM local generation failed: {e}")

class LLMConfig:
    """Configuration class for managing vLLM instances in server or local mode."""

    def __init__(self,
                 mode=None,
                 base_url=None,
                 model_name=None,
                 gpu_memory_utilization=None,
                 max_model_len=None,
                 max_num_seqs=None):
        """
        Initialize LLM configuration.

        Args:
            mode: "server" or "local" (defaults to VLLM_MODE from settings)
            base_url: Server URL for server mode (defaults to VLLM_SERVER_URL)
            model_name: Model name (defaults to VLLM_MODEL_NAME)
            gpu_memory_utilization: GPU memory fraction for local mode (defaults to VLLM_GPU_MEMORY_UTILIZATION)
            max_model_len: Max context length for local mode (defaults to VLLM_MAX_MODEL_LEN)
            max_num_seqs: Max parallel sequences for local mode (defaults to VLLM_MAX_NUM_SEQS)
        """

        # Use provided values or fall back to settings
        self.mode = mode if mode is not None else VLLM_MODE
        self.base_url = base_url if base_url is not None else VLLM_SERVER_URL
        self.model_name = model_name if model_name is not None else VLLM_MODEL_NAME
        self.gpu_memory_utilization = gpu_memory_utilization if gpu_memory_utilization is not None else VLLM_GPU_MEMORY_UTILIZATION
        self.max_model_len = max_model_len if max_model_len is not None else VLLM_MAX_MODEL_LEN
        self.max_num_seqs = max_num_seqs if max_num_seqs is not None else VLLM_MAX_NUM_SEQS

        # Validate mode
        if self.mode not in ["server", "local"]:
            raise ValueError(f"Invalid VLLM_MODE: {self.mode}. Must be 'server' or 'local'")

        print(f"[LLM CONFIG] Mode: {self.mode}")
        print(f"[LLM CONFIG] Model: {self.model_name}")
        if self.mode == "server":
            print(f"[LLM CONFIG] Server URL: {self.base_url}")
        else:
            print(f"[LLM CONFIG] GPU Memory Utilization: {self.gpu_memory_utilization}")
            print(f"[LLM CONFIG] Max Model Length: {self.max_model_len}")

        self._llm_retrieve = None
        self._llm_generate = None
        self._llm_critique = None
    
    def _create_client(self, temperature):
        """Create a client instance based on the current mode."""
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
    def llm_retrieve(self):
        """Get the retrieval LLM instance (server or local based on mode)."""
        if self._llm_retrieve is None:
            self._llm_retrieve = self._create_client(TEMPERATURE)
        return self._llm_retrieve

    @property
    def llm_generate(self):
        """Get the generation LLM instance (server or local based on mode)."""
        if self._llm_generate is None:
            self._llm_generate = self._create_client(TEMPERATURE)
        return self._llm_generate

    @property
    def llm_critique(self):
        """Get the critique LLM instance (server or local based on mode)."""
        if self._llm_critique is None:
            self._llm_critique = self._create_client(TEMPERATURE)
        return self._llm_critique

# Global LLM configuration instance
llm_config = LLMConfig()
