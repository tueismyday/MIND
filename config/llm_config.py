"""
LLM configuration for the Agentic RAG Medical Documentation System.
Manages the setup and configuration of different LLM instances using vLLM.
"""
from openai import OpenAI
import httpx
from .settings import GENERATION_TEMPERATURE, CRITIQUE_TEMPERATURE

class VLLMClient:
    """vLLM client wrapper compatible with your existing interface."""

    def __init__(self, model_name, temperature=0.7, base_url="http://localhost:8000", timeout=120.0):
        self.model_name = model_name
        self.temperature = temperature
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.last_usage = None  # Store token usage from last invocation
        self._create_client()

    def _create_client(self):
        """Create or recreate the OpenAI client with proper timeout configuration."""
        self.client = OpenAI(
            api_key="EMPTY",  # vLLM doesn't require API key
            base_url=f"{self.base_url}/v1",
            timeout=httpx.Timeout(self.timeout, connect=10.0),
            max_retries=0,  # We handle retries at a higher level
        )
    
    def invoke(self, prompt, **kwargs):
        """
        Invoke method to maintain compatibility with langchain interface.
        Captures token usage information for tracking purposes.
        Handles connection errors by recreating the client.
        """
        # Filter out unsupported kwargs that might be passed by retry logic
        supported_kwargs = ['temperature', 'max_tokens', 'top_p']
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_kwargs}

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=filtered_kwargs.get('temperature', self.temperature),
                max_tokens=filtered_kwargs.get('max_tokens', 6056),
                top_p=filtered_kwargs.get('top_p', 0.9),
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
        except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError, httpx.TimeoutException) as e:
            # Connection-related errors - recreate client for next attempt
            print(f"[ERROR] vLLM connection error: {str(e)}")
            print(f"[INFO] Recreating client connection for next retry...")
            self._create_client()
            raise RuntimeError(f"vLLM connection error: {e}")
        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] vLLM error: {error_msg}")

            # If we see a 500 error or connection issue, recreate the client
            if "500" in error_msg or "Connection" in error_msg or "EngineCore" in error_msg:
                print(f"[INFO] Detected server error - recreating client for next retry...")
                self._create_client()

            raise RuntimeError(f"vLLM generation failed: {e}")

class LLMConfig:
    """Configuration class for managing vLLM instances."""
    
    def __init__(self, base_url="http://localhost:8000", model_name="cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"):  

        ### POSSIBLE MODELS ###
        #leon-se/gemma-3-27b-it-qat-W4A16-G128 
        #cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit 
        #openai/gpt-oss-20b
        
        self.base_url = base_url
        self.model_name = model_name
        self._llm_retrieve = None
        self._llm_generate = None
        self._llm_critique = None
    
    @property
    def llm_retrieve(self) -> VLLMClient:
        """Get the retrieval LLM instance."""
        if self._llm_retrieve is None:
            self._llm_retrieve = VLLMClient(
                model_name=self.model_name,
                temperature=GENERATION_TEMPERATURE,
                base_url=self.base_url
            )
        return self._llm_retrieve
    
    @property
    def llm_generate(self) -> VLLMClient:
        """Get the generation LLM instance."""
        if self._llm_generate is None:
            self._llm_generate = VLLMClient(
                model_name=self.model_name,
                temperature=GENERATION_TEMPERATURE,
                base_url=self.base_url
            )
        return self._llm_generate
    
    @property
    def llm_critique(self) -> VLLMClient:
        """Get the critique LLM instance."""
        if self._llm_critique is None:
            self._llm_critique = VLLMClient(
                model_name=self.model_name,
                temperature=CRITIQUE_TEMPERATURE,
                base_url=self.base_url
            )
        return self._llm_critique

# Global LLM configuration instance
llm_config = LLMConfig()