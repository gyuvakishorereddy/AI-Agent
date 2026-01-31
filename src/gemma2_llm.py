"""
Gemma2-9B Q4 LLM Integration with llama-cpp-python
Provides a clean interface for the quantized Gemma2-9B model.
"""

import logging
from typing import Optional, List, Dict
from pathlib import Path

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError as e:
    LLAMA_CPP_AVAILABLE = False

logger = logging.getLogger(__name__)


class Gemma2LLM:
    """Wrapper for Gemma2-9B Q4 quantized model using llama-cpp-python"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,  # 0 = CPU only, increase for GPU
        n_threads: int = 4,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        max_tokens: int = 512,
        verbose: bool = False
    ):
        """
        Initialize Gemma2-9B Q4 model
        
        Args:
            model_path: Path to gemma-2-9b-it-Q4_K_M.gguf file
            n_ctx: Context window size (4096 recommended for Gemma2)
            n_gpu_layers: Number of layers to offload to GPU (0 = CPU only)
            n_threads: Number of CPU threads for inference
            temperature: Sampling temperature (0.7 = balanced)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            max_tokens: Maximum tokens to generate
            verbose: Enable verbose logging
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python is required. Install with: pip install llama-cpp-python")
        
        self.model_path = model_path or self._find_model_path()
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.verbose = verbose
        
        self.llm = None
        self.is_initialized = False
        
        # Try to initialize immediately if model path exists
        if self.model_path and Path(self.model_path).exists():
            self.initialize()
    
    def _find_model_path(self) -> Optional[str]:
        """Try to find Gemma2 model in common locations"""
        common_paths = [
            "models/gemma-2-9b-it-Q4_K_M.gguf",
            "../models/gemma-2-9b-it-Q4_K_M.gguf",
            "../../models/gemma-2-9b-it-Q4_K_M.gguf",
            Path.home() / "models" / "gemma-2-9b-it-Q4_K_M.gguf",
            Path.home() / ".cache" / "lm-studio" / "models" / "gemma-2-9b-it-Q4_K_M.gguf",
        ]
        
        for path in common_paths:
            path_obj = Path(path)
            if path_obj.exists():
                logger.info(f"✅ Found Gemma2 model at: {path_obj}")
                return str(path_obj)
        
        logger.warning("⚠️ Gemma2 model not found in common locations")
        return None
    
    def initialize(self) -> bool:
        """Load the Gemma2 model into memory"""
        if self.is_initialized:
            logger.info("✓ Gemma2 model already initialized")
            return True
        
        if not self.model_path or not Path(self.model_path).exists():
            logger.error(f"❌ Model file not found: {self.model_path}")
            logger.info("📥 Please download Gemma2-9B Q4 model:")
            logger.info("   - From Hugging Face: bartowski/gemma-2-9b-it-GGUF")
            logger.info("   - File: gemma-2-9b-it-Q4_K_M.gguf (~5.5GB)")
            logger.info(f"   - Save to: {Path('models').absolute()}")
            return False
        
        try:
            logger.info(f"🔄 Loading Gemma2-9B Q4 from: {self.model_path}")
            logger.info(f"⚙️ Config: ctx={self.n_ctx}, gpu_layers={self.n_gpu_layers}, threads={self.n_threads}")
            
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                verbose=self.verbose
            )
            
            self.is_initialized = True
            logger.info("✅ Gemma2-9B Q4 model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load Gemma2 model: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """
        Generate response from Gemma2 model
        
        Args:
            prompt: User query/prompt
            context: Optional context to include (from RAG retrieval)
            max_tokens: Override default max_tokens
            temperature: Override default temperature
            top_p: Override default top_p
            top_k: Override default top_k
            stop_sequences: List of strings that stop generation
            
        Returns:
            Generated text response
        """
        if not self.is_initialized:
            if not self.initialize():
                return "Error: Gemma2 model not available. Please configure model path."
        
        # Build full prompt with context (RAG pattern)
        full_prompt = self._build_rag_prompt(prompt, context) if context else prompt
        
        # Generation parameters
        gen_params = {
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
            "top_p": top_p or self.top_p,
            "top_k": top_k or self.top_k,
            "echo": False,
        }
        
        if stop_sequences:
            gen_params["stop"] = stop_sequences
        
        try:
            logger.info(f"🤖 Generating response (prompt length: {len(full_prompt)} chars)")
            
            response = self.llm(
                full_prompt,
                **gen_params
            )
            
            # Extract generated text
            generated_text = response["choices"][0]["text"].strip()
            
            logger.info(f"✅ Generated {len(generated_text)} chars")
            return generated_text
            
        except Exception as e:
            logger.error(f"❌ Generation error: {e}")
            return f"Error generating response: {str(e)}"
    
    def _build_rag_prompt(self, query: str, context: str) -> str:
        """Build RAG-style prompt with context from vector store"""
        # Gemma2 instruction format
        prompt = f"""<start_of_turn>user
You are KARE AI, an intelligent assistant for Kalasalingam Academy of Research and Education (KARE).

Use the following context to answer the user's question accurately and concisely.
If the context doesn't contain relevant information, say so politely.

Context:
{context}

Question: {query}
<end_of_turn>
<start_of_turn>model
"""
        return prompt
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Multi-turn chat interface (for conversation history)
        
        Args:
            messages: List of {"role": "user"/"assistant", "content": "..."}
            max_tokens: Override default max_tokens
            temperature: Override default temperature
            
        Returns:
            Generated response
        """
        if not self.is_initialized:
            if not self.initialize():
                return "Error: Gemma2 model not available."
        
        # Convert messages to Gemma2 chat format
        prompt = self._format_chat_history(messages)
        
        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature or self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                echo=False
            )
            
            return response["choices"][0]["text"].strip()
            
        except Exception as e:
            logger.error(f"❌ Chat error: {e}")
            return f"Error: {str(e)}"
    
    def _format_chat_history(self, messages: List[Dict[str, str]]) -> str:
        """Format conversation history for Gemma2"""
        formatted = ""
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                formatted += f"<start_of_turn>user\n{content}\n<end_of_turn>\n"
            elif role == "assistant":
                formatted += f"<start_of_turn>model\n{content}\n<end_of_turn>\n"
        
        # Add model turn start
        formatted += "<start_of_turn>model\n"
        
        return formatted
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about loaded model"""
        return {
            "model_name": "Gemma2-9B-IT-Q4_K_M",
            "model_path": self.model_path,
            "is_initialized": self.is_initialized,
            "context_window": self.n_ctx,
            "gpu_layers": self.n_gpu_layers,
            "threads": self.n_threads,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
    
    def unload(self):
        """Unload model from memory"""
        if self.llm:
            del self.llm
            self.llm = None
            self.is_initialized = False
            logger.info("✓ Gemma2 model unloaded")


# Global instance (lazy initialization)
_gemma2_instance = None

def get_gemma2_llm(
    model_path: Optional[str] = None,
    **kwargs
) -> Gemma2LLM:
    """Get or create global Gemma2 LLM instance"""
    global _gemma2_instance
    
    if _gemma2_instance is None:
        _gemma2_instance = Gemma2LLM(model_path=model_path, **kwargs)
    
    return _gemma2_instance


if __name__ == "__main__":
    # Test the LLM wrapper
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Gemma2-9B Q4 LLM Wrapper")
    print("=" * 50)
    
    # Initialize model
    llm = Gemma2LLM()
    
    if not llm.is_initialized:
        print("\n⚠️ Model not initialized. Please:")
        print("1. Download gemma-2-9b-it-Q4_K_M.gguf from Hugging Face")
        print("2. Save to 'models/' directory")
        print("3. Update model_path parameter")
    else:
        # Test generation
        test_query = "What is KARE university known for?"
        test_context = "KARE (Kalasalingam Academy of Research and Education) is a deemed university in Tamil Nadu, India. It offers programs in engineering, science, management, and humanities."
        
        print(f"\n📝 Query: {test_query}")
        print(f"📚 Context: {test_context[:100]}...")
        print("\n🤖 Generating response...")
        
        response = llm.generate(test_query, context=test_context)
        
        print(f"\n✅ Response:\n{response}")
        print("\n" + "=" * 50)
        print(llm.get_model_info())
