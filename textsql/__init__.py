"""textsql: fine-tune, quantize, serve, and benchmark a small coder model
for text-to-SQL on Spider.

The package is deliberately split so the *same* evaluation and latency code
can drive any serving backend (local HF, vLLM, llama.cpp, or any
OpenAI-compatible server). That is what makes the quantization x serving
comparison matrix cheap to run later.
"""

__version__ = "0.1.0"
