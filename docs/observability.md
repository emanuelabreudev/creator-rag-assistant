# Observability — guia rápido

O template já vem com `src/observability/trace.py` (structured logging: trace_id por
requisição, latência por operação, eventos de cache/routing). Isso atende a banda
**"Sólido"** da rubrica de Custo/Latência.

Para banda **"Excelente"**, integre **Langfuse**:

1. Conta grátis em https://langfuse.com → projeto novo → copie as chaves.
2. `uv pip install langfuse` e adicione ao `.env`:
   ```
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```
3. Decore as chamadas de LLM:
   ```python
   from langfuse.decorators import observe

   @observe()
   def call_llm(prompt: str, model: str) -> str:
       ...
   ```

Para o README (banda Excelente): screenshot do dashboard Langfuse com 10+ traces,
P95 de latência observado e hit-rate do cache.
