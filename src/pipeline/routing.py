"""Model routing cheap-first com fallback.

TODO 6 (implementado): classify_complexity. Roteia perguntas simples para o modelo
barato (llama-3.1-8b-instant) e perguntas complexas para o premium
(llama-3.3-70b-versatile). Ambos na Groq.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Termos que sinalizam raciocinio/elaboracao -> modelo premium.
_COMPLEX_KW = (
    "explique", "explica", "compare", "compara", "analise", "analisa", "projete",
    "por que", "porque", "como funciona", "passo a passo", "estrategia", "estratégia",
    "diferenca", "diferença", "vantagens e desvantagens", "detalhe", "detalha",
    "elabore", "elabora", "monte um plano", "crie um roteiro",
)


@dataclass(frozen=True)
class RouteDecision:
    model: str
    complexity: str  # "simple" | "complex"
    reason: str


# ------------------------------------------------------------------ TODO 6
def classify_complexity(query: str) -> RouteDecision:
    """Heuristica simples: decide modelo barato vs premium pela forma da query."""
    cheap_model = os.environ.get("CHEAP_MODEL", "llama-3.1-8b-instant")
    premium_model = os.environ.get("PREMIUM_MODEL", "llama-3.3-70b-versatile")

    q = query.strip()
    low = q.lower()

    if any(kw in low for kw in _COMPLEX_KW):
        return RouteDecision(premium_model, "complex", "contem termo que pede raciocinio/explicacao")
    if len(q) > 200:
        return RouteDecision(premium_model, "complex", "query longa/elaborada (>200 chars)")
    if len(q) < 80 and q.endswith("?"):
        return RouteDecision(cheap_model, "simple", "pergunta curta e direta")
    return RouteDecision(cheap_model, "simple", "default cheap-first")


def make_client():
    """Cliente OpenAI-compatible para a Groq."""
    from openai import OpenAI

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Configure GROQ_API_KEY no .env (https://console.groq.com/keys).")
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
