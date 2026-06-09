"""Bench de custo/latencia — gera os numeros da tabela do README.

Roda N queries (com repeticoes/parafrases para exercitar o cache) em 3 cenarios:
  1. baseline  — sempre modelo premium, sem cache
  2. + cache   — exact + semantic cache ligados
  3. + routing — cache + routing cheap-first

Uso:
    python scripts/bench.py            # usa o corpus em data/corpus
Requer GROQ_API_KEY no .env. Custos sao ESTIMADOS pela tabela de precos da Groq
(input/output por 1M tokens) — ajuste PRICES se mudar de modelo.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
load_dotenv()

from src.pipeline.cache import ExactCache, SemanticCache  # noqa: E402
from src.pipeline.rag import build_rag_pipeline  # noqa: E402
from src.pipeline.routing import classify_complexity  # noqa: E402

# Preco aproximado Groq (USD por 1M tokens) — input/output.
PRICES = {
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
}

# Mistura de queries: algumas repetidas e parafraseadas (exercita o cache).
QUERIES = [
    "Recebi um produto de graca, preciso marcar publicidade?",
    "Ganhei um item de uma marca, tenho que sinalizar como publi?",  # parafrase
    "Posso usar qualquer musica nos meus Reels?",
    "Sou nutricionista, posso postar antes e depois de cliente?",
    "Quais dados posso pedir num sorteio sem ferir a LGPD?",
    "Recebi um produto de graca, preciso marcar publicidade?",  # repeticao exata
    "Explique a diferenca entre alcance e retencao como metricas.",
    "Qual o limite de caracteres de uma legenda no Instagram?",
]


def _cost(model: str, usage) -> float:
    pin, pout = PRICES.get(model, (0.59, 0.79))
    return (usage.prompt_tokens * pin + usage.completion_tokens * pout) / 1_000_000


def run(scenario: str, pipe, use_cache: bool, use_routing: bool):
    exact, sem = ExactCache(), SemanticCache(threshold=0.90)
    total_cost, total_ms, hits = 0.0, 0.0, 0
    for q in QUERIES:
        t0 = time.perf_counter()
        if use_cache and (exact.get(q) or sem.get(q)):
            hits += 1
            total_ms += (time.perf_counter() - t0) * 1000
            continue
        model = classify_complexity(q).model if use_routing else "llama-3.3-70b-versatile"
        hits_ctx = pipe.retrieve(q, k=5)
        ctx = pipe._build_context(hits_ctx)
        from src.pipeline.rag import PROMPT_TEMPLATE

        resp = pipe.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(context=ctx, question=q)}],
            temperature=0.0,
        )
        total_cost += _cost(model, resp.usage)
        total_ms += (time.perf_counter() - t0) * 1000
        if use_cache:
            ans = resp.choices[0].message.content
            exact.put(q, ans)
            sem.put(q, ans)
    n = len(QUERIES)
    print(
        f"{scenario:<22} custo=${total_cost:.5f}  "
        f"lat_media={total_ms / n:7.0f}ms  cache_hits={hits}/{n}"
    )
    return total_cost


if __name__ == "__main__":
    pipe = build_rag_pipeline(corpus_dir=str(_ROOT / "data" / "corpus"))
    print(f"chunks indexados: {pipe.collection.count()}\n")
    base = run("baseline (premium)", pipe, use_cache=False, use_routing=False)
    run("+ cache", pipe, use_cache=True, use_routing=False)
    routed = run("+ cache + routing", pipe, use_cache=True, use_routing=True)
    if base:
        print(f"\nreducao de custo (baseline -> cache+routing): {(1 - routed / base) * 100:.0f}%")
