"""Smoke tests — rodam sem rede/secrets (logica pura). O teste de integracao
(pipeline completo) so roda se GROQ_API_KEY estiver setada.

    uv run pytest -q     # ou: pytest -q
"""

from __future__ import annotations

import json
import os

import pytest

from src.pipeline.routing import classify_complexity
from src.pipeline.tools import analisar_legenda, run_tool_call
from src.pipeline.cache import ExactCache


def test_routing_simple_vs_complex():
    simples = classify_complexity("Preciso marcar #publi?")
    complexa = classify_complexity("Explique a diferenca entre parceria paga e recebido.")
    assert simples.complexity == "simple"
    assert complexa.complexity == "complex"
    assert simples.model != complexa.model


def test_routing_returns_reason():
    d = classify_complexity("oi")
    assert d.reason and d.model and d.complexity in {"simple", "complex"}


def test_tool_detecta_falta_de_publi():
    relatorio = analisar_legenda("Amei esse whey! Comprei e recomendo demais.", "instagram")
    data = json.loads(relatorio)
    assert data["sinaliza_publicidade"] is False
    assert any("publicidade" in a.lower() for a in data["alertas"])


def test_tool_detecta_estouro_de_limite():
    legenda = "a" * 300
    data = json.loads(analisar_legenda(legenda, "x"))  # X/Twitter = 280
    assert data["dentro_do_limite"] is False
    assert data["caracteres"] == 300


def test_tool_ok_quando_completa():
    legenda = "Parceria paga com a marca! Salve este post 💪 #publi"
    data = json.loads(analisar_legenda(legenda, "instagram"))
    assert data["sinaliza_publicidade"] is True
    assert data["tem_cta"] is True


def test_run_tool_call_via_json():
    out = run_tool_call("analisar_legenda", json.dumps({"legenda": "teste #publi", "plataforma": "tiktok"}))
    assert "tiktok" in out


def test_run_tool_call_tool_inexistente():
    assert run_tool_call("nao_existe", "{}").startswith("ERROR")


def test_exact_cache():
    c = ExactCache()
    assert c.get("Oi") is None
    c.put("Oi", "Ola!")
    assert c.get("oi") == "Ola!"  # normaliza case/espaco
    assert c.stats()["size"] == 1


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="precisa de GROQ_API_KEY")
def test_pipeline_integration():
    from src.pipeline.rag import build_rag_pipeline

    pipe = build_rag_pipeline(corpus_dir="data/corpus")
    assert pipe.collection.count() > 0
    hits = pipe.retrieve("preciso marcar publicidade?", k=3)
    assert len(hits) == 3
    out = pipe.answer_with_tools("Recebi um produto de graca, preciso sinalizar?")
    assert isinstance(out["answer"], str) and out["answer"]
