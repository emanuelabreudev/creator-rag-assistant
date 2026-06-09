"""Streamlit UI — assistente para criadores de conteudo. Deploy 1-click no Streamlit Cloud.

Fluxo: exact cache -> semantic cache -> routing (cheap/premium) -> RAG + tool-use.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

load_dotenv()

import streamlit as st  # noqa: E402

from src.observability.trace import log_event, trace  # noqa: E402
from src.pipeline.cache import ExactCache, SemanticCache  # noqa: E402
from src.pipeline.rag import build_rag_pipeline  # noqa: E402
from src.pipeline.routing import classify_complexity  # noqa: E402

st.set_page_config(page_title="Assistente do Criador", page_icon="🎬", layout="centered")

st.title("🎬 Assistente do Criador de Conteúdo")
st.caption(
    "Tire dúvidas sobre publicidade, regras de saúde/estética, direitos autorais e LGPD — "
    "com a fonte citada. Cole uma legenda e peça uma análise para acionar a ferramenta."
)


@st.cache_resource(show_spinner=False)
def get_pipeline():
    return build_rag_pipeline(corpus_dir=str(_ROOT / "data" / "corpus"))


@st.cache_resource(show_spinner=False)
def get_exact_cache():
    return ExactCache()


@st.cache_resource(show_spinner=False)
def get_semantic_cache():
    return SemanticCache(threshold=0.90)


with st.spinner("Inicializando pipeline RAG (primeira vez baixa o modelo de embeddings)..."):
    pipeline = get_pipeline()
    exact_cache = get_exact_cache()
    semantic_cache = get_semantic_cache()


with st.sidebar:
    st.header("Métricas")
    st.metric("Chunks indexados", pipeline.collection.count())
    st.metric("Exact cache", exact_cache.stats()["size"])
    st.metric("Semantic cache", semantic_cache.stats()["size"])
    st.divider()
    st.subheader("Exemplos")
    st.markdown(
        "- Recebi um produto de graça, preciso marcar publicidade?\n"
        "- Sou nutricionista, posso postar antes e depois de cliente?\n"
        "- Posso usar qualquer música nos meus Reels?\n"
        "- Analisa essa legenda pro Instagram: *Amei esse whey! Link na bio 💪 #fit*"
    )

query = st.text_input("Sua pergunta:", placeholder="Pergunte algo ou cole uma legenda para analisar...")

if query:
    with trace("query_handle", query=query) as ctx:
        trace_id = ctx["trace_id"]

        # 1. Exact cache
        cached = exact_cache.get(query)
        if cached:
            st.success("⚡ Cache hit (exact)")
            st.write(cached)
            log_event("cache_hit", trace_id=trace_id, layer="exact")
            st.stop()

        # 2. Semantic cache
        try:
            cached = semantic_cache.get(query)
        except Exception:  # noqa: BLE001
            cached = None
        if cached:
            st.success("⚡ Cache hit (semantic)")
            st.write(cached)
            log_event("cache_hit", trace_id=trace_id, layer="semantic")
            st.stop()

        # 3. Routing cheap-first
        decision = classify_complexity(query)
        st.info(f"Routing: {decision.complexity} → `{decision.model}` ({decision.reason})")
        log_event("route_decision", trace_id=trace_id, **decision.__dict__)

        # 4. RAG + tool-use, no modelo escolhido pelo router
        with st.spinner("Pensando..."):
            result = pipeline.answer_with_tools(query, model=decision.model)

        if result.get("tool_calls"):
            st.caption("🛠️ Ferramenta usada: " + ", ".join(t["name"] for t in result["tool_calls"]))

        st.write(result["answer"])

        if result.get("sources"):
            with st.expander("Fontes citadas"):
                vistas = set()
                for source, page in result["sources"]:
                    chave = f"{source}:p{page}"
                    if chave not in vistas:
                        st.write(f"- `{chave}`")
                        vistas.add(chave)

        # 5. Cacheia para a proxima
        exact_cache.put(query, result["answer"])
        semantic_cache.put(query, result["answer"])
        log_event("answer_generated", trace_id=trace_id, sources=len(result.get("sources", [])))

st.divider()
st.caption(
    "Conteúdo educativo, não é aconselhamento jurídico ou médico. "
    "LLM: Groq · Embeddings: sentence-transformers (local) · Vector store: Chroma."
)
