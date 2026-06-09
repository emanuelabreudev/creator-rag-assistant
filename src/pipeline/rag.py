"""RAG pipeline — ingest, chunk, embed, index, retrieve, generate.

Stack do modulo M2 (mantida no projeto):
  - LLM:        Groq (endpoint OpenAI-compatible) — llama-3.3-70b-versatile / 8b-instant
  - Embeddings: LOCAIS via sentence-transformers (Groq nao expoe /v1/embeddings)
                modelo MULTILINGUE (corpus em PT-BR) — paraphrase-multilingual-MiniLM-L12-v2

Implementa os TODOs 1-3 do template e adiciona `answer_with_tools` (function-calling
de verdade com a tool de dominio de tools.py).
"""

from __future__ import annotations

import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pypdf import PdfReader

from src.pipeline.tools import TOOLS, run_tool_call

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_LLM = "llama-3.3-70b-versatile"
DEFAULT_EMBED = "paraphrase-multilingual-MiniLM-L12-v2"  # PT-BR friendly

PROMPT_TEMPLATE = """Voce e um assistente para criadores de conteudo. Responda APENAS com base
no contexto abaixo. Se a informacao nao estiver no contexto, diga "Nao encontrado no corpus".
Sempre cite a fonte usando o formato [arquivo:pagina]. Responda em portugues, de forma direta
e pratica.

CONTEXTO:
{context}

PERGUNTA: {question}

RESPOSTA:"""

SYSTEM_TOOLS = """Voce e um assistente para criadores de conteudo (foco em saude e bem-estar).
Responda com base no CONTEXTO abaixo e cite a fonte como [arquivo:pagina]. Se a informacao nao
estiver no contexto, diga "Nao encontrado no corpus".

Voce tem uma ferramenta `analisar_legenda`: use-a SOMENTE quando o usuario colar/enviar uma
legenda/caption e pedir uma analise, revisao ou checagem dela (tamanho, hashtags, sinalizacao de
publicidade, CTA). Para perguntas conceituais normais, NAO use a ferramenta — apenas responda
com base no contexto.

CONTEXTO:
{context}"""


def _make_client() -> OpenAI:
    """Cliente OpenAI-compatible apontando para a Groq."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Configure GROQ_API_KEY no .env (gere em https://console.groq.com/keys)."
        )
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def _read_documents(corpus_dir: Path) -> list[dict]:
    """Le .pdf (por pagina), .md e .txt (arquivo inteiro = 1 'pagina')."""
    docs: list[dict] = []
    for path in sorted(corpus_dir.iterdir()):
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append({"text": text, "source": path.name, "page": page_idx + 1})
        elif path.suffix.lower() in (".md", ".txt"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                docs.append({"text": text, "source": path.name, "page": 1})
    return docs


class RAGPipeline:
    """Pipeline RAG end-to-end com Chroma local + Groq."""

    def __init__(
        self,
        corpus_dir: str = "data/corpus",
        persist_dir: str = "data/chroma",
        collection_name: str = "docs",
        llm_model: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        self.client = _make_client()
        self.llm_model = llm_model or os.environ.get("LLM_MODEL", DEFAULT_LLM)
        self.embed_model = embed_model or os.environ.get("EMBED_MODEL", DEFAULT_EMBED)

        # Embeddings LOCAIS (Groq nao faz embeddings). Modelo multilingue p/ PT-BR.
        self.embed_fn = SentenceTransformerEmbeddingFunction(model_name=self.embed_model)

        self.corpus_dir = Path(corpus_dir)
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        chroma = chromadb.PersistentClient(path=persist_dir)
        self.collection = chroma.get_or_create_collection(
            name=collection_name, embedding_function=self.embed_fn
        )

    # ------------------------------------------------------------------ TODO 1
    def ingest_and_index(self) -> int:
        """Le os documentos do corpus, faz chunking e indexa em Chroma."""
        # TODO 1.A — ingestao
        docs = _read_documents(self.corpus_dir)
        if not docs:
            raise RuntimeError(
                f"Nenhum documento legivel em {self.corpus_dir} (use .pdf, .md ou .txt)."
            )

        # TODO 1.B — chunking recursivo 800/100
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks: list[dict] = []
        for doc in docs:
            for i, piece in enumerate(splitter.split_text(doc["text"])):
                chunks.append(
                    {
                        "id": f"{doc['source']}-p{doc['page']}-c{i}",
                        "text": piece,
                        "source": doc["source"],
                        "page": doc["page"],
                    }
                )

        # TODO 1.C — indexar em lotes (embeddings locais -> sem rate limit)
        batch = 200
        for start in range(0, len(chunks), batch):
            lote = chunks[start : start + batch]
            self.collection.add(
                ids=[c["id"] for c in lote],
                documents=[c["text"] for c in lote],
                metadatas=[{"source": c["source"], "page": c["page"]} for c in lote],
            )

        return self.collection.count()

    # ------------------------------------------------------------------ TODO 2
    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Busca top-k chunks similares a query."""
        result = self.collection.query(query_texts=[query], n_results=k)
        return [
            {
                "text": result["documents"][0][i],
                "source": result["metadatas"][0][i]["source"],
                "page": result["metadatas"][0][i]["page"],
                "distance": result["distances"][0][i],
            }
            for i in range(len(result["documents"][0]))
        ]

    @staticmethod
    def _build_context(hits: list[dict]) -> str:
        return "\n\n---\n\n".join(f"[{h['source']}:p{h['page']}]\n{h['text']}" for h in hits)

    # ------------------------------------------------------------------ TODO 3
    def answer(self, question: str, k: int = 5, model: str | None = None) -> dict:
        """RAG puro: retrieve + augment + generate. Retorna {answer, sources}."""
        hits = self.retrieve(question, k=k)
        context = self._build_context(hits)
        resp = self.client.chat.completions.create(
            model=model or self.llm_model,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(context=context, question=question),
                }
            ],
            temperature=0.0,
        )
        return {
            "answer": resp.choices[0].message.content,
            "sources": [(h["source"], h["page"]) for h in hits],
        }

    # ---------------------------------------------- orquestrador com tool-use
    def answer_with_tools(self, question: str, k: int = 5, model: str | None = None) -> dict:
        """RAG + function-calling. O LLM pode chamar `analisar_legenda` quando fizer sentido.

        Retorna {answer, sources, tool_calls}.
        """
        model = model or self.llm_model
        hits = self.retrieve(question, k=k)
        context = self._build_context(hits)
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_TOOLS.format(context=context)},
            {"role": "user", "content": question},
        ]
        tool_calls_made: list[dict] = []

        msg = None
        for _ in range(3):  # no maximo 3 idas ao modelo (evita loop infinito)
            resp = self.client.chat.completions.create(
                model=model, messages=messages, tools=TOOLS, temperature=0.0
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )
            for tc in msg.tool_calls:
                result = run_tool_call(tc.function.name, tc.function.arguments)
                tool_calls_made.append(
                    {"name": tc.function.name, "arguments": tc.function.arguments}
                )
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        return {
            "answer": (msg.content if msg else None) or "(sem resposta)",
            "sources": [(h["source"], h["page"]) for h in hits],
            "tool_calls": tool_calls_made,
        }


def build_rag_pipeline(corpus_dir: str = "data/corpus") -> RAGPipeline:
    """Factory: cria pipeline e indexa o corpus se ainda nao indexado."""
    pipeline = RAGPipeline(corpus_dir=corpus_dir)
    if pipeline.collection.count() == 0:
        pipeline.ingest_and_index()
    return pipeline
