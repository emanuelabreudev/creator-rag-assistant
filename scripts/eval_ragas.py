"""Avaliacao RAGAS do assistente — golden set sobre o guia de criadores de conteudo.

Roda o pipeline real (build_rag_pipeline), coleta resposta + contextos de cada pergunta
e calcula faithfulness, answer_relevancy e context_precision. Judge na Groq
(llama-3.1-8b-instant — TPD alto, aguenta a suite no free tier) e embeddings locais.

Instalar os extras de avaliacao:
    pip install -e ".[eval]"        # ragas, datasets, langchain-openai, langchain-huggingface

Rodar:
    python scripts/eval_ragas.py

Saida: imprime a tabela por pergunta + as 3 medias no formato do formulario, e salva
`ragas_creators_report.csv`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
load_dotenv()

from datasets import Dataset  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import answer_relevancy, context_precision, faithfulness  # noqa: E402
from ragas.run_config import RunConfig  # noqa: E402

from src.pipeline.rag import build_rag_pipeline  # noqa: E402

# Golden set: 12 perguntas com ground-truth ancorado no guia (guia-criadores-conteudo.md).
GOLDEN: list[dict] = [
    {
        "question": "Recebi um produto de graca de uma marca, preciso sinalizar como publicidade?",
        "ground_truth": "Sim. Qualquer contrapartida da marca (inclusive produto de graca) torna o conteudo publicidade, e ele precisa ser sinalizado de forma clara, no inicio.",
    },
    {
        "question": "Como sinalizar corretamente uma parceria paga?",
        "ground_truth": "Use a ferramenta de parceria paga/conteudo de marca da plataforma ou escreva #publi/#publicidade no inicio da legenda, de forma clara; nao esconda a sinalizacao no meio das hashtags.",
    },
    {
        "question": "Preciso avisar que ganho comissao em link de afiliado?",
        "ground_truth": "Sim. Links de afiliado e cupons geram comissao e sao publicidade; avise que voce ganha comissao.",
    },
    {
        "question": "Posso prometer cura com um suplemento?",
        "ground_truth": "Nao. Evite prometer cura ou resultado garantido; use linguagem honesta como 'pode ajudar' e nao atribua a um suplemento propriedades terapeuticas que ele nao tem.",
    },
    {
        "question": "Sou nutricionista, posso postar antes e depois de cliente?",
        "ground_truth": "Antes e depois e fortemente regulado e muitos conselhos restringem ou proibem; se usar, deixe claro que resultados variam de pessoa para pessoa e nunca prometa o mesmo resultado para todos.",
    },
    {
        "question": "Posso usar qualquer musica nos meus Reels?",
        "ground_truth": "Nao. Use o catalogo de audio licenciado da plataforma; contas comerciais/posts de publicidade tem acesso reduzido ao catalogo, e usar musica protegida fora do catalogo pode gerar bloqueio ou claim de direitos autorais.",
    },
    {
        "question": "Posso usar imagens de terceiros nos meus posts?",
        "ground_truth": "Nao sem permissao ou licenca. Prefira bancos com licenca adequada (ex.: Creative Commons para uso comercial) e de credito quando a licenca exigir.",
    },
    {
        "question": "Quais dados posso pedir num sorteio sem ferir a LGPD?",
        "ground_truth": "Apenas os necessarios (minimizacao) — normalmente nome e uma forma de contato; nao peca CPF ou dados sensiveis sem necessidade, informe a finalidade e use consentimento opt-in.",
    },
    {
        "question": "O que faz um bom gancho num video?",
        "ground_truth": "Uma promessa ou tensao nos primeiros segundos que faz a pessoa parar de rolar; evite abrir com 'oi gente, tudo bem?', que desperdica o momento de maior atencao.",
    },
    {
        "question": "Qual o limite de caracteres de uma legenda no Instagram?",
        "ground_truth": "Cerca de 2.200 caracteres; use as primeiras duas linhas como chamada porque o resto fica escondido atras de 'mais'.",
    },
    {
        "question": "Quais metricas importam mais do que numero de curtidas?",
        "ground_truth": "Retencao media, salvamentos e compartilhamentos, e taxa de cliques (CTR); seguidores e curtidas sao metricas de vaidade quando isoladas.",
    },
    {
        "question": "Posso anunciar 'emagreca 10kg em 7 dias' no meu infoproduto?",
        "ground_truth": "Nao. Promessa de resultado garantido (especialmente em saude/estetica) e enganosa e pode gerar punicao; use depoimentos reais com autorizacao e deixe claro que resultados variam.",
    },
]


def main() -> None:
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("Configure GROQ_API_KEY no .env antes de rodar a avaliacao.")

    pipe = build_rag_pipeline(corpus_dir=str(_ROOT / "data" / "corpus"))
    print(f"chunks indexados: {pipe.collection.count()} | golden set: {len(GOLDEN)} perguntas\n")

    records = []
    for it in GOLDEN:
        out = pipe.answer(it["question"], k=5)  # RAG puro p/ eval (sem tool)
        hits = pipe.retrieve(it["question"], k=5)
        records.append(
            {
                "user_input": it["question"],
                "response": out["answer"],
                "retrieved_contexts": [h["text"] for h in hits],
                "reference": it["ground_truth"],
            }
        )
    ds = Dataset.from_list(records)

    api_key = os.environ["GROQ_API_KEY"]
    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=os.environ.get("CHEAP_MODEL", "llama-3.1-8b-instant"),
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
            temperature=0.0,
        )
    )
    judge_embed = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=os.environ.get("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
        )
    )
    # 1 worker + backoff longo: respeita o TPM do free tier e re-tenta 429 em vez de NaN.
    run_cfg = RunConfig(max_workers=1, timeout=180, max_retries=10, max_wait=90)

    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge_llm,
        embeddings=judge_embed,
        run_config=run_cfg,
    )
    df = result.to_pandas()
    df.to_csv("ragas_creators_report.csv", index=False)

    cols = ["faithfulness", "answer_relevancy", "context_precision"]
    print(df[["user_input", *cols]].to_string(index=False))
    m = df[cols].mean()
    print("\n=== MEDIAS (cole no formulario, item RAGAS) ===")
    print(
        f"faithfulness={m['faithfulness']:.2f}, "
        f"answer_relevancy={m['answer_relevancy']:.2f}, "
        f"context_precision={m['context_precision']:.2f}"
    )
    print("\nsalvo: ragas_creators_report.csv")


if __name__ == "__main__":
    main()
