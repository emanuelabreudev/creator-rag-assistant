"""Function-calling / tool-use — tools do agente.

TODO 4 (implementado): `analisar_legenda` — analisa uma legenda/caption de rede social
e devolve um relatorio determinístico: tamanho vs limite da plataforma, hashtags,
sinalizacao de publicidade e presenca de CTA. Util de verdade para criadores (e nao
decorativa): captura erros comuns como esquecer #publi numa parceria ou estourar o
limite de caracteres.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

# Limites de caracteres por plataforma (aproximados, 2026).
PLATAFORMA_LIMITES: dict[str, int] = {
    "instagram": 2200,
    "instagram_reels": 2200,
    "tiktok": 2200,
    "x": 280,
    "twitter": 280,
    "threads": 500,
    "linkedin": 3000,
    "youtube_titulo": 100,
    "youtube_descricao": 5000,
}

# Marcadores de sinalizacao de publicidade (parceria paga / recebido / afiliado).
_DISCLOSURE = [
    "#publi", "#publicidade", "#publipost", "#ad", "#ads", "#anuncio", "#anúncio",
    "#parceria", "#parceriapaga", "#paid", "#patrocinado",
    "parceria paga", "oferecido por", "patrocinado por", "recebi da", "recebi de",
    "em parceria com", "conteudo de marca", "conteúdo de marca", "cupom", "link de afiliado",
]

# Marcadores de chamada para acao (CTA).
_CTA = [
    "link na bio", "clica", "clique", "arrasta", "arraste", "salva", "salve",
    "compartilha", "compartilhe", "comenta", "comente", "segue", "siga",
    "se inscreve", "inscreva", "ative o sininho", "use o cupom", "use o codigo",
    "use o código", "saiba mais", "agende", "garanta",
]


def _conta(texto_lower: str, termos: list[str]) -> list[str]:
    return [t for t in termos if t in texto_lower]


# ------------------------------------------------------------------ TODO 4
def analisar_legenda(legenda: str, plataforma: str = "instagram") -> str:
    """Analisa uma legenda e retorna um relatorio textual com diagnostico e alertas."""
    plataforma = (plataforma or "instagram").strip().lower()
    limite = PLATAFORMA_LIMITES.get(plataforma, 2200)

    n_chars = len(legenda)
    hashtags = re.findall(r"#\w+", legenda)
    mencoes = re.findall(r"@\w+", legenda)
    low = legenda.lower()
    disclosure_found = _conta(low, _DISCLOSURE)
    cta_found = _conta(low, _CTA)

    alertas: list[str] = []
    if n_chars > limite:
        alertas.append(f"Estourou o limite de {plataforma} ({n_chars} > {limite} chars).")
    if len(hashtags) > 30:
        alertas.append(f"Hashtags em excesso ({len(hashtags)}); prefira poucas e relevantes.")
    if not disclosure_found:
        alertas.append(
            "Nenhuma sinalizacao de publicidade detectada — se for parceria/recebido/afiliado, "
            "adicione #publi/#publicidade no inicio."
        )
    if not cta_found:
        alertas.append("Sem chamada para acao (CTA) clara no texto.")

    relatorio = {
        "plataforma": plataforma,
        "caracteres": n_chars,
        "limite_plataforma": limite,
        "dentro_do_limite": n_chars <= limite,
        "qtd_hashtags": len(hashtags),
        "hashtags": hashtags[:30],
        "qtd_mencoes": len(mencoes),
        "sinaliza_publicidade": bool(disclosure_found),
        "marcadores_publicidade": disclosure_found,
        "tem_cta": bool(cta_found),
        "marcadores_cta": cta_found,
        "alertas": alertas or ["Nenhum alerta — legenda OK nos itens checados."],
    }
    return json.dumps(relatorio, ensure_ascii=False, indent=2)


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "analisar_legenda",
            "description": (
                "Analisa uma legenda/caption de rede social e retorna diagnostico: numero de "
                "caracteres vs limite da plataforma, quantidade de hashtags, se ha sinalizacao "
                "de publicidade (#publi etc.) e se ha chamada para acao (CTA). Use quando o "
                "usuario colar uma legenda e pedir analise/revisao/checagem dela."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "legenda": {
                        "type": "string",
                        "description": "O texto completo da legenda a analisar.",
                    },
                    "plataforma": {
                        "type": "string",
                        "description": "Plataforma alvo.",
                        "enum": list(PLATAFORMA_LIMITES.keys()),
                    },
                },
                "required": ["legenda"],
            },
        },
    }
]


TOOL_REGISTRY: dict[str, Callable[..., str]] = {
    "analisar_legenda": analisar_legenda,
}


def run_tool_call(name: str, arguments_json: str) -> str:
    """Executa uma tool call e retorna o resultado como string."""
    if name not in TOOL_REGISTRY:
        return f"ERROR: tool '{name}' nao registrada"
    try:
        kwargs = json.loads(arguments_json) if arguments_json else {}
        return TOOL_REGISTRY[name](**kwargs)
    except Exception as e:  # noqa: BLE001
        return f"ERROR ao executar {name}: {e}"
