# Corpus

Esta pasta contem o corpus indexado pelo pipeline.

## Arquivo atual

- `guia-criadores-conteudo.md` — guia autoral (conteudo original, sem direitos autorais
  de terceiros; seguro para deploy publico) sobre publicidade, alegacoes de saude,
  direitos autorais, LGPD e boas praticas por plataforma para criadores de conteudo.

O `ingest_and_index` le `.md`, `.txt` e `.pdf`. Para trocar o corpus, basta colocar
seus arquivos aqui e apagar a pasta `data/chroma/` (o indice e reconstruido no proximo run).

## Restricoes

- Pelo menos 1 documento legivel.
- PDFs precisam ter texto extraivel (escaneados sem OCR nao funcionam — use `ocrmypdf` antes).
- Use documentos sem direitos autorais ou com licenca compativel com uso publico.
