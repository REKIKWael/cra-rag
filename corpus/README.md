# Corpus

`raw/` holds the regulatory texts exactly as the EU Publications Office served
them, one HTML file per regulation and language, plus `manifest.json` (EUR-Lex
link, download URL, retrieval date, SHA-256 checksum).

Refresh them with:

```bash
uv run cra-rag download
```

The files come from Cellar (`publications.europa.eu`), the repository behind
EUR-Lex that is meant for programs. The EUR-Lex website itself answers scripts
with an anti-bot challenge.

Source: EUR-Lex, © European Union, https://eur-lex.europa.eu.
Reuse is authorised under Commission Decision 2011/833/EU. Only the texts
published in the Official Journal of the European Union are authentic.
