"""Download the regulatory texts into corpus/raw/.

The texts come from Cellar, the Publications Office repository behind EUR-Lex.
Cellar is the channel meant for programs: the EUR-Lex website itself answers
scripts with an anti-bot challenge (HTTP 202) instead of the document.

The raw HTML is committed to the repository. EU legal texts may be reused
(Commission Decision 2011/833/EU), and a pinned snapshot keeps the pipeline
reproducible and the CI independent of the network.
"""

import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

CELLAR_URL = "https://publications.europa.eu/resource/celex/{celex}"
# Human-readable page, used as the citation link.
EURLEX_URL = "https://eur-lex.europa.eu/legal-content/{lang}/TXT/?uri=CELEX:{celex}"
USER_AGENT = "cra-rag/0.1 (+https://github.com/REKIKWael/cra-rag)"

# Our language codes -> the three-letter codes Cellar expects.
LANGUAGES = {"EN": "eng", "FR": "fra"}
# Formats asked from Cellar, in order of preference.
MEDIA_TYPES = ("application/xhtml+xml", "text/html")

# Every article heading in an Official Journal HTML document carries a CSS class
# containing "ti-art". A page without it is not the legal text.
ARTICLE_MARKER = "ti-art"
CELLAR_LINK = re.compile(r'href="(https?://publications\.europa\.eu/resource/cellar/[^"]+)"')
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class Source:
    key: str  # short name used in file names, e.g. "cra"
    celex: str  # EUR-Lex document identifier
    title: str


SOURCES = (
    Source("cra", "32024R2847", "Regulation (EU) 2024/2847 (Cyber Resilience Act)"),
    Source(
        "red-da",
        "32022R0030",
        "Commission Delegated Regulation (EU) 2022/30 (RED delegated act)",
    ),
)


class DownloadError(RuntimeError):
    """The legal text could not be retrieved."""


def _get(
    client: httpx.Client, url: str, headers: dict, *, retries: int = 3, backoff: float = 2.0
) -> httpx.Response:
    """GET with retries on network errors and temporary server errors."""
    for attempt in range(1, retries + 1):
        try:
            response = client.get(url, headers=headers)
        except httpx.TransportError as exc:
            if attempt == retries:
                raise DownloadError(f"Network error on {url}: {exc}") from exc
        else:
            if response.status_code not in RETRYABLE_STATUSES or attempt == retries:
                return response
        time.sleep(backoff * attempt)
    raise AssertionError("unreachable")


def _is_legal_text(response: httpx.Response) -> bool:
    return response.status_code == 200 and ARTICLE_MARKER in response.text


def _describe(response: httpx.Response) -> str:
    """One line explaining what we received instead of the legal text."""
    start = " ".join(response.text[:200].split())[:120]
    content_type = response.headers.get("content-type", "no content type")
    return f"HTTP {response.status_code}, {content_type}, {response.url}, starts with {start!r}"


def fetch_document(
    client: httpx.Client, source: Source, lang: str, *, backoff: float = 2.0
) -> tuple[bytes, str]:
    """Return (HTML bytes, URL it was served from) for one source in one language."""
    url = CELLAR_URL.format(celex=source.celex)
    problems = []
    for media_type in MEDIA_TYPES:
        headers = {"Accept": media_type, "Accept-Language": LANGUAGES[lang]}
        response = _get(client, url, headers, backoff=backoff)

        if response.status_code == 300:
            # Several files for this format: Cellar lists them, we take the legal text.
            links = list(dict.fromkeys(CELLAR_LINK.findall(response.text)))
            for link in links:
                item = _get(client, link, {"Accept": media_type}, backoff=backoff)
                if _is_legal_text(item):
                    return item.content, str(item.url)
            problems.append(f"{media_type}: {len(links)} files listed, none is the legal text")
            continue

        if _is_legal_text(response):
            return response.content, str(response.url)
        problems.append(f"{media_type}: {_describe(response)}")

    raise DownloadError(
        f"Could not download {source.celex} ({lang}) from Cellar. " + " | ".join(problems)
    )


def _load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def download_all(
    out_dir: Path,
    client: httpx.Client | None = None,
    *,
    sources: tuple[Source, ...] = SOURCES,
    languages: tuple[str, ...] = tuple(LANGUAGES),
    pause: float = 1.0,
    backoff: float = 2.0,
    log: Callable[[str], None] = lambda message: None,
) -> dict:
    """Download every source in every language, then write manifest.json.

    The manifest records where each file comes from and its SHA-256 checksum.
    If a text has not changed since the last run, its entry (and retrieval date)
    is kept as is, so re-running the command does not create noise in git.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    previous = _load_manifest(manifest_path)
    manifest: dict = {}

    own_client = client is None
    if client is None:
        client = httpx.Client(
            headers={"User-Agent": USER_AGENT}, timeout=60.0, follow_redirects=True
        )
    try:
        jobs = [(source, lang) for source in sources for lang in languages]
        for index, (source, lang) in enumerate(jobs):
            if index:
                time.sleep(pause)  # be polite with the server
            filename = f"{source.key}_{lang.lower()}.html"
            log(f"Downloading {filename} ...")
            content, served_from = fetch_document(client, source, lang, backoff=backoff)
            (out_dir / filename).write_bytes(content)

            sha256 = hashlib.sha256(content).hexdigest()
            if previous.get(filename, {}).get("sha256") == sha256:
                manifest[filename] = previous[filename]
            else:
                manifest[filename] = {
                    "title": source.title,
                    "celex": source.celex,
                    "lang": lang,
                    "url": EURLEX_URL.format(lang=lang, celex=source.celex),
                    "downloaded_from": served_from,
                    "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "sha256": sha256,
                    "bytes": len(content),
                }
    finally:
        if own_client:
            client.close()

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest
