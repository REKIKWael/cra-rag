"""Tests for the download step. No network: the Cellar server is simulated."""

import hashlib
import json

import httpx
import pytest

from cra_rag import download

LEGAL_PAGE = '<html><body><p class="oj-ti-art">Article 1</p></body></html>'


def run(tmp_path, handler):
    """Run the download against a fake server whose answers come from `handler`."""
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        return download.download_all(tmp_path, client, pause=0, backoff=0)


def legal_text(request):
    return httpx.Response(200, text=LEGAL_PAGE)


def test_writes_one_file_per_source_and_language(tmp_path):
    manifest = run(tmp_path, legal_text)

    assert sorted(manifest) == ["cra_en.html", "cra_fr.html", "red-da_en.html", "red-da_fr.html"]
    for filename, entry in manifest.items():
        content = (tmp_path / filename).read_bytes()
        assert entry["sha256"] == hashlib.sha256(content).hexdigest()


def test_asks_cellar_for_the_right_document_and_language(tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        return legal_text(request)

    run(tmp_path, handler)

    first = requests[0]
    assert str(first.url) == "https://publications.europa.eu/resource/celex/32024R2847"
    assert first.headers["Accept-Language"] == "eng"
    assert {r.headers["Accept-Language"] for r in requests} == {"eng", "fra"}


def test_manifest_is_saved_next_to_the_files(tmp_path):
    manifest = run(tmp_path, legal_text)

    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved == manifest
    assert saved["cra_fr.html"]["url"].startswith("https://eur-lex.europa.eu/legal-content/FR/")


def test_falls_back_to_html_when_xhtml_is_not_available(tmp_path):
    def handler(request):
        if request.headers["Accept"] == "application/xhtml+xml":
            return httpx.Response(406)
        return legal_text(request)

    assert len(run(tmp_path, handler)) == 4


def test_follows_the_list_cellar_returns_when_there_are_several_files(tmp_path):
    item = "https://publications.europa.eu/resource/cellar/abc.0006.03/DOC_1"

    def handler(request):
        if str(request.url) == item:
            return legal_text(request)
        return httpx.Response(300, text=f'<a href="{item}">DOC_1</a>')

    assert len(run(tmp_path, handler)) == 4


def test_rejects_a_page_that_is_not_the_legal_text(tmp_path):
    def handler(request):
        return httpx.Response(202, text="<html>Please verify you are human</html>")

    with pytest.raises(download.DownloadError, match="HTTP 202"):
        run(tmp_path, handler)
    assert not (tmp_path / "cra_en.html").exists()


def test_retries_after_a_server_error(tmp_path):
    calls = {"count": 0}

    def handler(request):
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(503)
        return legal_text(request)

    assert len(run(tmp_path, handler)) == 4
    assert calls["count"] == 5  # one failure, then four successes


def test_unchanged_text_keeps_its_retrieval_date(tmp_path):
    first = run(tmp_path, legal_text)
    second = run(tmp_path, legal_text)

    assert second["cra_en.html"]["retrieved_at"] == first["cra_en.html"]["retrieved_at"]
