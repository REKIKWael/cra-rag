"""Tests for splitting the texts into sections.

Two kinds of tests:
- a tiny hand-written document, to check each rendering rule in isolation;
- the real corpus (committed in corpus/raw), to check the result on the actual texts.
"""

from pathlib import Path

import pytest
from defusedxml import EntitiesForbidden

from cra_rag import parse

RAW_DIR = Path(__file__).parent.parent / "corpus" / "raw"

SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><body>
<div class="eli-subdivision" id="rct_1">
  <table><tbody><tr>
    <td><p class="oj-normal">(1)</p></td>
    <td><p class="oj-normal">Cybersecurity is a key challenge.</p></td>
  </tr></tbody></table>
</div>
<div class="eli-subdivision" id="enc_1">
  <div id="cpt_II">
    <p class="oj-ti-section-1">CHAPTER II</p>
    <div class="eli-title"><p class="oj-ti-section-2"><span>OBLIGATIONS OF THE EU</span></p></div>
    <div class="eli-subdivision" id="art_14">
      <p class="oj-ti-art">Article&#160;14</p>
      <div class="eli-title"><p class="oj-sti-art">Reporting obligations</p></div>
      <div><p class="oj-normal">1.   The manufacturer shall submit:</p>
        <table><tbody><tr>
          <td><p class="oj-normal">(a)</p></td>
          <td><p class="oj-normal">an early warning within 24 hours, per Regulation
            <a href="#ntr1-X">(<span class="oj-super oj-note-tag">1</span>)</a>, and</p>
            <table><tbody><tr>
              <td><p class="oj-normal">(i)</p></td>
              <td><p class="oj-normal">a sub-point;</p></td>
            </tr></tbody></table>
          </td>
        </tr></tbody></table>
      </div>
    </div>
  </div>
</div>
<div class="eli-container" id="anx_IV">
  <p class="oj-doc-ti">ANNEX IV</p>
  <p class="oj-doc-ti">CRITICAL PRODUCTS</p>
  <div><p style="display: inline;">1.   </p>
    <p style="display: inline;"><span>Smartcards</span></p></div>
  <hr class="oj-note"/>
  <p class="oj-note">(1) A footnote that must not appear.</p>
</div>
</body></html>
"""


@pytest.fixture
def sample(tmp_path):
    path = tmp_path / "sample.html"
    path.write_text(SAMPLE, encoding="utf-8")
    sections = parse.parse_file(path, source="cra", lang="en", url="https://example.eu/doc")
    return {s.id: s for s in sections}


def test_finds_articles_annexes_and_recitals(sample):
    assert sorted(sample) == ["cra:anx_IV", "cra:art_14", "cra:rct_1"]


def test_article_has_label_title_chapter_and_link(sample):
    article = sample["cra:art_14"]
    assert article.label == "Article 14"
    assert article.title == "Reporting obligations"
    assert article.chapter == "Chapter II: Obligations of the EU"
    assert article.url == "https://example.eu/doc#art_14"


def test_points_are_written_with_their_letter_and_sub_points_indented(sample):
    assert sample["cra:art_14"].text.splitlines() == [
        "1. The manufacturer shall submit:",
        "(a) an early warning within 24 hours, per Regulation, and",
        "  (i) a sub-point;",
    ]


def test_footnotes_are_left_out(sample):
    assert "(1)" not in sample["cra:art_14"].text
    assert "footnote" not in sample["cra:anx_IV"].text


def test_annex_label_title_and_inline_numbering(sample):
    annex = sample["cra:anx_IV"]
    assert (annex.label, annex.title) == ("Annex IV", "Critical products")
    assert annex.text == "1. Smartcards"


def test_recital_text_starts_after_its_number(sample):
    recital = sample["cra:rct_1"]
    assert recital.label == "Recital 1"
    assert recital.text == "Cybersecurity is a key challenge."


# --- The real corpus --------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus():
    return parse.parse_corpus(RAW_DIR)


def count(sections, source, lang, kind):
    return sum(1 for s in sections if (s.source, s.lang, s.kind) == (source, lang, kind))


@pytest.mark.parametrize("lang", ["en", "fr"])
def test_cyber_resilience_act_is_complete(corpus, lang):
    assert count(corpus, "cra", lang, "article") == 71
    assert count(corpus, "cra", lang, "annex") == 8
    assert count(corpus, "cra", lang, "recital") == 130


@pytest.mark.parametrize("lang", ["en", "fr"])
def test_red_delegated_act_is_complete(corpus, lang):
    assert count(corpus, "red-da", lang, "article") == 3
    assert count(corpus, "red-da", lang, "recital") == 19


def test_corpus_has_no_problems(corpus):
    assert parse.check(corpus) == []


def test_known_passages_are_where_they_should_be(corpus):
    by_key = {(s.id, s.lang): s for s in corpus}
    assert "within 24 hours" in by_key[("cra:art_14", "en")].text
    assert "au plus tard 24 heures" in by_key[("cra:art_14", "fr")].text
    assert by_key[("cra:art_1", "fr")].label == "Article premier"
    assert by_key[("cra:anx_I", "en")].title == "Essential cybersecurity requirements"
    assert "without known exploitable vulnerabilities" in by_key[("cra:anx_I", "en")].text


def test_an_xml_bomb_is_refused(tmp_path):
    # "Billion laughs": nested entities that expand to gigabytes of text in memory.
    bomb = tmp_path / "bomb.html"
    bomb.write_text(
        '<?xml version="1.0"?>\n<!DOCTYPE lolz [<!ENTITY lol "lol">'
        '<!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">]>\n'
        "<html><body><p>&lol2;</p></body></html>",
        encoding="utf-8",
    )
    with pytest.raises(EntitiesForbidden):
        parse.parse_file(bomb, source="cra", lang="en", url="")
