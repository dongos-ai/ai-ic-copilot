"""Tests for copilot.retrieval: tokenizer, chunking, BM25, and memo-section retrieval."""
from __future__ import annotations

import math

import pytest

from copilot.config import CHUNK_WORDS
from copilot.retrieval import (
    KIND_FALLBACK,
    KINDS,
    MEMO_QUERIES,
    BM25Index,
    Chunk,
    Document,
    chunk_documents,
    retrieve_for_memo,
    tokenize,
)


def doc(text: str, kind: str = "mdna", section: str = "Item 7. MD&A", source: str = "10-K FY2025") -> Document:
    return Document(source=source, section=section, kind=kind, text=text)


def numbered_words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def chunk_word_count(chunk: Chunk) -> int:
    return len(chunk.text.split())


# --- tokenize ----------------------------------------------------------------------------------

def test_tokenize_lowercases_and_drops_stopwords_and_single_characters():
    assert tokenize("The Company's REVENUE grew in 2025, a record!") == ["company", "revenu", "grew", "2025", "record"]


def test_tokenize_handles_curly_apostrophes_and_contractions():
    assert tokenize("Apple\u2019s margin isn't what we're watching") == ["appl", "margin", "watch"]


@pytest.mark.parametrize("forms", [
    ["company", "companies", "company's"],
    ["business", "businesses"],
    ["increase", "increased", "increases", "increasing"],
    ["decline", "declined", "declining", "declines"],
    ["revenue", "revenues"],
    ["margin", "margins"],
    ["loss", "losses"],
    ["receivable", "receivables"],
])
def test_tokenize_maps_word_forms_to_one_stem(forms):
    assert len({tokenize(word)[0] for word in forms}) == 1


def test_tokenize_leaves_short_and_special_words_alone():
    assert tokenize("analysis status gas AI 10") == ["analysis", "status", "gas", "ai", "10"]


def test_tokenize_empty_and_stopword_only_text():
    assert tokenize("") == []
    assert tokenize("a I U.S. and of the it's") == []


# --- chunk_documents ---------------------------------------------------------------------------

def test_chunks_respect_max_words_and_keep_every_word_in_order():
    lines = [numbered_words(20, prefix=f"p{i}x") for i in range(30)]  # 30 paragraphs x 20 words
    chunks = chunk_documents([doc("\n".join(lines))], max_words=100)
    assert all(chunk_word_count(c) <= 100 for c in chunks)
    assert len(chunks) == 6  # 5 paragraphs fit in each chunk
    assert " ".join(c.text for c in chunks).split() == " ".join(lines).split()
    assert all(line in "\n".join(c.text for c in chunks).splitlines() for line in lines)  # paragraphs not cut


def test_long_paragraph_is_split_at_sentence_ends():
    sentences = [numbered_words(30, prefix=f"s{i}x") + "." for i in range(10)]  # 10 sentences x 30 words
    paragraph = " ".join(sentences)
    chunks = chunk_documents([doc(paragraph)], max_words=100)
    assert [chunk_word_count(c) for c in chunks] == [90, 90, 90, 30]
    assert all(c.text.endswith(".") for c in chunks)
    assert " ".join(c.text for c in chunks) == paragraph


def test_giant_sentence_is_cut_into_word_windows():
    chunks = chunk_documents([doc(numbered_words(250))], max_words=100)
    assert [chunk_word_count(c) for c in chunks] == [100, 100, 50]


def test_chunks_never_span_documents_and_carry_document_metadata():
    docs = [
        doc(numbered_words(150, prefix="risk"), kind="risk", section="Item 1A. Risk Factors"),
        doc(numbered_words(150, prefix="call"), kind="call_qa", section="Q&A", source="Earnings call: q3.txt"),
    ]
    chunks = chunk_documents(docs, max_words=100)
    assert [c.id for c in chunks] == ["C1", "C2", "C3", "C4"]
    assert [c.kind for c in chunks] == ["risk", "risk", "call_qa", "call_qa"]
    for chunk in chunks:
        prefix = "risk" if chunk.kind == "risk" else "call"
        assert all(word.startswith(prefix) for word in chunk.text.split())
    assert chunks[2].source == "Earnings call: q3.txt" and chunks[2].section == "Q&A"
    assert chunks[2].short_label == "Call Q&A"


def test_short_leftover_chunk_is_dropped_but_a_tiny_document_is_kept():
    docs = [doc(numbered_words(178) + "\nPage 12 of the annual report"),  # the 6-word footer overflows
            doc("Short note about pricing.", kind="call_full")]
    chunks = chunk_documents(docs)
    assert [chunk_word_count(c) for c in chunks] == [178, 4]
    assert chunks[1].kind == "call_full"


def test_short_chunk_threshold_scales_with_small_max_words():
    chunks = chunk_documents([doc("one two three\nfour five six\nseven eight nine")], max_words=4)
    assert [c.text for c in chunks] == ["one two three", "four five six", "seven eight nine"]


def test_empty_documents_produce_no_chunks():
    assert chunk_documents([doc(""), doc("  \n\n  ")]) == []
    assert chunk_documents([]) == []


def test_forward_looking_statement_disclaimers_are_not_indexed():
    guidance = "For the fourth quarter we expect revenue between $430 million and $445 million."
    disclaimer = ("This report contains forward-looking statements within the meaning of the Private Securities "
                  "Litigation Reform Act; words such as expect, outlook and guidance identify them.")
    chunks = chunk_documents([doc(f"{disclaimer}\n{guidance}", kind="call_prepared")])
    assert [c.text for c in chunks] == [guidance]


def test_exact_duplicate_chunks_are_kept_once():
    repeated = "The funds transfer pricing process allocates interest rate and liquidity costs to each segment."
    chunks = chunk_documents([doc(repeated), doc(repeated, kind="tenk_other")])
    assert [c.text for c in chunks] == [repeated]


def test_default_chunk_size_comes_from_config():
    chunks = chunk_documents([doc(numbered_words(CHUNK_WORDS + 1))])
    assert [chunk_word_count(c) for c in chunks] == [CHUNK_WORDS]  # the 1-word leftover is dropped


def test_max_words_must_be_positive():
    with pytest.raises(ValueError):
        chunk_documents([doc("text")], max_words=0)


# --- BM25Index ---------------------------------------------------------------------------------

def make_chunks(texts: list[str], kinds: list[str] | None = None) -> list[Chunk]:
    kinds = kinds or ["mdna"] * len(texts)
    return [Chunk(id=f"C{i + 1}", source="10-K FY2025", section="Item 7", kind=kind, text=text)
            for i, (text, kind) in enumerate(zip(texts, kinds))]


CORPUS = make_chunks(
    [
        "Revenue grew 20% on strong demand for data center products and new customers.",
        "Gross margin declined because of higher tariff costs and freight.",
        "The board approved a new share repurchase program and raised the quarterly dividend.",
        "Risk factors include supply chain disruption, competition, and cybersecurity incidents.",
        "Revenue from services increased while hardware revenue declined slightly.",
    ],
    kinds=["mdna", "mdna", "mdna", "risk", "call_qa"],
)


def test_search_ranks_the_most_relevant_chunk_first():
    index = BM25Index(CORPUS)
    assert index.search("share repurchase dividend")[0][0].id == "C3"
    assert index.search("tariff costs gross margin")[0][0].id == "C2"


def test_search_returns_only_positive_scores_best_first():
    results = BM25Index(CORPUS).search("revenue declined", k=10)
    assert {c.id for c, _ in results} == {"C1", "C2", "C5"}
    scores = [score for _, score in results]
    assert all(score > 0 for score in scores)
    assert scores == sorted(scores, reverse=True)
    assert results[0][0].id == "C5"  # the only chunk with both terms


def test_search_respects_k():
    index = BM25Index(CORPUS)
    assert len(index.search("revenue", k=1)) == 1
    assert index.search("revenue", k=0) == []
    assert index.search("revenue", k=-3) == []


def test_ties_keep_chunk_order():
    chunks = make_chunks(["pricing pressure", "unrelated words here", "pricing pressure"])
    results = BM25Index(chunks).search("pricing")
    assert [c.id for c, _ in results] == ["C1", "C3"]
    assert results[0][1] == results[1][1]


def test_idf_and_score_follow_the_bm25_formula():
    chunks = make_chunks(["alpha beta", "gamma delta", "epsilon zeta"])  # equal lengths -> no length penalty
    index = BM25Index(chunks)
    expected_idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    assert index.idf["alpha"] == pytest.approx(expected_idf)
    # tf = 1 and length == average length, so the BM25 term weight reduces to the idf
    assert index.search("alpha")[0][1] == pytest.approx(expected_idf)


def test_rare_terms_outweigh_common_terms():
    chunks = make_chunks(["revenue revenue goodwill", "revenue growth", "revenue margin", "revenue costs"])
    results = BM25Index(chunks).search("revenue goodwill")
    assert results[0][0].id == "C1"
    assert results[0][1] > 2 * results[1][1]


def test_kinds_filter():
    index = BM25Index(CORPUS)
    assert [c.id for c, _ in index.search("revenue", kinds=["call_qa"])] == ["C5"]
    assert [c.id for c, _ in index.search("revenue", kinds="call_qa")] == ["C5"]  # a bare string works too
    assert index.search("revenue", kinds=["risk"]) == []
    assert index.search("revenue", kinds=[]) == []


def test_empty_index_and_empty_queries_are_safe():
    assert BM25Index([]).search("revenue growth") == []
    index = BM25Index(CORPUS)
    assert index.search("") == []
    assert index.search("the and of it") == []
    assert index.search("zebra") == []


# --- MEMO_QUERIES and retrieve_for_memo --------------------------------------------------------

EXPECTED_SECTIONS = {
    "business": (["business"], 4),
    "growth": (["mdna", "call_prepared"], 4),
    "headwinds": (["mdna", "risk", "call_qa"], 4),
    "risks": (["risk"], 5),
    "financial_performance": (["mdna"], 4),
    "cash_flow": (["mdna"], 1),
    "management_outlook": (["call_prepared", "call_qa"], 5),
    "capital_allocation": (["mdna", "call_prepared"], 3),
    "accounting_quality": (["mdna", "risk"], 3),
}


def test_memo_queries_match_the_contract():
    assert {key: (spec["kinds"], spec["k"]) for key, spec in MEMO_QUERIES.items()} == EXPECTED_SECTIONS
    for spec in MEMO_QUERIES.values():
        assert spec["title"]
        assert set(spec["kinds"]) <= set(KINDS)
        assert len(set(tokenize(spec["query"]))) >= 10


def test_kind_fallbacks_point_to_real_kinds():
    assert set(KIND_FALLBACK) | set(KIND_FALLBACK.values()) <= set(KINDS)


def all_queries_text(variant: int) -> str:
    """Text that matches every memo query (variant changes the length, so scores differ)."""
    words = " ".join(spec["query"] for spec in MEMO_QUERIES.values())
    return words + " filler" * variant


def corpus_with_kinds(kinds: list[str], per_kind: int = 6) -> BM25Index:
    texts, chunk_kinds = [], []
    for kind in kinds:
        for i in range(per_kind):
            texts.append(all_queries_text(i))
            chunk_kinds.append(kind)
    return BM25Index(make_chunks(texts, chunk_kinds))


def count_kinds(hits: list[tuple[Chunk, float]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk, _ in hits:
        counts[chunk.kind] = counts.get(chunk.kind, 0) + 1
    return counts


def test_retrieve_for_memo_uses_section_kinds_quotas_and_order():
    index = corpus_with_kinds(["business", "risk", "mdna", "call_prepared", "call_qa"])
    results = retrieve_for_memo(index)
    assert list(results) == list(MEMO_QUERIES)
    for key, hits in results.items():
        kinds, k = EXPECTED_SECTIONS[key]
        assert len(hits) == k
        assert {c.kind for c, _ in hits} <= set(kinds)
        assert len({c.id for c, _ in hits}) == len(hits)  # no duplicates
        scores = [score for _, score in hits]
        assert scores == sorted(scores, reverse=True)
    assert count_kinds(results["growth"]) == {"mdna": 2, "call_prepared": 2}
    assert count_kinds(results["headwinds"]) == {"mdna": 2, "risk": 1, "call_qa": 1}
    assert count_kinds(results["management_outlook"]) == {"call_prepared": 3, "call_qa": 2}


def test_failed_sections_and_unsplit_call_use_fallback_kinds():
    index = corpus_with_kinds(["tenk_other", "call_full"])
    results = retrieve_for_memo(index)
    assert count_kinds(results["business"]) == {"tenk_other": 4}
    assert count_kinds(results["headwinds"]) == {"tenk_other": 2, "call_full": 2}
    assert count_kinds(results["management_outlook"]) == {"call_full": 5}


def test_without_a_transcript_the_10k_gets_the_whole_quota():
    results = retrieve_for_memo(corpus_with_kinds(["business", "risk", "mdna"]))
    assert count_kinds(results["growth"]) == {"mdna": 4}
    assert count_kinds(results["headwinds"]) == {"mdna": 2, "risk": 2}
    # no call at all: the guidance section searches every document instead of coming back empty
    assert len(results["management_outlook"]) == 5


def test_section_with_no_available_kind_searches_everything():
    results = retrieve_for_memo(corpus_with_kinds(["call_full"], per_kind=2))
    assert count_kinds(results["business"]) == {"call_full": 2}
    assert count_kinds(results["risks"]) == {"call_full": 2}


def test_k_override_applies_to_every_section():
    results = retrieve_for_memo(corpus_with_kinds(["business", "risk", "mdna", "call_prepared", "call_qa"]),
                                k_override=2)
    assert all(len(hits) == 2 for hits in results.values())
    assert count_kinds(results["growth"]) == {"mdna": 1, "call_prepared": 1}


def test_cash_flow_section_finds_the_operating_cash_flow_explanation():
    # Shaped like a real MD&A: the OCF paragraph competes with results, segment and buyback paragraphs that
    # also say "increased", "primarily driven" and "cash". No other section's query used to pick it up.
    texts = [
        "Net operating revenues increased 2% compared to the prior year, primarily driven by favorable pricing "
        "and higher concentrate sales volume. Gross margin and operating income increased.",
        "The EMEA segment reported operating income that increased, primarily driven by higher volume, partially "
        "offset by higher commodity costs and increased marketing spending.",
        "During the year the Company repurchased shares under its share repurchase program and paid dividends; "
        "cash returned to shareholders increased.",
        "Net cash provided by operating activities was $7,408 million and $6,805 million, respectively, an "
        "increase of 9%, primarily driven by lower tax payments and the timing of changes in working capital, "
        "partially offset by higher marketing payments. The 2025 activity included a $6.1 billion milestone "
        "payment; 2024 included a $6.0 billion tax deposit.",
    ]
    results = retrieve_for_memo(BM25Index(make_chunks(texts, ["mdna"] * len(texts))))
    assert [chunk.text for chunk, _ in results["cash_flow"]] == [texts[3]]


def test_empty_index_and_unmatched_text_give_empty_sections():
    assert retrieve_for_memo(BM25Index([])) == {key: [] for key in MEMO_QUERIES}
    unmatched = BM25Index(make_chunks(["zebra giraffe okapi"], ["business"]))
    assert retrieve_for_memo(unmatched) == {key: [] for key in MEMO_QUERIES}


# --- real 10-K text (optional fixtures) --------------------------------------------------------

def test_real_10k_full_text_fallback(tenk_text):
    text = tenk_text("AAPL")  # the whole 10-K, as the full-text fallback indexes it
    chunks = chunk_documents([doc(text, kind="tenk_other", section="10-K text (outside parsed sections)")])
    assert len(chunks) > 100
    assert all(chunk_word_count(c) <= CHUNK_WORDS for c in chunks)
    assert not any("forward-looking statements" in c.text.lower() for c in chunks)

    results = retrieve_for_memo(BM25Index(chunks))
    for key, hits in results.items():
        assert len(hits) == MEMO_QUERIES[key]["k"], key
        assert all(c.kind == "tenk_other" for c, _ in hits)
    top_capital = " ".join(c.text for c, _ in results["capital_allocation"]).lower()
    assert "repurchase" in top_capital or "dividend" in top_capital
