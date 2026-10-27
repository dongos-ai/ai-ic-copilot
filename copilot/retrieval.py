"""Chunk documents and retrieve evidence for each memo section with BM25 (no vector database)."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from .config import CHUNK_WORDS

KINDS = ("business", "risk", "mdna", "tenk_other", "call_prepared", "call_qa", "call_full")
KIND_FALLBACK = {
    "business": "tenk_other", "risk": "tenk_other", "mdna": "tenk_other",
    "call_prepared": "call_full", "call_qa": "call_full",
}
SHORT_LABELS = {
    "business": "10-K Item 1", "risk": "10-K Item 1A", "mdna": "10-K Item 7", "tenk_other": "10-K text",
    "call_prepared": "Call remarks", "call_qa": "Call Q&A", "call_full": "Earnings call",
}
MIN_CHUNK_WORDS = 25  # shorter chunks (stray headings, page footers) are dropped unless the document is tiny
# Safe-harbor disclaimers list every outlook word ("expect", "guidance", "outlook", ...) and would otherwise
# win the guidance query, so paragraphs mentioning forward-looking statements are not indexed.
_BOILERPLATE = re.compile(r"forward[-\s]looking statements?", re.IGNORECASE)


@dataclass
class Document:
    source: str   # e.g. "10-K FY2025 (filed 2025-10-31)"
    section: str  # e.g. "Item 1A. Risk Factors"
    kind: str     # one of KINDS
    text: str


@dataclass
class Chunk:
    id: str       # "C1", "C2", ...
    source: str
    section: str
    kind: str
    text: str

    @property
    def short_label(self) -> str:
        return SHORT_LABELS.get(self.kind, self.section)


# One BM25 query per memo section. "kinds" says which documents to search; "k" is how many passages to keep.
# Queries are keyword bags: order and repeats don't matter (tokenize() also merges word forms), so just list
# the vocabulary a good passage for that section would use.
MEMO_QUERIES: dict[str, dict] = {
    "business": {
        "title": "Business overview",
        "kinds": ["business"],
        "k": 4,
        "query": "company designs develops manufactures sells products services customers markets segments "
                 "platform subscription revenue sources distribution channels competition competitive strategy "
                 "brand technology",
    },
    "growth": {
        "title": "Growth drivers (bull case)",
        "kinds": ["mdna", "call_prepared"],
        "k": 4,
        "query": "growth grew increased higher demand strong record expansion new products launch adoption "
                 "customers market share backlog bookings momentum opportunity accelerate revenue sales",
    },
    "headwinds": {
        "title": "Headwinds (bear case)",
        "kinds": ["mdna", "risk", "call_qa"],
        "k": 4,
        "query": "decline decreased lower weakness weaker pressure headwinds slowdown softness competition pricing "
                 "tariffs trade inflation costs uncertain uncertainty adverse impact negative reduced challenges "
                 "delays macroeconomic margin compression",
    },
    "risks": {
        "title": "Top risks",
        "kinds": ["risk"],
        "k": 5,
        "query": "material adverse effect risk depend dependence customers suppliers supply chain competition "
                 "regulation litigation cybersecurity economic conditions demand loss key",
    },
    "financial_performance": {
        "title": "Financial performance",
        "kinds": ["mdna"],
        "k": 4,
        "query": "net sales revenue increased decreased compared prior year gross margin operating income "
                 "operating expenses net income results fiscal percentage primarily driven",
    },
    # The MD&A paragraph that explains operating cash flow (one-off payments, working capital). One passage:
    # the runner-ups on real 10-Ks were unrelated "primarily driven" paragraphs.
    "cash_flow": {
        "title": "Cash flow",
        "kinds": ["mdna"],
        "k": 1,
        "query": "cash flows net cash provided used operating activities increased decreased primarily due driven "
                 "payments working capital partially offset",
    },
    "management_outlook": {
        "title": "Management outlook and guidance",
        "kinds": ["call_prepared", "call_qa"],
        "k": 5,
        "query": "expect outlook guidance next quarter full year fiscal anticipate range approximately midpoint "
                 "raise reiterate revenue margin trends going forward",
    },
    "capital_allocation": {
        "title": "Capital allocation",
        "kinds": ["mdna", "call_prepared"],
        "k": 3,
        "query": "capital allocation share repurchase buyback dividend returned shareholders acquisitions "
                 "capital expenditures debt repayment liquidity cash flow authorization program",
    },
    "accounting_quality": {
        "title": "Accounting quality",
        "kinds": ["mdna", "risk"],
        "k": 3,
        "query": "critical accounting estimates judgment receivables inventory reserves allowance "
                 "impairment goodwill stock-based compensation revenue recognition internal control material "
                 "weakness restatement write-down",
    },
}

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been before being below between both
but by can could did do does doing down during each few for from further had has have having he her here hers him
his how i if in into is it its itself just let me more most my no nor not of off on once only or other our ours out
over own same she should so some such than that the their theirs them then there these they this those through to
too under until up us very was we were what when where which while who whom why will with would you your yours
don't doesn't didn't isn't aren't wasn't weren't hasn't haven't hadn't can't couldn't won't wouldn't shouldn't
i'm i've i'll i'd we're we've we'll we'd you're you've you'll they're they've they'll
""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with stopwords removed and light, deterministic suffix stripping."""
    tokens = []
    for token in _TOKEN.findall(text.lower().replace("\u2019", "'")):
        if token.endswith("'s"):
            token = token[:-2]
        if len(token) < 2 or token in _STOPWORDS:
            continue
        tokens.append(_stem(token))
    return tokens


def _stem(word: str) -> str:
    """Tiny suffix stripper so "companies/company" and "increased/increasing/increase" match."""
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("s") and len(word) > 4 and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    if word.endswith("ing") and len(word) > 5:
        word = word[:-3]
    elif word.endswith("ed") and len(word) > 5:
        word = word[:-2]
    if word.endswith("e") and len(word) > 3:
        word = word[:-1]  # "increase" -> "increas", matching "increased" -> "increas"
    return word


def chunk_documents(docs: list[Document], max_words: int = CHUNK_WORDS) -> list[Chunk]:
    """Split documents into chunks of at most max_words words. Chunks never span documents.

    Exact duplicate chunks (10-Ks repeat some paragraphs word for word) are kept only once.
    """
    if max_words < 1:
        raise ValueError("max_words must be at least 1")
    min_words = min(MIN_CHUNK_WORDS, max_words // 4)  # stay sensible when max_words is small
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for doc in docs:
        pieces = _split_text(doc.text, max_words)
        if len(pieces) > 1:  # a tiny document keeps its single short chunk
            pieces = [p for p in pieces if len(p.split()) >= min_words]
        for piece in pieces:
            if piece in seen:
                continue
            seen.add(piece)
            chunks.append(Chunk(id=f"C{len(chunks) + 1}", source=doc.source, section=doc.section,
                                kind=doc.kind, text=piece))
    return chunks


def _split_text(text: str, max_words: int) -> list[str]:
    """Pack consecutive lines (paragraphs) into chunks; over-long paragraphs are split first."""
    pieces = []
    for line in text.splitlines():
        line = line.strip()
        if _BOILERPLATE.search(line):
            continue
        if len(line.split()) > max_words:
            pieces.extend(_split_paragraph(line, max_words))
        elif line:
            pieces.append(line)
    return _pack(pieces, max_words, "\n")


def _split_paragraph(paragraph: str, max_words: int) -> list[str]:
    """Split one long paragraph at sentence ends; a giant sentence is cut into fixed word windows."""
    pieces = []
    for sentence in re.split(r"(?<=[.!?;])\s+", paragraph):
        words = sentence.split()
        pieces.extend(" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words))
    return _pack(pieces, max_words, " ")


def _pack(pieces: list[str], max_words: int, sep: str) -> list[str]:
    """Greedily join consecutive pieces (each <= max_words words) into groups of <= max_words words."""
    groups: list[str] = []
    current: list[str] = []
    count = 0
    for piece in pieces:
        n = len(piece.split())
        if current and count + n > max_words:
            groups.append(sep.join(current))
            current, count = [], 0
        current.append(piece)
        count += n
    if current:
        groups.append(sep.join(current))
    return groups


class BM25Index:
    """Okapi BM25 over chunks (a small in-memory inverted index)."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        term_counts = [Counter(tokenize(chunk.text)) for chunk in self.chunks]
        self.lengths = [sum(counts.values()) for counts in term_counts]
        self.avg_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        # term -> [(chunk position, term frequency), ...]
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for position, counts in enumerate(term_counts):
            for term, tf in counts.items():
                self.postings[term].append((position, tf))
        n = len(self.chunks)
        self.idf = {term: math.log(1 + (n - len(hits) + 0.5) / (len(hits) + 0.5))
                    for term, hits in self.postings.items()}

    def search(self, query: str, k: int = 5, kinds: Iterable[str] | None = None) -> list[tuple[Chunk, float]]:
        """Top-k (chunk, score) pairs with score > 0, best first; ties keep chunk order."""
        if k <= 0:
            return []
        if isinstance(kinds, str):
            kinds = [kinds]
        allowed = set(kinds) if kinds is not None else None
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            for position, tf in self.postings.get(term, []):
                if allowed is not None and self.chunks[position].kind not in allowed:
                    continue
                length_norm = 1 - self.b + self.b * self.lengths[position] / self.avg_length
                scores[position] += self.idf[term] * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [(self.chunks[position], score) for position, score in ranked[:k] if score > 0]


def retrieve_for_memo(index: BM25Index, k_override: int | None = None) -> dict[str, list[tuple[Chunk, float]]]:
    """Top passages for every memo section in MEMO_QUERIES, best first.

    k is split round-robin across the section's kinds. A kind with no chunks is replaced by its
    KIND_FALLBACK (e.g. a failed Item 1A falls back to the rest of the 10-K text) or skipped.
    A section that still finds nothing searches every document.
    """
    present = {chunk.kind for chunk in index.chunks}
    results: dict[str, list[tuple[Chunk, float]]] = {}
    for key, spec in MEMO_QUERIES.items():
        k = spec["k"] if k_override is None else k_override
        kinds = _available_kinds(spec["kinds"], present)
        hits: list[tuple[Chunk, float]] = []
        for i, kind in enumerate(kinds):
            quota = k // len(kinds) + (1 if i < k % len(kinds) else 0)  # round-robin split of k
            # Each chunk has exactly one kind and `kinds` has no repeats, so these results never overlap.
            hits += index.search(spec["query"], k=quota, kinds=[kind])
        if not hits:
            hits = index.search(spec["query"], k=k)
        results[key] = sorted(hits, key=lambda hit: -hit[1])  # stable sort keeps kind order on ties
    return results


def _available_kinds(wanted: list[str], present: set[str]) -> list[str]:
    """Map each wanted kind to itself or its fallback if it has chunks; drop the rest (order kept, no repeats)."""
    kinds: list[str] = []
    for kind in wanted:
        if kind not in present:
            kind = KIND_FALLBACK.get(kind, "")
        if kind in present and kind not in kinds:
            kinds.append(kind)
    return kinds
