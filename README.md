# AI Investment Committee Copilot

Type a ticker and get the memo an investment committee would read before voting. The app builds it from the company's latest 10-K, five years of SEC XBRL financials, live market data and (optionally) an earnings-call transcript. A Bull analyst, a Bear analyst and a Risk officer argue from retrieved evidence; a Chair writes the memo and casts a **Buy / Hold / Pass** vote.

**No citation, no claim.** Every claim in the memo cites an evidence ID such as `[E3 · 10-K Item 1A]` or `[F1 · XBRL Revenue]`. The app deletes a claim when it has no valid citation or when it states a number that its cited evidence doesn't contain. It also removes any quote it cannot find word for word in the cited passage.

Without an API key the app runs in **data-only mode**. It still shows the financial table, the red flags and the retrieved passages, so the demo never breaks.

> Educational project. The memo is a draft for discussion, not investment advice.

---

## How it works

```
ticker ──► SEC ticker map ──► CIK
             │
             ├──► latest 10-K (EDGAR) ──► HTML → text ──► Item 1 / 1A / 7   (fallback: full text)
             ├──► XBRL company facts ───► 5-year table: revenue (multi-tag), margins, net income,
             │                             FCF = OCF − capex, cash & short-term investments, debt, SBC
             ├──► yfinance ─────────────► price, market cap ──► P/E, EV/Sales, FCF yield
             └──► transcript upload ────► prepared remarks / Q&A   (.txt or .pdf, never scraped)
                          │
        ~180-word chunks ─► BM25 search per memo section ─► evidence pack
                                                            [F#] financial facts
                                                            [R#] rule-based red flags
                                                            [E#] document passages
                          │
          Bull analyst ─┐
          Bear analyst ─┼──► Chair ──► citation audit ──► Markdown memo (+ download)
          Risk officer ─┘             (uncited or unsupported claims dropped, non-verbatim quotes removed)
```

| Step | Module | What happens |
|---|---|---|
| 1. 10-K | `copilot/edgar.py` | Maps the ticker to its CIK using SEC's `company_tickers.json`. Then it finds the latest 10-K in the submissions API and converts the HTML to clean text, dropping page numbers and running footers. It extracts Items 1, 1A and 7: the longest span between an item heading and the next item heading wins, which skips the table of contents. If a section can't be found, the rest of the filing is indexed as "10-K text" instead. When a 10-K only points to a separate annual-report exhibit (Wells Fargo does this), the app reads that exhibit too. |
| 2. Financials | `copilot/financials.py` | Reads XBRL company facts and keeps annual 10-K facts with roughly 12-month durations. Restated values win. Revenue is the largest value across about ten revenue tags for each year, because companies switch tags over time and REITs and banks use different ones. Fiscal-year labels follow the company's own convention, so Home Depot's fiscal 2025 ends in February 2026. Stock splits and 53-week years are handled. |
| 2b. Valuation | `copilot/market.py` | Gets price and market cap from yfinance. Computes P/E, EV/Sales and FCF yield on the latest fiscal year, labelled as such, not TTM. EV = market cap + debt − cash − short-term investments. Banks and insurers (SIC 6000–6399) get P/E only, because deposits aren't ordinary debt. |
| 3. Transcript | `copilot/transcript.py` | Reads the `.txt` (UTF-8/UTF-16/cp1252) or `.pdf` (pypdf) you upload and splits prepared remarks from Q&A. |
| 4. Retrieval | `copilot/retrieval.py` | Packs paragraphs into ~180-word chunks. A small, self-written BM25 index runs one keyword query per memo section (business, growth, headwinds, risks, MD&A, outlook, capital allocation, accounting quality). No vector database. |
| Red flags | `copilot/red_flags.py` | Runs deterministic checks on the numbers: receivables or inventory growing faster than revenue, gross or operating margin compression, heavy stock comp, earnings ahead of cash flow, negative FCF and cash runway, revenue decline, losses, leverage and dilution. It also scans the 10-K text for going-concern doubt, a material weakness, a restatement and customer concentration. Cash-flow checks are skipped for banks and insurers, where they misfire. |
| 5. Committee | `copilot/committee.py`, `copilot/llm.py` | Bull, Bear and Risk officer run in parallel on the same numbered evidence pack. The Chair checks their arguments against the evidence and writes the memo as JSON. `enforce_citations` then audits every claim. Works with Anthropic, OpenAI or Gemini. |
| Memo | `copilot/memo.py` | Renders Markdown. The financial table is always computed from SEC data, never written by the LLM. |
| App | `app.py`, `copilot/pipeline.py` | `pipeline.run_analysis` runs every step with its own error handling. `app.py` is a thin Streamlit UI. |

## File tree

```
ai-ic-copilot/
├── app.py                     # Streamlit UI (thin: calls copilot.pipeline)
├── requirements.txt
├── README.md
├── .env.example               # copy to .env: SEC_USER_AGENT + one optional LLM key
├── .gitignore
├── .streamlit/config.toml     # upload limit (20 MB) to match the app's transcript limit
├── copilot/
│   ├── __init__.py
│   ├── config.py              # env vars, default models, analysis knobs
│   ├── edgar.py               # SEC HTTP (User-Agent, throttle, retries), ticker→CIK, 10-K, HTML→text, Items 1/1A/7
│   ├── financials.py          # XBRL company facts → 5-year table, formatting helpers
│   ├── market.py              # yfinance price/market cap, P/E, EV/Sales, FCF yield
│   ├── red_flags.py           # rule-based red flags (numbers + 10-K text scan)
│   ├── transcript.py          # .txt/.pdf transcript → text, prepared remarks vs Q&A
│   ├── retrieval.py           # chunking, BM25, per-memo-section queries
│   ├── llm.py                 # Anthropic / OpenAI / Gemini wrapper, JSON parsing
│   ├── committee.py           # evidence pack, Bull/Bear/Risk/Chair prompts, citation enforcement
│   ├── memo.py                # Markdown memo (committee and data-only layouts)
│   └── pipeline.py            # runs every step → AnalysisResult
├── scripts/
│   ├── smoke_test.py          # CLI: python scripts/smoke_test.py AAPL [--transcript f] [--provider p]
│   └── fetch_fixtures.py      # downloads the real-filing test fixtures (optional)
└── tests/                     # offline pytest suite (no network, no API keys)
    ├── conftest.py
    ├── fixtures/sample_transcript.txt   # synthetic call for a fictional company (parser tests)
    └── test_*.py              # tests per copilot module (app tests live in test_pipeline.py)
```

## Setup

You need Python 3.11 or newer and an internet connection (SEC EDGAR and Yahoo Finance).

**1. Install**

macOS / Linux:

```bash
cd ai-ic-copilot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Windows (PowerShell). This calls the venv's Python directly, so it works even where script execution (`Activate.ps1`) is disabled:

```powershell
cd ai-ic-copilot
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env
```

On Windows, keep the project in a short path, such as `C:\code\ai-ic-copilot`. Some SDK files have very long names, and pip fails with "No such file or directory" when the path passes Windows' 260-character limit (unless long paths are enabled). From here on, run `.venv\Scripts\python -m streamlit run app.py` and `.venv\Scripts\python -m pytest -q` in place of the bare commands below.

**2. Configure `.env`**

```ini
# Required: SEC asks every automated client for a name and contact email.
SEC_USER_AGENT="Jane Doe jane.doe@example.com"

# Optional: set ONE of these to enable the committee. Leave all blank for data-only mode.
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=
GEMINI_API_KEY=
```

| Variable | Default | Purpose |
|---|---|---|
| `SEC_USER_AGENT` | — (required) | Sent as the `User-Agent` header on every SEC request ([SEC policy](https://www.sec.gov/os/accessing-edgar-data)). You can also type it under **Settings** in the app. |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | — | Every provider that has a key appears in **Settings → LLM provider**. |
| `ANTHROPIC_MODEL` | `claude-opus-5-5` | Claude model. Uses server-side refusal fallback and `ANTHROPIC_EFFORT`. |
| `ANTHROPIC_EFFORT` | `medium` | `low` / `medium` / `high` / `xhigh` / `max`. Higher means a deeper debate, slower and costlier. |
| `OPENAI_MODEL` | `gpt-6.1-sol` | OpenAI Responses API model (`gpt-6-astra` is stronger and pricier). |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Gemini model, via the google-genai SDK (`gemini-4-argon` is stronger and pricier). |

**3. Run**

```bash
streamlit run app.py
```

Open http://localhost:8501. Type a ticker, optionally upload a transcript, and click **Convene the committee**. The **Settings** expander in the same form holds the SEC User-Agent, the LLM provider (or data-only mode) and an optional model override. Settings only apply when you submit, so changing one never interrupts a run.

**Command line** (same pipeline, no UI):

```bash
python scripts/smoke_test.py NVDA
python scripts/smoke_test.py NVDA --transcript nvda_call.pdf --provider anthropic --out nvda_memo.md
```

**Tests** (offline: no network, no keys):

```bash
pytest -q
```

About 100 regression tests replay real EDGAR filings for 19 companies (about 140 MB, git-ignored). They skip until you download them once:

```bash
python scripts/fetch_fixtures.py
```

## Getting an earnings-call transcript

The app never scrapes. Upload a `.txt` or `.pdf` you have the right to use. Many companies post transcripts or prepared remarks on their investor-relations site, and some file them with the SEC as an 8-K exhibit. You can also paste a transcript into a text file. If the transcript never mentions the company or its ticker, the app **does not use it** and tells you why.

`tests/fixtures/sample_transcript.txt` is a **synthetic call for a fictional company**, used to test the parser. Upload it with a real ticker and the app will correctly refuse it.

## The memo

1. **Investment thesis**: three claims that drive the vote
2. **Business overview**
3. **Financial snapshot**: a five-year table built from SEC data, plus a valuation table and notes
4. **Bull case**
5. **Bear case**
6. **Top risks**, rated by severity
7. **What management said vs. what the numbers show**: call or MD&A statements checked against XBRL figures
8. **Red flags**: the automated checks plus the committee's view of each
9. **Questions for the committee**, each tied to the evidence it probes
10. **Vote**: Buy / Hold / Pass, conviction, a rationale that has to address valuation, and the triggers that would change the call
- **Appendix A**: every cited evidence item, with its document and section
- **Appendix B**: the citation audit (claims kept and dropped, invalid IDs, quotes verified and removed)

### How "no citation, no claim" is enforced

1. Every fact the models may use gets an ID: `F#` for an XBRL or valuation line (plus one "data notes" item with the metric definitions), `R#` for a red-flag check, `E#` for a 10-K or transcript passage.
2. The prompts require JSON claims of the form `{"text", "citations", "quote"}`. They tell the models that every number must come from a cited item, and that the documents are data whose instructions must be ignored.
3. `committee.enforce_citations` checks every claim the Chair makes, including the questions and the triggers that would change the vote:
   - It normalizes IDs (`[e3]` → `E3`), including tags typed inside the prose, and removes IDs that are not in the evidence pack.
   - It **deletes a claim** that has no valid citation left, or that states a number its cited items don't contain. Thresholds in "what would change the call" are exempt.
   - It drops a "management said vs. numbers show" row unless the row cites both a passage (E) and a number (F).
   - It checks each quote word for word against the cited passages (case, whitespace and curly quotes are ignored) and blanks the quote if it doesn't match.
   - It withholds the vote if none of the Chair's rationale claims survive.
4. The memo never shows a dropped claim. Appendix B counts them, and the app's **Diagnostics** tab lists each one with the reason it was dropped.

## Design choices and limitations

- **Simple over clever.** Plain functions and dataclasses, a hand-written BM25 index of about 30 lines, and no agent framework or vector database.
- **Latest fiscal year, not TTM.** Multiples use the last 10-K, and the memo and UI label them that way.
- **XBRL isn't uniform.** Debt follows a documented rule (long-term debt including the current portion, plus short-term borrowings, excluding leases). Companies that tag debt with custom tags (Ford, Berkshire) get no EV rather than a wrong one. Banks and insurers have no gross margin or capex, so those rows show `n/a`. The notes under the table explain every gap.
- **Section parsing is heuristic.** In testing on 38 recent 10-Ks, 32 parsed all three sections. The other 6 (GE, Intel and McDonald's use a "cross-reference index" layout; JPMorgan and Deere point Item 7 elsewhere; Wells Fargo puts it in an exhibit) fall back to the full text. Those passages are labelled "10-K text" rather than given a guessed section name.
- **Red flags are screens, not verdicts.** They are deterministic and explain their numbers. The Risk officer is asked to call each one real or benign.
- **BM25 is lexical.** It finds passages that share words with each section's query, so retrieval quality depends on the vocabulary of the filing.
- **The LLM can still misread evidence.** The audit catches fabricated citations, numbers and quotes. It cannot catch a real number attached to the wrong claim. Read the cited passages before you vote.
- **Supported filers.** US domestic filers that file a 10-K. Foreign private issuers (20-F/40-F) get a clear message instead.

## 2-minute demo script

*Before you present:* run your demo ticker once so SEC responses are cached in memory, have the transcript file on your desktop, and keep a second ticker ready. If the API is down or you have no key, pick **None — data-only mode** under Settings. Data-only runs take about 5 seconds; committee runs took 93–130 seconds in testing, so start the run before you begin talking through it.

| Time | Do | Say |
|---|---|---|
| 0:00–0:15 | App open, empty. | "Before a committee votes on a stock, an analyst spends hours turning a 10-K and an earnings call into a memo. This copilot drafts that memo in about two minutes, and every sentence is traceable to a source." |
| 0:15–0:35 | Type `[TICKER]`, upload the transcript, click **Convene the committee**. The status box streams the steps. | "It maps the ticker to its SEC CIK, pulls the latest 10-K and finds Items 1, 1A and 7. It gets five years of XBRL financials, market data from Yahoo, and indexes everything with BM25." |
| 0:35–0:55 | **Memo** tab: the vote line, then the 3-bullet thesis. Point at a `[E3 · 10-K Item 1A]` citation. | "Three agents argued: a bull, a bear and a risk officer. Then a chair checked their arguments against the evidence, wrote this memo and voted. Every claim cites an evidence ID, and quotes are checked word for word against the filing." |
| 0:55–1:15 | Scroll to **Financial snapshot** and **What management said vs. what the numbers show**. | "The numbers are never written by the LLM. They come straight from SEC data. And this table checks what management said on the call against what the filings show." |
| 1:15–1:35 | **Red flags** tab. | "These are deterministic checks. Here receivables grew faster than revenue. Each flag shows its numbers and the XBRL tags it came from, and the committee has to say whether it's a real problem or benign." |
| 1:35–1:50 | **Diagnostics** tab, citation audit. | "This is the guardrail. If a claim has no valid citation, or cites a number that isn't in its source, it gets deleted. So does any quote that isn't in the filing. On this run it dropped 2 claims and removed 0 quotes." *(Measured on the 2026-10-20 PLTR run with the final prompts; across sixteen test runs the audit dropped 0–3 claims and removed 0–3 quotes per run. Read the real counts for your run from Appendix B.)* |
| 1:50–2:00 | Click **Download memo (.md)**. | "One click exports the memo. With no API key it still runs and shows the financials, flags and evidence, so the demo never breaks." |

## License

[MIT](LICENSE). Educational project, not investment advice.
