# PubMed → Google Scholar Scraper & Report

This project finds papers for a list of authors, enriches them with PubMed metadata, looks them up on Google Scholar (via SerpAPI), and produces a clean HTML report plus a CSV you can annotate.

---

## What it does

1. Fetch PMIDs from Europe PMC for each author in your CSV (optionally filtered by date).
2. Scrape PubMed for each PMID using Playwright:
   - title, abstract, authors, affiliations, keywords, DOI, journal info, grants, etc.
3. Query Google Scholar (via SerpAPI) per author:
   - first tries: `"<author>" AND (term1 OR term2 …)`
   - then falls back to author‑only relevance search if needed
   - optionally tries a title‑only search
4. Capture Scholar details:
   - matching result link, citations, snippet, and which search terms appear in the snippet.
5. Normalize affiliations (e.g., `"12 Biological Physics Group, …"` → `"Biological Physics Group"`) and compute per‑author and per‑affiliation metrics.
6. Generate outputs:
   - A searchable HTML report with charts, highlighted terms in snippets, and the ability to annotate entries with a dropdown box.
   - A companion CSV (same table) with blank `Claim` and `Notes` columns for you to fill in.
   - The pipeline also saves intermediate CSVs along the way.

---

## Project layout (key files)

```
.
├─ LiteratureSearch.py   		# Main pipeline: Europe PMC → PubMed → Scholar → outputs
├─ GenerateReport.py             # Builds the HTML report and the editable CSV
├─ environment.yml               # Conda environment (includes Playwright etc.)
├─ .env                          # Your settings (you create this)
└─ authors.csv                   # Your input list of authors (you provide this)
```

---

## Prerequisites

- Conda / Miniconda installed
- A SerpAPI key (for Google Scholar). Create a free/paid key at serpapi.com.

---

## 1) Create and activate the conda environment

```bash
# From the project folder
conda env create -f environment.yml
conda activate litreport
```

---

## 2) Install the Playwright browser

Playwright needs a browser binary (Chromium) installed inside the environment:

```bash
playwright install
```

---

## 3) Prepare your input files

### `authors.csv` (example)

A minimal sheet looks like this (headers matter):

```csv
First Name,Surname,e-mail
Michael,Haley,michael.haley@university.com
John,Smith,js@uni.com
```

> You can have more columns—these three are the ones used in the pipeline to build the author’s full name and map back to an email.

### `.env` (your settings)

Create a file named `.env` in the project folder. Example:

```dotenv
# Google Search API key
# You can create a free account that gives you 250 searches a month: https://serpapi.com/
SERPAPI_KEY=YOUR_KEY_HERE

# This should be a .csv file with columns 'First Name', 'Surname', and 'e-mail'
AUTHORS_CSV=authors.csv

# Filter to papers published between specific dates
# Use American format dates, or "none" if not used
DATE_FILTER_START=2024-01-01
DATE_FILTER_END=None

# Only authors with an affiliation that contains this will be included
AFFILIATION_FILTERS=Manchester,Oxford

# Search terms to look for on Google Scholar search
SCHOLAR_TERMS=inflammation,malaria,imaging

# Drop down options for the report
STATUS_COLUMN=Acknowledgement
STATUS_OPTIONS=Acknowledged,Authorship,Used without acknowledgement,Not used

########### Other options, only really used for debugging

# Run Google Scholar search
RUN_SCHOLAR = True

# Sleep between Scholar searches
SCHOLAR_SLEEP = 2

# Number of characters to match between PubMed and Google Scholar titles
SCHOLAR_PREFIX_N = 25

# Whether to do extra Google Scholar searches to retrieve links using only title if other searches didn't retreive a match
ENABLE_TITLE_SEARCH = True

# Whether to filter to only authors that have an affiliation that matches the filter above - if not we will get all author matches, regardless of institution. Will also remove reviews/journals/editorials
APPLY_FILTER = True
```

Notes
- `PUBMED_INPUT_CSV`: if you already ran PubMed scraping before, set this to that CSV to skip scraping and just do the Scholar stage.
- `DATE_FILTER_START`/`END`: set to `None` to remove a bound.
- `SCHOLAR_TERMS`: terms will be OR’d together; keep them concise to avoid super‑long queries.

---

## 4) Run the pipeline

From the project folder (with the env activated):

```bash
python LiteratureSearch.py
```

What you’ll see in the console:
- Europe PMC author queries with date filters
- PubMed scraping progress (one URL per PMID)
- Google Scholar matching progress
- Report generation message at the end

---

## 5) Outputs & where to find them

- Main Scholar output CSV (configurable; defaults with a date stamp), e.g.:
  - `pubmed_googlescholar_YYYY-MM-DD.csv`

- HTML report (auto‑generated from the Scholar CSV):
  - Same name as above, ending `.html`, e.g. `pubmed_googlescholar_YYYY-MM-DD.html`
  - Open it in your browser. It includes:
    - Summary cards
    - Charts (top authors, top affiliations, publications over time)
    - Per‑author and per‑affiliation tables
    - A searchable full results table (type in the search box)

- Report CSV for annotation:
  - Same base name as the HTML, ending `.csv`, e.g. `pubmed_googlescholar_YYYY-MM-DD.csv`
  - This companion CSV contains raw URLs and blank `Claim` / `Notes` columns so you can mark things up directly in Excel/Sheets.

  The report CSV includes these URL fields as plain text:
  - `google_url` (Google Scholar result URL)
  - `pubmed_url`  (built from PMID)

- Intermediate CSVs:
  - PubMed scrape output: `pubmed_unfiltered_YYYY-MM-DD.csv` (or whatever you set in `.env`)
  - Author counts summary: `author_counts_YYYY-MM-DD.csv`

---

## How to re‑run just the report

If you only want to rebuild the HTML/annotatable CSV from an existing Scholar CSV:

```bash
python GenerateReport.py pubmed_googlescholar_YYYY-MM-DD.csv
# or
python GenerateReport.py input.csv output.html
```

This will:
- Generate `output.html`
- Generate a sibling `output.csv` (the annotatable copy with `Claim` and `Notes`)

---

## Understanding key columns (cheat‑sheet)

- PubMed / Europe PMC
  - `pmid` — PubMed ID
  - `title_pubmed` — title from PubMed
  - `affiliations` — full raw affiliations text
  - `matched_affiliation` — the affiliation for the matched author (if found)
  - `affiliation_match` — `True/False` based on your filters
  - `firstPublicationDate` — publication date (YYYY, YYYY‑MM, or YYYY‑MM‑DD)
  - `doi` — DOI if found

- Google Scholar (via SerpAPI)
  - `scholar_match_title`, `scholar_match_url`, `scholar_match_citations`
  - `scholar_snippet` — short text around the match
  - `scholar_terms_found` — which terms (from your list) appear in the snippet
  - `scholar_title_match_found` — True if a result was matched
  - `scholar_matched_via` — `"author+terms"`, `"author_only"`, `"title_only"`, or `"no_match"`

- Report artifacts
  - `affil_group` — normalized affiliation head (e.g., `"Biological Physics Group"`)
  - `google_url`, `pubmed_url` — raw URLs added for your convenience in the CSV that accompanies the HTML report

---

## Troubleshooting

### 1) Playwright timeouts / “Chromium not found”
Run:
```bash
python -m playwright install chromium
```
If timeouts persist, your network may be slow—consider increasing timeouts in the scraper or running fewer PMIDs at a time.

### 2) SerpAPI errors / 429 rate limit
- Ensure `SERPAPI_KEY` is present in `.env`.
- Reduce request rate (`SCHOLAR_SLEEP=3` or `4`).
- Limit the number of authors/terms or batch runs.

### 3) “Very long query” / no results for huge term lists
Google/Scholar queries above ~2,000 URL characters can fail or silently drop terms. Keep term lists short and specific.
