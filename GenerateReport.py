#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import base64
import io
import re
import sys
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import matplotlib.pyplot as plt
import hashlib
from dotenv import load_dotenv

# =========================
# .env helpers & load
# =========================
def get_env_bool(key, default=False):
    val = os.getenv(key, str(default))
    return str(val).lower() in ("true", "1", "yes", "y", "on")

def get_env_list(key, sep=","):
    val = os.getenv(key, "")
    return [x.strip() for x in val.split(sep) if x.strip()]

load_dotenv()

# =========================
# Dynamic CSS helpers
# =========================

def _css_escape_attr_value(s: str) -> str:
    # Safe for CSS attribute selectors like [value='...']
    return str(s).replace("\\", "\\\\").replace("'", "\\'")

def _hue_from_label(label: str) -> int:
    # Deterministic hue from label text
    h = int(hashlib.sha1(label.encode("utf-8")).hexdigest(), 16)
    return h % 360

def build_status_css(options) -> str:
    """
    For each non-blank option in DROPDOWN_OPTIONS, generate a row highlight and
    left-accent color using a deterministic pastel from its label.
    """
    rules = ["/* Auto-generated per-Status row colors */"]
    for opt in options:
        if not opt or not str(opt).strip():
            continue
        hue = _hue_from_label(opt)
        bg = f"hsl({hue}, 85%, 94%)"     # pastel background (light)
        accent = f"hsl({hue}, 70%, 38%)" # strong accent for left border
        val = _css_escape_attr_value(opt)
        rules.append(
            f"#results tbody tr:has(select.dd option:checked[value='{val}'])"
            " { "
            f"background: {bg};"
            " }"
        )
        rules.append(
            f"#results tbody tr:has(select.dd option:checked[value='{val}']) td:first-child"
            " { "
            f"border-left: 4px solid {accent};"
            " }"
        )
    return "\n".join(rules)

DATE_STR = date.today().isoformat()

# --------- configuration with precedence: CLI > .env > defaults ----------
# input/output files
_env_input = os.getenv("PUBMED_INPUT_CSV", None)
_env_output = os.getenv("OUTPUT_HTML", None)

INPUT_CSV = (
    sys.argv[1] if len(sys.argv) > 1 else
    (_env_input if _env_input else f"pubmed_googlescholar_{DATE_STR}.csv")
)
OUTPUT_HTML = (
    sys.argv[2] if len(sys.argv) > 2 else
    (_env_output if _env_output else f"publication_report_{DATE_STR}.html")
)

# feature toggles
ENABLE_EXPORT = get_env_bool("ENABLE_EXPORT", True)
ENABLE_LOCALSTORAGE = get_env_bool("ENABLE_LOCALSTORAGE", True)
ENABLE_CHARTS = get_env_bool("ENABLE_CHARTS", True)
CHART_TOP_N = int(os.getenv("CHART_TOP_N", "10"))

# optional filtering of rows by affiliation group (case-insensitive contains)
APPLY_FILTER = get_env_bool("APPLY_FILTER", True)
AFFILIATION_FILTERS = tuple(get_env_list("AFFILIATION_FILTERS"))  # e.g. "Biology, Physics"

# dropdown config for the HTML table
DROPDOWN_NAME = os.getenv("STATUS_COLUMN", "Status")
_status_options = get_env_list("STATUS_OPTIONS")  # e.g. "Include,Exclude,Maybe"
_include_blank = get_env_bool("STATUS_INCLUDE_BLANK", True)
if _status_options:
    DROPDOWN_OPTIONS = ([""] if _include_blank else []) + _status_options
else:
    DROPDOWN_OPTIONS = ([""] if _include_blank else []) + ["Include", "Exclude", "Maybe"]

# Column names in your CSV (adjust if your file differs)
COL_PMID = "pmid"
COL_PAPER_TITLE = "title_pubmed"   # fallback to "title" if not present
COL_AUTHOR = "search_author"
COL_DATE = "firstPublicationDate"  # may be missing / partial date
COL_DOI = "doi_pubmed"
COL_CITES = "scholar_match_citations"
COL_GOOGLE_URL = "scholar_match_url"   # raw url in source df
COL_SNIPPET = "scholar_snippet"
COL_TERMS_FOUND = "scholar_terms_found"
COL_VIA = "scholar_matched_via"   # "author+terms" | "author_only" | "no_match"
COL_MATCH_BOOL = "scholar_title_match_found"
COL_AFFILIATIONS_RAW = "affiliations"            # optional
COL_AFFILIATION_MATCHED = "matched_affiliation"  # preferred if present

# (auto-detected) grants column candidates
GRANT_CANDIDATES = ["grants", "grant_ids", "funding", "funding_ids", "funding_text"]

# ---------- helpers ----------
def read_head_cols(path):
    return pd.read_csv(path, nrows=1).columns.tolist()

cols_available = read_head_cols(INPUT_CSV)
def col_exists(c): return c in cols_available

def pick_grants_col():
    # Allow optional override via env
    override = os.getenv("GRANTS_COLUMN", "").strip()
    if override and override in cols_available:
        return override
    for c in GRANT_CANDIDATES:
        if c in cols_available:
            return c
    return None

COL_GRANTS = pick_grants_col()

# Fallbacks if some columns differ/missing
if not col_exists(COL_PAPER_TITLE):
    COL_PAPER_TITLE = "title" if col_exists("title") else None
if not col_exists(COL_DATE):
    COL_DATE = None
if not col_exists(COL_CITES):
    COL_CITES = None
if not col_exists(COL_VIA):
    COL_VIA = None
if not col_exists(COL_MATCH_BOOL):
    COL_MATCH_BOOL = None

# --- affiliation cleaner: strip leading digits/spaces; take text before first comma
LEADING_NUMS_SPACES = re.compile(r"^\s*\d+\s*")

def clean_single_affil(s: str) -> str:
    if not isinstance(s, str) or not s.strip():
        return ""
    s = LEADING_NUMS_SPACES.sub("", s.strip())
    first = s.split(",", 1)[0].strip()
    return first

def derive_affil_group(row) -> str:
    raw = row.get(COL_AFFILIATION_MATCHED, "") if col_exists(COL_AFFILIATION_MATCHED) else ""
    if not raw and col_exists(COL_AFFILIATIONS_RAW):
        raw_all = str(row.get(COL_AFFILIATIONS_RAW, "") or "")
        raw = raw_all.split(";")[0].strip()
    return clean_single_affil(raw)

def parse_date_maybe(s):
    if pd.isna(s): return pd.NaT
    s = str(s)
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except Exception:
            pass
    return pd.to_datetime(s, errors="coerce")

def make_chart_uri(plot_fn):
    fig = plt.figure(figsize=(6, 6))
    try:
        plot_fn()
        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    finally:
        plt.close(fig)

# HTML escaper (basic)
def esc(x: str) -> str:
    if pd.isna(x): return ""
    return str(x).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

# Highlight terms (case-insensitive) using <mark>
def highlight_terms(html_text: str, terms: list[str]) -> str:
    if not html_text or not terms:
        return html_text
    pats = [re.escape(t.strip()) for t in terms if t and t.strip()]
    if not pats:
        return html_text
    pattern = re.compile(r"(" + "|".join(pats) + r")", flags=re.IGNORECASE)
    return pattern.sub(r"<mark>\1</mark>", html_text)

# Build HTML link
def html_link(url, text):
    if pd.isna(url) or not str(url).strip():
        return ""
    t = (text or url).replace("<","&lt;").replace(">","&gt;")
    u = str(url).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").strip()
    return f'<a href="{u}" target="_blank" rel="noopener">{t}</a>'

def existing(df, cols):
    return [c for c in cols if c and c in df.columns]

def mask_secret(s: str) -> str:
    if not s:
        return ""
    s = str(s)
    if len(s) <= 6:
        return "*" * max(len(s) - 2, 0) + s[-2:]
    return s[:2] + "…" + s[-4:]

# Render a dropdown <select> for a given row id
def make_dropdown_html(row_id: int) -> str:
    opts = "".join(
        f'<option value="{esc(o)}">{esc(o) if o else "—"}</option>'
        for o in DROPDOWN_OPTIONS
    )
    return f'<select class="dd" data-row="{row_id}">{opts}</select>'

def first_nonempty_date_from_df(df, candidates=("DATE_RETRIEVED","retrieved","date_retrieved","retrieved_date")):
    for c in candidates:
        if c in df.columns:
            ser = df[c].dropna().astype(str)
            if not ser.empty:
                return ser.iloc[0].replace("/", "-")
    return None

# ---------- load & prepare data ----------
df = pd.read_csv(INPUT_CSV)

# basic coercions
if COL_CITES and COL_CITES in df.columns:
    df[COL_CITES] = pd.to_numeric(df[COL_CITES], errors="coerce").fillna(0).astype(int)

if COL_DATE and COL_DATE in df.columns:
    df["__pubdate"] = df[COL_DATE].apply(parse_date_maybe)
else:
    df["__pubdate"] = pd.NaT

# DATE_RETRIEVED (from DF, else env, else today)
DATE_RETRIEVED = first_nonempty_date_from_df(df) or os.getenv("DATE_RETRIEVED", DATE_STR)

# derive affiliation group early
df["affil_group"] = df.apply(derive_affil_group, axis=1)

# optional filtering by affiliation keywords (if configured)
#if APPLY_FILTER and AFFILIATION_FILTERS:
    # Build a case-insensitive regex OR pattern from filters
#    safe_parts = [re.escape(x) for x in AFFILIATION_FILTERS if x]
#    if safe_parts:
#        patt = re.compile("|".join(safe_parts), flags=re.IGNORECASE)
#        df = df[df["affil_group"].astype(str).apply(lambda s: bool(patt.search(s)))]

# ---- build link columns (HTML + raw) BEFORE subsets ----
# Raw URLs:
df["google_url"] = df.get(COL_GOOGLE_URL, "")
if COL_PMID in df.columns:
    df["pubmed_url"] = df[COL_PMID].map(lambda p: f"https://pubmed.ncbi.nlm.nih.gov/{p}/")
else:
    df["pubmed_url"] = ""

# Clickable links for the HTML table
df["Google Link"] = df["google_url"].map(lambda u: html_link(u, "Google Link"))
df["PubMed Link"] = df["pubmed_url"].map(lambda u: html_link(u, "PubMed Link"))

# Prepare snippet with highlights (for HTML)
def build_highlighted_snippet(row):
    snip = row.get(COL_SNIPPET, "")
    if pd.isna(snip):
        return ""   # return empty instead of "nan"
    snip = esc(snip)

    terms_str = row.get(COL_TERMS_FOUND, "")
    if pd.isna(terms_str) or not str(terms_str).strip():
        return snip

    terms = [t.strip() for t in str(terms_str).split(";") if t.strip()]
    return highlight_terms(snip, terms)

df["__snippet_html"] = df.apply(build_highlighted_snippet, axis=1)

# =========================
# Metrics & summaries
# =========================
total_rows = len(df)
matched_rows = int(df[COL_MATCH_BOOL].sum()) if COL_MATCH_BOOL and COL_MATCH_BOOL in df.columns else 0
terms_matched_rows = (
    int((df.get("scholar_terms_matched", False) == True).sum())
    if "scholar_terms_matched" in df.columns
    else int(df.get(COL_TERMS_FOUND, "").fillna("").astype(str).str.len().gt(0).sum())
)
unique_authors = df[COL_AUTHOR].nunique() if COL_AUTHOR in df.columns else None
total_citations = int(df[COL_CITES].sum()) if COL_CITES and COL_CITES in df.columns else 0
unique_affils = df["affil_group"].replace("", pd.NA).dropna().nunique()

# ---------- Grants analytics ----------
if COL_GRANTS and COL_GRANTS in df.columns:
    # Normalize grants list (split by semicolon, strip, drop empties)
    grants_df = (
        df[[COL_PAPER_TITLE, COL_GRANTS]]
        .assign(_gl=lambda x: x[COL_GRANTS].fillna("").astype(str).str.split(";"))
        .explode("_gl")
        .assign(grant=lambda x: x["_gl"].astype(str).str.strip())
    )
    # drop blanks and known empties
    grants_df = grants_df[grants_df["grant"].astype(bool) & (grants_df["grant"].str.lower() != "nan")]

    grants_counts = (
        grants_df.groupby("grant", dropna=False)[COL_PAPER_TITLE]
        .nunique()
        .reset_index(name="publications")
        .sort_values("publications", ascending=False)
    )
    top_grants = grants_counts.head(CHART_TOP_N).copy() if not grants_counts.empty else pd.DataFrame()
    papers_with_any_grant = int(df[COL_GRANTS].fillna("").astype(str).str.strip().astype(bool).sum())
else:
    grants_counts = pd.DataFrame()
    top_grants = pd.DataFrame()
    papers_with_any_grant = 0

# =========================
# Aggregate tables
# =========================
# per-author summary
if COL_AUTHOR in df.columns:
    per_author = (
        df.groupby(COL_AUTHOR)
          .agg(
              total=("pmid", "count") if "pmid" in df.columns else (COL_PAPER_TITLE, "count"),
              matched=(COL_MATCH_BOOL, "sum") if COL_MATCH_BOOL in df.columns else ("google_url", lambda s: s.astype(str).str.len().gt(0).sum()),
              terms_matched=("scholar_terms_matched", "sum") if "scholar_terms_matched" in df.columns else (COL_TERMS_FOUND, lambda s: s.fillna("").str.len().gt(0).sum()),
              citations=(COL_CITES, "sum") if COL_CITES and COL_CITES in df.columns else (COL_PAPER_TITLE, "count"),
          )
          .sort_values(["matched","total"], ascending=[False, False])
          .reset_index()
    )
else:
    per_author = pd.DataFrame()

# per-affiliation summary
per_affil = (
    df.assign(__has_terms=lambda d: d.get("scholar_terms_matched", d.get(COL_TERMS_FOUND, "").astype(str).str.len().gt(0)))
      .groupby("affil_group", dropna=False)
      .agg(
          total=("pmid","count") if "pmid" in df.columns else (COL_PAPER_TITLE, "count"),
          matched=(COL_MATCH_BOOL, "sum") if COL_MATCH_BOOL in df.columns else ("google_url", lambda s: s.astype(str).str.len().gt(0).sum()),
          terms_matched=("__has_terms", "sum"),
          citations=(COL_CITES, "sum") if COL_CITES and COL_CITES in df.columns else (COL_PAPER_TITLE, "count"),
      )
      .sort_values(["matched","total"], ascending=[False, False])
      .reset_index()
)

# top by citations
if COL_CITES and COL_CITES in df.columns and not df.empty:
    top_by_cites = df.sort_values(COL_CITES, ascending=False).head(20).copy()
else:
    top_by_cites = pd.DataFrame()

# recent papers
recent = df.dropna(subset=["__pubdate"]).sort_values("__pubdate", ascending=False).head(20)

# timeline
timeline = (
    df.dropna(subset=["__pubdate"])
      .assign(month=lambda d: d["__pubdate"].dt.to_period("M").dt.to_timestamp())
      .groupby("month")
      .size()
      .reset_index(name="count")
      .sort_values("month")
)

# =========================
# Charts
# =========================
if ENABLE_CHARTS and not per_author.empty:
    top_auth_chart_uri = make_chart_uri(lambda: (
        plt.bar(per_author.head(CHART_TOP_N)[COL_AUTHOR], per_author.head(CHART_TOP_N)["matched"]),
        plt.xticks(rotation=45, ha="right"),
        plt.title("Top authors by matched Scholar titles"),
        plt.ylabel("Matched papers"),
        plt.xlabel("Author"),
    ))
else:
    top_auth_chart_uri = ""

if ENABLE_CHARTS and not per_affil.empty:
    top_affil_chart_uri = make_chart_uri(lambda: (
        plt.bar(per_affil.head(CHART_TOP_N)["affil_group"], per_affil.head(CHART_TOP_N)["matched"]),
        plt.xticks(rotation=45, ha="right"),
        plt.title("Top affiliations by matched Scholar titles"),
        plt.ylabel("Matched papers"),
        plt.xlabel("Affiliation group"),
    ))
else:
    top_affil_chart_uri = ""

if ENABLE_CHARTS and not timeline.empty:
    timeline_chart_uri = make_chart_uri(lambda: (
        plt.plot(timeline["month"], timeline["count"]),
        plt.title("Publications over time"),
        plt.ylabel("Count"),
        plt.xlabel("Month"),
    ))
else:
    timeline_chart_uri = ""

if ENABLE_CHARTS and not top_grants.empty:
    top_grants_chart_uri = make_chart_uri(lambda: (
        plt.bar(top_grants["grant"], top_grants["publications"]),
        plt.xticks(rotation=45, ha="right"),
        plt.title("Top grants by publications"),
        plt.ylabel("Publications"),
        plt.xlabel("Grant"),
    ))
else:
    top_grants_chart_uri = ""

# ------------------------------
# Build main table for HTML view
# ------------------------------
view_cols_html = []
for c in [
    COL_AUTHOR, COL_DOI, "affil_group", COL_DATE,
    COL_PAPER_TITLE, "Google Link", "PubMed Link",
    COL_SNIPPET, COL_TERMS_FOUND, COL_VIA, COL_CITES
]:
    if c == COL_SNIPPET:
        if COL_SNIPPET in df.columns:
            view_cols_html.append("__snippet_html")  # highlighted version
    elif c and c in df.columns:
        view_cols_html.append(c)

display_names_html = {"__snippet_html": "scholar_snippet"}
table_df_html = df[view_cols_html].copy().rename(columns=display_names_html)

# Insert dropdown column at the front of the HTML table
table_df_html.insert(0, DROPDOWN_NAME, [make_dropdown_html(i) for i in range(len(table_df_html))])

# Escape non-HTML fields for the HTML report
for col in table_df_html.columns:
    if col in ("Google Link", "PubMed Link", "scholar_snippet", DROPDOWN_NAME):
        continue
    table_df_html[col] = table_df_html[col].map(esc)

# Include the pandas index column in HTML (keeps Export CSV clean)
table_html = table_df_html.to_html(index=True, escape=False)

# ------------------------------
# Build CSV for user annotation (baseline copy on disk)
# ------------------------------
# Use plain snippet (no <mark>), and include raw URLs + add Claim/Notes/Status columns
view_cols_csv = []
for c in [
    COL_AUTHOR, COL_DOI, "affil_group", COL_DATE,
    COL_PAPER_TITLE, "google_url", "pubmed_url",  # RAW URL columns
    COL_SNIPPET, COL_TERMS_FOUND, COL_VIA, COL_CITES
]:
    if c and c in df.columns:
        view_cols_csv.append(c)

csv_df = df[view_cols_csv].copy()
csv_df["Claim"] = ""
csv_df["Notes"] = ""
csv_df[DROPDOWN_NAME] = ""  # mirror dropdown choice for manual annotation

csv_output = Path(OUTPUT_HTML).with_suffix(".csv")
csv_df.to_csv(csv_output, index=True)
print(f"📝 Base table saved for annotation → {csv_output}")

# ------------------------------
# Settings table (from .env + derived)
# ------------------------------
settings_rows = []

def add_setting(k, v, e):
    settings_rows.append({"Setting": k, "Value": "" if v is None else str(v), "Explanation":e})

# Core pipeline/search settings from LiteratureSearch.py
add_setting("DATE_RETRIEVED", DATE_RETRIEVED, "Date when the data was retrieved from PubMed")
add_setting("AUTHORS_CSV", os.getenv("AUTHORS_CSV", "authors.csv"), "Path to the authors CSV file")
add_setting("PUBMED_OUTPUT", os.getenv("PUBMED_OUTPUT", f"pubmed_unfiltered_{DATE_STR}.csv"), "Path to the PubMed output CSV file")
#add_setting("COUNTS_CSV", os.getenv("COUNTS_CSV", f"author_counts_{DATE_STR}.csv"), "Path to the counts CSV file")
add_setting("DATE_FILTER_START", os.getenv("DATE_FILTER_START", ""), "Start date for filtering")
add_setting("DATE_FILTER_END", os.getenv("DATE_FILTER_END", ""), "End date for filtering")
add_setting("AFFILIATION_FILTERS", ", ".join(AFFILIATION_FILTERS) if AFFILIATION_FILTERS else "", "Affiliation keywords used to ensure correct author identification")
add_setting("PUBMED_INPUT_CSV", os.getenv("PUBMED_INPUT_CSV", ""), "Path to the PubMed input CSV file")
add_setting("RUN_SCHOLAR", get_env_bool("RUN_SCHOLAR", True), "Whether to run Google Scholar search")
add_setting("SCHOLAR_TERMS", ", ".join(get_env_list("SCHOLAR_TERMS")), "Search terms for Google Scholar")
add_setting("SCHOLAR_OUTPUT", os.getenv("SCHOLAR_OUTPUT", f"pubmed_googlescholar_{DATE_STR}.csv"), "Path to the Google Scholar output CSV file")
#add_setting("SCHOLAR_SLEEP", os.getenv("SCHOLAR_SLEEP", "2"), "Sleep time between Google Scholar requests")
#add_setting("SCHOLAR_PREFIX_N", os.getenv("SCHOLAR_PREFIX_N", "25"), "Number of results to retrieve from Google Scholar")
add_setting("APPLY_FILTER", APPLY_FILTER, "Whether to filter searches to only matched affiliations, and exclude reviews/editorials/letters")
#add_setting("ENABLE_TITLE_SEARCH", get_env_bool("ENABLE_TITLE_SEARCH", True), "Whether to enable title search")
#add_setting("SERPAPI_KEY", mask_secret(os.getenv("SERPAPI_KEY", "")), "API key for SerpApi")

# Report generation context
add_setting("REPORT_INPUT_CSV", INPUT_CSV, "Path to the report input CSV file")
add_setting("REPORT_OUTPUT_HTML", OUTPUT_HTML, "Path to the report output HTML file")
add_setting("STATUS_COLUMN", DROPDOWN_NAME, "Name of the status column")
add_setting("STATUS_OPTIONS", ", ".join(DROPDOWN_OPTIONS), "Options for the status dropdown")
#add_setting("ENABLE_EXPORT", ENABLE_EXPORT, "Whether to enable export functionality")
#add_setting("ENABLE_LOCALSTORAGE", ENABLE_LOCALSTORAGE, "Whether to enable local storage")
#add_setting("ENABLE_CHARTS", ENABLE_CHARTS, "Whether to enable charts")
#add_setting("CHART_TOP_N", CHART_TOP_N, "Top N charts to display")
#add_setting("GRANTS_COLUMN", COL_GRANTS if COL_GRANTS else "", "Column for grants information")

settings_df = pd.DataFrame(settings_rows, columns=["Setting", "Explanation", "Value"])
settings_df = settings_df.set_index("Setting", drop=True)
def italic_formatter(val):
    return f"<i>{val}</i>" if val else ""

settings_html = settings_df.to_html(
    index=True,
    escape=False,  # Allow HTML tags
    formatters={"Explanation": italic_formatter}
)

# Per-author & per-affiliation tables (HTML)
per_author_html = per_author.to_html(index=False) if not per_author.empty else "<em>No author summary available.</em>"
per_affil_html = per_affil.to_html(index=False) if not per_affil.empty else "<em>No affiliation summary available.</em>"

# Top by citations (safe slice with existing columns)
def build_top_by_cites_html(df_all, df_top):
    if df_top.empty:
        return ""
    cols = existing(df_all, [COL_PAPER_TITLE, COL_AUTHOR, "affil_group", COL_CITES, "Google Link", "PubMed Link"])
    if not cols:
        return ""
    return (
        "<h2>Top by citations</h2><div class='table-wrap'>"
        + df_top[cols].to_html(index=False, escape=False)
        + "</div>"
    )

top_by_cites_html = build_top_by_cites_html(df, top_by_cites)

# Grants section HTML
def build_top_grants_html(top_df):
    if top_df.empty:
        return "<em>No grant data available.</em>"
    # Keep a small, clear table
    show = top_df.rename(columns={"grant": "Grant", "publications": "Publications"})
    return "<div class='table-wrap'>" + show.to_html(index=False, escape=True) + "</div>"

top_grants_html = build_top_grants_html(top_grants)

# ---- HTML assembly ----
now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

dynamic_status_css = build_status_css(DROPDOWN_OPTIONS)


styles = f"""
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; margin: 24px; color: #111; }}
h1,h2,h3 {{ margin: 0.6em 0 0.3em; }}
.summary {{ display: grid; grid-template-columns: repeat(2, minmax(220px, 1fr)); gap: 12px; margin: 12px 0 24px; }}
.card {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px 14px; background: #fafafa; }}
.small {{ color:#555; font-size: 14px; }}
.table-wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ border: 1px solid #e5e7eb; padding: 8px; vertical-align: top; }}
th {{ background: #f3f4f6; text-align: left; }}
th.sticky {{ position: sticky; top: 0; z-index: 1; }}
.controls {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin: 10px 0 16px; }}
input[type="search"] {{ padding: 8px 10px; border: 1px solid #e5e7eb; border-radius: 8px; width: 360px; }}
button.export {{ padding:8px 12px; border:1px solid #e5e7eb; border-radius:8px; background:#fff; cursor:pointer; }}
button.export:hover {{ background:#f9fafb; }}
.badge {{ display:inline-block; padding:2px 8px; border-radius:999px; background:#eef2ff; color:#3730a3; font-size:12px; }}
mark {{ background: #fff3a3; padding: 0 2px; }}
.explainer {{ border-left: 4px solid #6366f1; background:#f8fafc; padding:10px 12px; border-radius:8px; margin:12px 0; }}
.mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; font-size: 12px; color:#374151; }}
.footer {{ margin-top: 28px; color: #666; font-size: 12px; }}
/* Dropdown styling */
select.dd {{ padding: 6px 8px; border: 1px solid #e5e7eb; border-radius: 8px; background: #fff; }}
/* Settings block */
.settings-wrap {{ margin: 10px 0 22px; }}

/* Optional: neutral default for any non-blank (specific rules below will override) */
#results tbody tr:has(select.dd option:checked:not([value=""])) {{
  background: #f8fafc;
}}

/* === AUTO-GENERATED FROM DROPDOWN_OPTIONS === */
{dynamic_status_css}
</style>
"""


search_js = """
<script>
function filterTable() {
  const q = document.getElementById('q').value.toLowerCase();
  const trs = document.querySelectorAll('#results tbody tr');
  trs.forEach(tr => {
    const rowText = tr.innerText.toLowerCase();
    tr.style.display = rowText.includes(q) ? '' : 'none';
  });
}
</script>
"""

# Persist dropdown selections in localStorage (optional)
dropdown_js = """
<script>
(function(){
  const KEY = 'publication_report_dropdowns';
  const saved = JSON.parse(localStorage.getItem(KEY) || '{}');

  function save() { localStorage.setItem(KEY, JSON.stringify(saved)); }

  // Restore saved values on load
  window.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('select.dd').forEach(sel => {
      const row = sel.getAttribute('data-row');
      if (saved[row] !== undefined) sel.value = saved[row];
    });
  });

  // Save on change
  document.addEventListener('change', (e) => {
    if (!e.target.matches('select.dd')) return;
    const row = e.target.getAttribute('data-row');
    saved[row] = e.target.value;
    save();
  });
})();
</script>
"""

# Export CSV of the CURRENT table (including dropdown selections)
export_js = """
<script>
(function(){
  function csvEscape(value) {
    const v = String(value ?? '');
    if (/[",\\n]/.test(v)) return '"' + v.replace(/"/g, '""') + '"';
    return v;
  }

  function buildRowsFromTable(table) {
    const rows = [];
    const headerCells = Array.from(table.querySelectorAll('thead th'));
    const header = headerCells.map(th => th.innerText.trim());
    rows.push(header);

    const bodyRows = table.querySelectorAll('tbody tr');
    bodyRows.forEach(tr => {
      const cells = Array.from(tr.children).map((td) => {
        const sel = td.querySelector('select.dd');
        if (sel) return sel.value; // dropdown current value

        const a = td.querySelector('a');
        if (a) return a.href; // export links as URLs

        // strip any HTML like <mark> via textContent
        return td.textContent.trim();
      });
      rows.push(cells);
    });

    return rows;
  }

  function downloadCSV(filename, rows) {
    const csv = rows.map(r => r.map(csvEscape).join(',')).join('\\r\\n');
    const blob = new Blob(['\\uFEFF' + csv], { type: 'text/csv;charset=utf-8;' }); // BOM for Excel
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function exportTable() {
    const table = document.getElementById('results');
    if (!table) return;
    const rows = buildRowsFromTable(table);
    const date = new Date().toISOString().slice(0,10);
    const fnameBase = (document.title || 'publication_report').toLowerCase().replace(/\\s+/g,'_');
    downloadCSV(fnameBase + '_selections_' + date + '.csv', rows);
  }

  window.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('exportBtn');
    if (btn) btn.addEventListener('click', exportTable);
  });
})();
</script>
"""

# --- Explainer content (edit freely) ---
explainer_top = """
<div class="explainer">
  <strong>About this report</strong><br/>
  <div class="small">
    This report combines Europe PMC hits with PubMed records and enriches them using Google Scholar.
    • <em>Scholar snippet</em> shows context where matched terms appear. 
    • <em>Google Link</em> goes to the Scholar result; <em>PubMed Link</em> opens the PubMed record.
    • A CSV copy of the main table is saved alongside this HTML with blank <em>Claim</em> and <em>Notes</em> columns for annotation, plus a <em>{dcol}</em> column to mirror the dropdown.
    • Use the <em>Export CSV</em> button below to download the current table, including your dropdown selections.
  </div>
</div>
""".format(dcol=DROPDOWN_NAME)

explainer_affil = """
<div class="explainer">
  <strong>Affiliations</strong><br/>
  <div class="small">
    Affiliations were normalized by removing any leading numbers/spaces and keeping the text before the first comma.
    Example: <span class="mono">"12 Biological Physics Group, Department of Physics &amp; Astronomy, …"</span>
    becomes <span class="mono">"Biological Physics Group"</span>.
    Use this section to see output by group/unit.
  </div>
</div>
"""

# Controls block (search + optional export button)
controls_html = """
  <div class="controls">
    <input id="q" type="search" placeholder="Quick filter (title, author, terms, snippet…)" oninput="filterTable()">
    {export_btn}
  </div>
""".format(export_btn=('<button id="exportBtn" class="export" title="Download CSV of the current table (includes dropdown selections)">Export CSV</button>' if ENABLE_EXPORT else ""))

# Grants info block (small KPIs)
grants_cards_html = ""
if COL_GRANTS and not grants_counts.empty:
    grants_cards_html = f"""
    <div class="summary">
      <div class="card"><div>Papers with grants</div><h2>{papers_with_any_grant}</h2></div>
      <div class="card"><div>Unique grants</div><h2>{grants_counts['grant'].nunique()}</h2></div>
    </div>
    """

# Build HTML
now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
html = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8" />
<title>LiteratureReport</title>
{styles}
<body>
  <h1>PubMed × Google Scholar Report</h1>
  <div class="small">Generated: {now_str}</div>

  <h2>Search & pipeline settings</h2>
  <div class="settings-wrap table-wrap">
    {settings_html}
  </div>

  {explainer_top}

  <div class="summary">
    <div class="card"><div>Total rows</div><h2>{total_rows}</h2></div>
    <div class="card"><div>Scholar title matches</div><h2>{matched_rows}</h2></div>
    <div class="card"><div>Rows with terms matched</div><h2>{terms_matched_rows}</h2></div>
    <div class="card"><div>Total citations (matched)</div><h2>{total_citations}</h2></div>
    {"<div class='card'><div>Unique authors</div><h2>"+str(unique_authors)+"</h2></div>" if unique_authors is not None else ""}
    <div class="card"><div>Unique affiliations</div><h2>{unique_affils}</h2></div>
  </div>

  {"<h2>Top authors by matched titles</h2><img src='"+top_auth_chart_uri+"' alt='Top authors chart' />" if top_auth_chart_uri else ""}

  {"<h2>Top affiliations by matched titles</h2><img src='"+top_affil_chart_uri+"' alt='Top affiliations chart' />" if top_affil_chart_uri else ""}

  {"<h2>Publications over time</h2><img src='"+timeline_chart_uri+"' alt='Timeline chart' />" if timeline_chart_uri else ""}

  <h2>Grants</h2>
  {grants_cards_html if grants_cards_html else "<div class='small'>No grant metadata detected in this dataset.</div>"}
  {"<img src='"+top_grants_chart_uri+"' alt='Top grants chart' />" if top_grants_chart_uri else ""}
  {top_grants_html}

  <h2>Per-author summary</h2>
  <div class="table-wrap">{per_author_html}</div>

  <h2>Per-affiliation summary</h2>
  {explainer_affil}
  <div class="table-wrap">{per_affil_html}</div>

  {top_by_cites_html}

  {"<h2>Recent papers</h2><div class='table-wrap'>"+recent[[COL_PAPER_TITLE, COL_AUTHOR, "affil_group", COL_DATE]].to_html(index=False, escape=False)+"</div>" if not recent.empty and COL_DATE else ""}

  <h2>All results</h2>
  {controls_html}
  <div class="table-wrap">
    {table_html.replace("<table", "<table id='results'").replace("<th", "<th class='sticky'")}
  </div>

  <div class="explainer">
    <strong>Editing notes</strong>
    <div class="small">
      • The CSV saved alongside this report includes raw URLs and blank <em>Claim</em>/<em>Notes</em>/<em>{DROPDOWN_NAME}</em> fields for annotation in Excel/Sheets.<br/>
      • The <em>Export CSV</em> button downloads the live table (links exported as URLs, snippets as plain text, and your current <em>{DROPDOWN_NAME}</em> selections).
    </div>
  </div>

  <div class="footer">Report generated locally — single self-contained HTML + CSV.</div>

  {search_js}
  {dropdown_js if ENABLE_LOCALSTORAGE else ""}
  {export_js if ENABLE_EXPORT else ""}
</body>
</html>
"""

Path(OUTPUT_HTML).write_text(html, encoding="utf-8")
print(f"✅ Report written to {OUTPUT_HTML}")
