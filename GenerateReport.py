#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import io
import re
import sys
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import matplotlib.pyplot as plt

DATE_STR = date.today().isoformat()

# --------- configuration ----------
INPUT_CSV = sys.argv[1] if len(sys.argv) > 1 else f"pubmed_googlescholar_{DATE_STR}.csv"
OUTPUT_HTML = sys.argv[2] if len(sys.argv) > 2 else f"publication_report_{DATE_STR}.html"

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
# ----------------------------------

# ---------- helpers ----------
def read_head_cols(path):
    return pd.read_csv(path, nrows=1).columns.tolist()

cols_available = read_head_cols(INPUT_CSV)
def col_exists(c): return c in cols_available

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
    fig = plt.figure()
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

# ---------- load & prepare data ----------
df = pd.read_csv(INPUT_CSV)

# basic coercions
if COL_CITES and COL_CITES in df.columns:
    df[COL_CITES] = pd.to_numeric(df[COL_CITES], errors="coerce").fillna(0).astype(int)

if COL_DATE and COL_DATE in df.columns:
    df["__pubdate"] = df[COL_DATE].apply(parse_date_maybe)
else:
    df["__pubdate"] = pd.NaT

# derive affiliation group early
df["affil_group"] = df.apply(derive_affil_group, axis=1)

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

# metrics
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

# charts
if not per_author.empty:
    top_auth_chart_uri = make_chart_uri(lambda: (
        plt.bar(per_author.head(10)[COL_AUTHOR], per_author.head(10)["matched"]),
        plt.xticks(rotation=45, ha="right"),
        plt.title("Top authors by matched Scholar titles"),
        plt.ylabel("Matched papers"),
        plt.xlabel("Author"),
    ))
else:
    top_auth_chart_uri = ""

if not per_affil.empty:
    top_affil_chart_uri = make_chart_uri(lambda: (
        plt.bar(per_affil.head(10)["affil_group"], per_affil.head(10)["matched"]),
        plt.xticks(rotation=45, ha="right"),
        plt.title("Top affiliations by matched Scholar titles"),
        plt.ylabel("Matched papers"),
        plt.xlabel("Affiliation group"),
    ))
else:
    top_affil_chart_uri = ""

if not timeline.empty:
    timeline_chart_uri = make_chart_uri(lambda: (
        plt.plot(timeline["month"], timeline["count"]),
        plt.title("Publications over time"),
        plt.ylabel("Count"),
        plt.xlabel("Month"),
    ))
else:
    timeline_chart_uri = ""

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

# Escape non-HTML fields for the HTML report
for col in table_df_html.columns:
    if col in ("Google Link", "PubMed Link", "scholar_snippet"):
        continue
    table_df_html[col] = table_df_html[col].map(esc)

table_html = table_df_html.to_html(index=True, escape=False)

# ------------------------------
# Build CSV for user annotation
# ------------------------------
# Use plain snippet (no <mark>), and include raw URLs + add Claim/Notes columns
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

csv_output = Path(OUTPUT_HTML).with_suffix(".csv")
csv_df.to_csv(csv_output, index=True)
print(f"📝 Base table saved for annotation → {csv_output}")

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

# ---- HTML assembly ----
now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

styles = """
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; margin: 24px; color: #111; }
h1,h2,h3 { margin: 0.6em 0 0.3em; }
.summary { display: grid; grid-template-columns: repeat(2, minmax(220px, 1fr)); gap: 12px; margin: 12px 0 24px; }
.card { border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px 14px; background: #fafafa; }
.small { color:#555; font-size: 12px; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { border: 1px solid #e5e7eb; padding: 8px; vertical-align: top; }
th { background: #f3f4f6; text-align: left; }
th.sticky { position: sticky; top: 0; z-index: 1; }
.search { margin: 10px 0 16px; }
input[type="search"] { padding: 8px 10px; border: 1px solid #e5e7eb; border-radius: 8px; width: 360px; }
.badge { display:inline-block; padding:2px 8px; border-radius:999px; background:#eef2ff; color:#3730a3; font-size:12px; }
mark { background: #fff3a3; padding: 0 2px; }
.explainer { border-left: 4px solid #6366f1; background:#f8fafc; padding:10px 12px; border-radius:8px; margin:12px 0; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; font-size: 12px; color:#374151; }
.footer { margin-top: 28px; color: #666; font-size: 12px; }
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

# --- Explainer content (edit freely) ---
explainer_top = """
<div class="explainer">
  <strong>About this report</strong><br/>
  <div class="small">
    <!-- EDIT ME: add context for your audience -->
    This report combines Europe PMC hits with PubMed records and enriches them using Google Scholar.
    • <em>Scholar snippet</em> shows context where matched terms appear. 
    • <em>Google Link</em> goes to the Scholar result; <em>PubMed Link</em> opens the PubMed record.
    • A CSV copy of the main table is saved alongside this HTML with blank <em>Claim</em> and <em>Notes</em> columns for annotation.
  </div>
</div>
"""

explainer_affil = """
<div class="explainer">
  <strong>Affiliations</strong><br/>
  <div class="small">
    <!-- EDIT ME: affiliation methodology -->
    Affiliations were normalized by removing any leading numbers/spaces and keeping the text before the first comma.
    Example: <span class="mono">"12 Biological Physics Group, Department of Physics &amp; Astronomy, …"</span>
    becomes <span class="mono">"Biological Physics Group"</span>.
    Use this section to see output by group/unit.
  </div>
</div>
"""

# Build HTML
html = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8" />
<title>PubMed × Scholar Report</title>
{styles}
<body>
  <h1>PubMed × Google Scholar Report</h1>
  <div class="small">Generated: {now_str} — Source: {Path(INPUT_CSV).name}</div>

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

  <h2>Per-author summary</h2>
  <div class="table-wrap">{per_author_html}</div>

  <h2>Per-affiliation summary</h2>
  {explainer_affil}
  <div class="table-wrap">{per_affil_html}</div>

  {top_by_cites_html}

  {"<h2>Recent papers</h2><div class='table-wrap'>"+recent[[COL_PAPER_TITLE, COL_AUTHOR, "affil_group", COL_DATE]].to_html(index=False, escape=False)+"</div>" if not recent.empty and COL_DATE else ""}

  <h2>All results</h2>
  <div class="search"><input id="q" type="search" placeholder="Quick filter (title, author, terms, snippet…)" oninput="filterTable()"></div>
  <div class="table-wrap">
    {table_html.replace("<table", "<table id='results'").replace("<th", "<th class='sticky'")}
  </div>

  <div class="explainer">
    <strong>Editing notes</strong>
    <div class="small">
      <!-- EDIT ME: any distribution notes -->
      • The CSV saved alongside this report includes raw URLs and blank <em>Claim</em>/<em>Notes</em> fields for annotation in Excel/Sheets.<br/>
      • For column sorting beyond the HTML search box, open the CSV in Excel/Sheets.
    </div>
  </div>

  <div class="footer">Report generated locally — single self‑contained HTML + CSV.</div>

  {search_js}
</body>
</html>
"""

Path(OUTPUT_HTML).write_text(html, encoding="utf-8")
print(f"✅ Report written to {OUTPUT_HTML}")
