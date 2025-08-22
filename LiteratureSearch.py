#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Europe PMC → PubMed → Google Scholar (author+terms) matcher

Requirements:
    pip install requests pandas playwright python-dotenv serpapi
    playwright install chromium

Environment:
    SERPAPI_KEY in your .env (for Scholar step)
"""

import os
import re
import time
from datetime import date
import unicodedata
import asyncio
import requests
import subprocess
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from playwright.async_api import async_playwright

# Retrieving settings from .env file

def get_env_bool(key, default=False):
    val = os.getenv(key, str(default))
    return val.lower() in ("true", "1", "yes")

def get_env_list(key, sep=","):
    val = os.getenv(key, "")
    return [x.strip() for x in val.split(sep) if x.strip()]

load_dotenv()

# ==================================
# Part 1: Fetch PMIDs from Europe PMC
# ==================================

def fetch_epmc_results_for_authors(authors, page_size=50, delay_seconds=1.0, date_filter=None, save=None):
    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    all_results = []
    summary_data = []

    for author in authors:
        # Build query with date filter if provided
        query = f'AUTHOR:"{author}"'
        if date_filter:
            if isinstance(date_filter, str):
                query += f' AND FIRST_PDATE:[{date_filter} TO *]'
            elif isinstance(date_filter, tuple) and len(date_filter) == 2:
                lower, upper = date_filter
                query += f' AND FIRST_PDATE:[{lower} TO {upper}]'

        print(f"🔍 Searching Europe PMC for: {author} | Query: {query}")

        params = {"query": query, "format": "json", "pageSize": page_size}

        try:
            response = requests.get(base_url, params=params, timeout=20)
            response.raise_for_status()
            articles = response.json().get("resultList", {}).get("result", [])

            # Always add search_author to each article
            for article in articles:
                article["search_author"] = author

            all_results.extend(articles)
            summary_data.append({"search_author": author, "articles_found": len(articles)})

        except Exception as e:
            print(f"❌ Error searching for '{author}': {e}")
            summary_data.append({"search_author": author, "articles_found": 0})

        time.sleep(delay_seconds)

    # Ensure search_author is always present, even if no articles
    if all_results:
        df = pd.DataFrame(all_results)
    else:
        # Create an empty DataFrame with search_author column
        df = pd.DataFrame(columns=["search_author"])

    df_counts = pd.DataFrame(summary_data).set_index("search_author")

    if not df.empty and "pmid" in df.columns:
        df = df.drop_duplicates(subset="pmid").set_index("pmid")
        df = df[df.index.notnull()]

        # No need for post-filtering by date, already done in query!
        if save:
            Path(save).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save)
    else:
        print("⚠️ No PMIDs found in the results.")

    return df, df_counts


# ==================================
# Part 2: Scrape PubMed Data
# ==================================

async def scrape_pubmed_article(pmid, page):
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    data = {"pmid": pmid, "pubmed_url": url}
    print(f"🔎 Scraping: {url}")

    try:
        await page.goto(url, timeout=8000)
        page.set_default_timeout(5000)

        title_el = page.locator("h1.heading-title").first
        data["title"] = (await title_el.text_content() or "").strip() if await title_el.is_visible() else ""

        abstract_box = page.locator("div.abstract-content")
        if await abstract_box.count() > 0:
            try:
                paragraphs = await abstract_box.locator("p").all_inner_texts()
                data["abstract"] = "\n".join(paragraphs).strip()
            except:
                data["abstract"] = ""
        else:
            data["abstract"] = ""

        try:
            authors = await page.locator("div.authors-list span.authors-list-item a.full-name").all_inner_texts()
            data["authors"] = "; ".join(authors)
        except:
            data["authors"] = ""

        try:
            affs = await page.locator("div.affiliations li").all_inner_texts()
            data["affiliations"] = "; ".join(affs)
        except:
            data["affiliations"] = ""

        try:
            kw_el = page.locator("div.keywords-section li")
            data["keywords"] = "; ".join(await kw_el.all_inner_texts()) if await kw_el.count() > 0 else ""
        except:
            data["keywords"] = ""

        try:
            citation_box = page.locator("div.article-citation")
            journal_btn = citation_box.locator("button.journal-actions-trigger").first
            data["journal_abbrev"] = (await journal_btn.text_content(timeout=2000) or "").strip()
            data["journal_fullname"] = await journal_btn.get_attribute("title", timeout=2000) or ""
            data["journal_citation"] = (await citation_box.locator("span.cit").first.text_content(timeout=2000) or "").strip()
            doi_text = await citation_box.locator("span.citation-doi").first.text_content(timeout=2000)
            data["doi"] = doi_text.replace("doi:", "").strip() if doi_text else ""
            epub = await citation_box.locator("span.secondary-date").first.text_content(timeout=2000)
            data["epub_date"] = epub.replace("Epub", "").strip() if epub else ""
        except:
            pass

        try:
            grant_block = page.locator("#grants")
            if await grant_block.count() > 0:
                show_all_btn = grant_block.locator("button.show-all")
                try:
                    if await show_all_btn.is_visible(timeout=2000):
                        await show_all_btn.click()
                        await page.wait_for_timeout(1000)
                except:
                    pass
                grants = await grant_block.locator(".grant-item a").all_inner_texts()
                data["grants"] = "; ".join(grants)
            else:
                data["grants"] = ""
        except:
            data["grants"] = ""

        return data
    except Exception as e:
        print(f"❌ Error scraping {pmid}: {e}")
        return None


async def fetch_and_scrape_all(
    authors_csv,
    page_size=8,
    output_csv="pubmed_combined.csv",
    counts_csv="author_counts.csv",
    date_filter=None,
    affiliation_filters=('Manchester',),
):
    authors_df = pd.read_csv(authors_csv)
    authors_df['author'] = authors_df['First Name'].astype(str) + " " + authors_df['Surname'].astype(str)
    authors = authors_df['author'].tolist()
    email_dict = authors_df.set_index('author')['e-mail'].to_dict()

    df_epmc, df_counts = fetch_epmc_results_for_authors(authors, page_size=page_size, delay_seconds=0.5, date_filter=date_filter)
    pmids = df_epmc.index.tolist() if not df_epmc.empty else []

    Path(counts_csv).parent.mkdir(parents=True, exist_ok=True)
    df_counts.to_csv(counts_csv)

    if not pmids:
        print("❌ No PMIDs to scrape.")
        return pd.DataFrame(), df_counts

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"),
            locale="en-US",
            viewport={"width": 1280, "height": 800},
        )
        page = await context.new_page()

        scrape_data = []
        for pmid in pmids:
            result = await scrape_pubmed_article(pmid, page)
            if result:
                scrape_data.append(result)

        await browser.close()

    df_scraped = pd.DataFrame(scrape_data).set_index("pmid") if scrape_data else pd.DataFrame(columns=["pmid"]).set_index("pmid")
    df_combined = df_epmc.join(df_scraped, how="left", lsuffix="_epmc", rsuffix="_pubmed")

    def match_author_affiliation(row, affiliation_filters):
        
        search_author = row.get("search_author", "")
        authors = str(row.get("authors", "")).split(";")
        affiliations = str(row.get("affiliations", "")).split(";")

        search_last = search_author.strip().split()[-1].lower() if search_author else ""
        matched_affil = ""
        match = False

        for i, author in enumerate(authors):
            author_norm = author.strip().lower()
            if search_last and search_last in author_norm:
                if i < len(affiliations):
                    matched_affil = affiliations[i].strip()
                    match = any(keyword.lower() in matched_affil.lower() for keyword in affiliation_filters)
                    break

        return pd.Series({"matched_affiliation": matched_affil, "affiliation_match": match})

    if not df_combined.empty:
        df_combined[["matched_affiliation", "affiliation_match"]] = df_combined.apply(
            lambda row: match_author_affiliation(row, affiliation_filters), axis=1
        )

    if not df_combined.empty:
        df_combined['author_email'] = df_combined['search_author'].map(email_dict).values

    df_combined["retrieved"] = date.today().isoformat()
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(output_csv)

    if not df_combined.empty and "affiliation_match" in df_combined.columns:
        affiliation_counts = (
            df_combined[df_combined["affiliation_match"] == True]
            .groupby("search_author")
            .size()
            .rename("affiliations_matched")
        )
        df_counts = df_counts.join(affiliation_counts, how="left").fillna(0)
        df_counts["affiliations_matched"] = df_counts["affiliations_matched"].astype(int)
        df_counts.to_csv(counts_csv)

    if "search_author" not in df_combined.columns:
        df_combined = df_combined.reset_index()

    return df_combined, df_counts


# ==================================
# Part 3: Google Scholar via SerpAPI (search_author + terms, title match)
# ==================================

# ---- Title normalization & compare (your method) ----
def normalize_hyphens(s: str) -> str:
    out_chars = []
    for ch in s:
        cat = unicodedata.category(ch)
        if cat == 'Pd' or ch in {'\u00AD', '\u2011', '\u2212', '\u207B', '\u2E17'}:
            out_chars.append('-')
        else:
            out_chars.append(ch)
    return ''.join(out_chars)

def normalize_string(s: str) -> str:
    s = unicodedata.normalize('NFKC', s)
    s = normalize_hyphens(s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = s.casefold()
    return s

def compare_strings(s1: str, s2: str, N: int) -> bool:
    n1 = normalize_string(s1 or "")
    n2 = normalize_string(s2 or "")
    # print(f"Normalized String 1: '{n1}'")
    # print(f"Normalized String 2: '{n2}'")
    return n1[:N] == n2[:N]

# ---- Optional pre-filter like before (can be skipped if you want everything) ----
def filter_eligible_articles(
    df,
    #oa_column="isOpenAccess",
    pubtype_column="pubType",
    affiliation_column="affiliation_match",
    exclude_types=None
):
    if exclude_types is None:
        exclude_types = ["letter", "review", "editorial"]
    df = df.copy()
    #if oa_column in df.columns:
    #    df = df[df[oa_column] == "Y"]
    if affiliation_column in df.columns:
        df = df[df[affiliation_column] == True]

    def valid_pubtype(pubtype):
        if not isinstance(pubtype, str):
            return True  # keep if we don't know
        return not any(bad in pubtype.lower() for bad in exclude_types)

    if pubtype_column in df.columns:
        df = df[df[pubtype_column].apply(valid_pubtype)]
    return df

def extract_terms_in_snippet(snippet: str, terms: list[str]) -> list[str]:
    text = (snippet or "").lower()
    found = []
    for t in terms:
        t_norm = t.lower()
        if t_norm in text:
            found.append(t)
    return found

def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def scholar_search_by_author_and_terms(
    df: pd.DataFrame,
    terms: list[str],
    serpapi_key: str,
    sleep: float = 2.0,
    N_prefix: int = 25,
    as_ylo_year: int | None = None,
    sort_author_terms_by_date: bool = True,
    enable_title_search: bool = False  # NEW
) -> pd.DataFrame:
    """
    For each search_author:
      1) Query Scholar with: author AND (t1 OR t2 ...) [date sort optional, default on]
      2) Try to match each PubMed title by normalized first N chars
      3) If NOT matched, fallback: author-only query sorted by RELEVANCE (no scisbd), still with as_ylo if set
      4) Populate snippet, terms-in-snippet, link, citations, and provenance flags
    """
    from serpapi import GoogleSearch

    if not terms:
        raise ValueError("terms must be provided.")
    if not serpapi_key:
        raise ValueError("serpapi_key must be provided.")

    def extract_terms_in_snippet(snippet: str, terms_list: list[str]) -> list[str]:
        text = (snippet or "").lower()
        return [t for t in terms_list if t.lower() in text]

    def safe_get(d, *keys, default=None):
        cur = d
        for k in keys:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        return cur

    def run_scholar(params: dict) -> dict:
        try:
            search = GoogleSearch(params)
            return search.get_dict() or {}
        except Exception as e:
            print(f"⚠️ Scholar error for query '{params.get('q', '')[:80]}...': {e}")
            return {}

    # Ensure required output columns exist
    out_cols = [
        "scholar_title_match_found",
        "scholar_match_title",
        "scholar_match_url",
        "scholar_match_citations",
        "scholar_snippet",
        "scholar_terms_found",
        "scholar_terms_matched",
        "scholar_matched_via",  # "author+terms" | "author_only" | "no_match"
    ]
    df = df.copy()
    for c in out_cols:
        if c not in df.columns:
            if c in ("scholar_title_match_found", "scholar_terms_matched"):
                df[c] = False
            else:
                df[c] = ""

    pm_title_col = "title_pubmed"  # change to your actual title column if needed

    for author, group_idx in df.groupby("search_author").groups.items():
        if not isinstance(author, str) or not author.strip():
            continue

        # ----- Build queries -----
        quoted_terms = [f'"{t}"' if " " in t else t for t in terms]
        query_terms = f'"{author}" AND (' + " OR ".join(quoted_terms) + ")"
        query_author_only = f'"{author}"'

        # Common base for both queries
        base = {
            "engine": "google_scholar",
            "api_key": serpapi_key,
        }
        if as_ylo_year:
            base["as_ylo"] = as_ylo_year

        # 1) Author + terms — keep date sort if requested
        params_terms = dict(base)
        params_terms["q"] = query_terms
        if sort_author_terms_by_date:
            params_terms["scisbd"] = "1"  # sort by date
        # else: leave off scisbd to sort by relevance

        res_terms = run_scholar(params_terms)
        org_terms = res_terms.get("organic_results", []) or []
        norm_titles_terms = [(r, normalize_string(r.get("title", "") or "")) for r in org_terms]

        # For each PubMed title, attempt matching; fallback to author-only (relevance)
        for idx in list(group_idx):
            
            # Skip if already matched (e.g. we have changed Google API keys half way through)
            if df.at[idx, "scholar_title_match_found"]:
                continue
            
            pm_title = df.at[idx, pm_title_col] if pm_title_col in df.columns else ""
            if not isinstance(pm_title, str) or not pm_title.strip():
                df.at[idx, "scholar_matched_via"] = "no_match"
                continue

            pm_norm = normalize_string(pm_title)

            # --- Try match in author+terms results ---
            candidates = []
            for r, r_norm in norm_titles_terms:
                if compare_strings(pm_norm, r_norm, N_prefix):
                    cites = safe_get(r, "inline_links", "cited_by", "total", default=0) or 0
                    candidates.append((cites, r))

            matched = None
            via = None

            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                matched = candidates[0][1]
                via = "author+terms"
            else:
                # --- Fallback: author-only, sorted by RELEVANCE (i.e., NO scisbd) ---
                params_author_only = dict(base)
                params_author_only["q"] = query_author_only
                # intentionally DO NOT set scisbd here → relevance sorting
                res_author = run_scholar(params_author_only)
                org_author = res_author.get("organic_results", []) or []
                norm_titles_author = [(r, normalize_string(r.get("title", "") or "")) for r in org_author]

                candidates2 = []
                for r, r_norm in norm_titles_author:
                    if compare_strings(pm_norm, r_norm, N_prefix):
                        cites = safe_get(r, "inline_links", "cited_by", "total", default=0) or 0
                        candidates2.append((cites, r))

                if candidates2:
                    candidates2.sort(key=lambda x: x[0], reverse=True)
                    matched = candidates2[0][1]
                    via = "author_only"

            if matched:
                snippet = matched.get("snippet", "") or ""
                terms_found = extract_terms_in_snippet(snippet, terms)

                df.at[idx, "scholar_title_match_found"] = True
                df.at[idx, "scholar_match_title"] = matched.get("title", "") or ""
                df.at[idx, "scholar_match_url"] = matched.get("link", "") or ""
                df.at[idx, "scholar_match_citations"] = safe_get(matched, "inline_links", "cited_by", "total", default="") or ""
                df.at[idx, "scholar_snippet"] = snippet
                df.at[idx, "scholar_terms_found"] = "; ".join(terms_found)
                df.at[idx, "scholar_terms_matched"] = len(terms_found) > 0
                df.at[idx, "scholar_matched_via"] = via
            else:
                # --- Optional: Title-only search ---
                if enable_title_search and isinstance(pm_title, str) and pm_title.strip():
                    params_title_only = dict(base)
                    params_title_only["q"] = f'"{pm_title}"'
                    res_title = run_scholar(params_title_only)
                    org_title = res_title.get("organic_results", []) or []
                    norm_titles_title = [(r, normalize_string(r.get("title", "") or "")) for r in org_title]

                    candidates3 = []
                    for r, r_norm in norm_titles_title:
                        if compare_strings(pm_norm, r_norm, N_prefix):
                            cites = safe_get(r, "inline_links", "cited_by", "total", default=0) or 0
                            candidates3.append((cites, r))

                    if candidates3:
                        candidates3.sort(key=lambda x: x[0], reverse=True)
                        matched = candidates3[0][1]
                        via = "title_only"

                if matched:
                    snippet = matched.get("snippet", "") or ""
                    terms_found = extract_terms_in_snippet(snippet, terms)

                    df.at[idx, "scholar_title_match_found"] = True
                    df.at[idx, "scholar_match_title"] = matched.get("title", "") or ""
                    df.at[idx, "scholar_match_url"] = matched.get("link", "") or ""
                    df.at[idx, "scholar_match_citations"] = safe_get(matched, "inline_links", "cited_by", "total", default="") or ""
                    df.at[idx, "scholar_snippet"] = snippet
                    df.at[idx, "scholar_terms_found"] = "; ".join(terms_found)
                    df.at[idx, "scholar_terms_matched"] = len(terms_found) > 0
                    df.at[idx, "scholar_matched_via"] = via
                else:
                    df.at[idx, "scholar_title_match_found"] = False
                    df.at[idx, "scholar_match_title"] = ""
                    df.at[idx, "scholar_match_url"] = ""
                    df.at[idx, "scholar_match_citations"] = ""
                    df.at[idx, "scholar_snippet"] = ""
                    df.at[idx, "scholar_terms_found"] = ""
                    df.at[idx, "scholar_terms_matched"] = False
                    df.at[idx, "scholar_matched_via"] = "no_match"

        time.sleep(sleep)  # be respectful per author

    return df



# ==================================
# Orchestrator
# ==================================

async def run_pipeline(
    authors_csv: str = None,
    epmc_page_size: int = 8,
    pubmed_output_csv: str = "pubmed_combined.csv",
    counts_csv: str = "author_counts.csv",
    affiliation_filters=('Manchester',),

    # Scholar step (author+terms)
    run_scholar_check: bool = True,
    scholar_terms: tuple[str, ...] = ("hyperion", "imc", "imaging mass cytometry", "cytof"),
    scholar_output_csv: str = "pubmed_combined_with_scholar_check.csv",
    serpapi_key: str | None = None,
    scholar_sleep: int = 2,
    scholar_prefix_N: int = 25,
    date_filter: str | tuple = None,
    apply_filter: bool = False,
    pubmed_input_csv: str = None,  # NEW: path to load existing pubmed output
    enable_title_search: bool = False
):
    # 1) EPMC + PubMed scrape or load from disk
    if pubmed_input_csv and Path(pubmed_input_csv).exists():
        print(f"📂 Loading PubMed data from {pubmed_input_csv}")
        df_combined = pd.read_csv(pubmed_input_csv, index_col="pmid")
    else:
        if not authors_csv:
            raise ValueError("authors_csv must be provided if not loading from disk.")
        df_combined, _ = await fetch_and_scrape_all(
            authors_csv=authors_csv,
            page_size=epmc_page_size,
            output_csv=pubmed_output_csv,
            counts_csv=counts_csv,
            date_filter=date_filter,
            affiliation_filters=affiliation_filters,
        )

    # Extract year from DATE_FILTER
    as_ylo_year = None
    if date_filter:
        if isinstance(date_filter, tuple) and len(date_filter) > 0:
            as_ylo_year = int(str(date_filter[0])[:4])
        elif isinstance(date_filter, str):
            as_ylo_year = int(date_filter[:4])

    # 2) Google Scholar check (after the above)
    if run_scholar_check:
        
        serpapi_key = serpapi_key or os.getenv("SERPAPI_KEY")
        if not serpapi_key:
            raise RuntimeError("SERPAPI_KEY not provided and not found in environment.")

        if df_combined.empty:
            print("⚠️ No combined PubMed data to check on Scholar.")
            return df_combined

        df_target = df_combined
        if apply_filter:
            df_target = filter_eligible_articles(df_target)

        print("🔎 Running Google Scholar (author + terms) and matching titles…")
        try:
            df_checked = scholar_search_by_author_and_terms(
                df=df_target.reset_index(),
                terms=list(scholar_terms),
                serpapi_key=serpapi_key,
                sleep=scholar_sleep,
                N_prefix=scholar_prefix_N,
                as_ylo_year=as_ylo_year,
                enable_title_search=enable_title_search
            )
        except Exception as e:
            print(f"❌ Error during Scholar search: {e}")
            # Save partial results
            partial_path = scholar_output_csv.replace(".csv", "_partial.csv")
            df_target.to_csv(partial_path, index=False)
            print(f"⚠️ Partial results saved to {partial_path}")
            raise  # Optionally re-raise or return partial DataFrame

        Path(scholar_output_csv).parent.mkdir(parents=True, exist_ok=True)
        df_checked.to_csv(scholar_output_csv, index=False)
        print(f"✅ Scholar author+terms match saved → {scholar_output_csv}")
        return df_checked

    return df_combined


# ==================================
# Main
# ==================================

if __name__ == "__main__":
    DATE_STR = date.today().isoformat()
    
    AUTHORS_CSV = os.getenv("AUTHORS_CSV", "authors.csv")
    PUBMED_OUTPUT = os.getenv("PUBMED_OUTPUT", f"pubmed_unfiltered_{DATE_STR}.csv")
    COUNTS_CSV = os.getenv("COUNTS_CSV", f"author_counts_{DATE_STR}.csv")
    DATE_FILTER_START = os.getenv("DATE_FILTER_START", None)
    DATE_FILTER_END = os.getenv("DATE_FILTER_END", None)
    
    # Logic to set DATE_FILTER
    if not DATE_FILTER_START or DATE_FILTER_START.lower() == "none":
        DATE_FILTER_START = None
    if not DATE_FILTER_END or DATE_FILTER_END.lower() == "none":
        DATE_FILTER_END = None
    if DATE_FILTER_START is None and DATE_FILTER_END is None:
        DATE_FILTER = None
    else:
        DATE_FILTER = (DATE_FILTER_START, DATE_FILTER_END)
    
    affil_raw = os.getenv("AFFILIATION_FILTERS", "Manchester")
    AFFILIATION_FILTERS = tuple([x.strip() for x in affil_raw.split(",") if x.strip()])
    PUBMED_INPUT_CSV = os.getenv("PUBMED_INPUT_CSV", None)
    RUN_SCHOLAR = get_env_bool("RUN_SCHOLAR", True)
    SCHOLAR_TERMS = tuple(get_env_list("SCHOLAR_TERMS"))
    SCHOLAR_OUTPUT = os.getenv("SCHOLAR_OUTPUT", f"pubmed_googlescholar_{DATE_STR}.csv")
    SCHOLAR_SLEEP = int(os.getenv("SCHOLAR_SLEEP", "2"))
    SCHOLAR_PREFIX_N = int(os.getenv("SCHOLAR_PREFIX_N", "25"))
    APPLY_FILTER = get_env_bool("APPLY_FILTER", True)
    ENABLE_TITLE_SEARCH = get_env_bool("ENABLE_TITLE_SEARCH", True)
    SERPAPI_KEY = os.getenv("SERPAPI_KEY", None)

    result_df = asyncio.run(
        run_pipeline(
            authors_csv=AUTHORS_CSV,
            epmc_page_size=8,
            pubmed_output_csv=PUBMED_OUTPUT,
            counts_csv=COUNTS_CSV,
            date_filter=DATE_FILTER,
            affiliation_filters=AFFILIATION_FILTERS,
            run_scholar_check=RUN_SCHOLAR,
            scholar_terms=SCHOLAR_TERMS,
            scholar_output_csv=SCHOLAR_OUTPUT,
            serpapi_key=SERPAPI_KEY,
            scholar_sleep=SCHOLAR_SLEEP,
            scholar_prefix_N=SCHOLAR_PREFIX_N,
            apply_filter=APPLY_FILTER,
            pubmed_input_csv=PUBMED_INPUT_CSV,
            enable_title_search=ENABLE_TITLE_SEARCH
        )
    )
    print("🎉 Pipeline complete.")

    report_script = "GenerateReport.py"
    output_csv = SCHOLAR_OUTPUT
    output_html = output_csv.replace(".csv", ".html")

    print(f"📝 Generating report: {output_html}")
    subprocess.run([
        "python", report_script, output_csv, output_html
    ], check=True)