# =============================================================================
# ingestion/pdf_parser.py — PDF text + table extraction (pdfplumber)
#
# WHY THIS FILE EXISTS:
# Turns a factsheet PDF into plain text, page by page. This is the first stage
# of the pipeline: parse → chunk → embed → store. Graduated from the notebook
# cells `table_to_text()` and `extract_page_content()`.
# =============================================================================


# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
# You need:
#   - `pdfplumber`         → open the PDF, extract text and tables
#
# Write your imports here:
import pdfplumber


# -----------------------------------------------------------------------------
# table_to_text(table: list[list]) -> str
#
# WHY: A pdfplumber table is a list of rows (each row a list of cells). The
# embedding model wants flat text, so we render each row as a pipe-separated
# line. Graduated directly from your notebook.
#
# STEPS:
#   1. Guard: if table is empty/None, return ""
#   2. For each row, join cells with " | " (treat None cells as "")
#   3. Join all rows with newlines, return the string
#
# Write table_to_text() here:
def table_to_text(table: list[list]) -> str:
    if not table:
        return ""
    lines = []
    for row in table:
       lines.append(" | ".join(cell or "" for cell in row))

    return "\n".join(lines)

# -----------------------------------------------------------------------------
# extract_page_content(page) -> str
#
# WHY: Combine a page's plain text and its tables into one string. This is the
# per-page unit the chunker will later slice into chunks.
#
# STEPS:
#   1. plain_text = page.extract_text() or ""      ← guard against None
#   2. tables = page.extract_tables()
#   3. For each table, convert via table_to_text() and collect
#   4. Return plain_text + "\n" + "\n".join(table_texts)
#
# Write extract_page_content() here:

def extract_page_content(page) -> str:
    # Step 1: extract plain text
    plain_text = page.extract_text() or ""
    
    # Step 2: extract tables and convert each to text using table_to_text()
    tables = page.extract_tables()
    
    table_text = []
    for table in tables:
        table_text.append(table_to_text(table))
    
    # Step 3: combine both — plain text first, then table text below it
    combined_text = plain_text + '\n' + '\n'.join(table_text)
    
    # Step 4: return the combined string
    return combined_text


# -----------------------------------------------------------------------------
# detect_scheme_name(page) -> str | None
#
# WHY: A consolidated factsheet PDF (e.g. KotakMF) covers MANY schemes — Large
# Cap, Mid Cap, Liquid, etc. — usually one scheme per page, led by a header line
# with the scheme name. `fund_name` must be the SCHEME (what users type), not the
# fund house, or `has_fund()` / the FundNotFound gate can't match. This isolates
# "which scheme is this page about?" into one testable function.
#
# Returns None when no header is found (TOC, disclaimer, table-only pages) — the
# pipeline decides the fallback, not the parser.
#
# WHY NOT "first non-empty line": extract_text() orders text by vertical position,
# so callouts that sit physically higher than the title (e.g. "Investment style",
# "Scan to Invest Now") come out first and get mistaken for the scheme name. The
# real signal the eye uses is SIZE — the scheme title is the visually dominant
# text. So rank lines by font size, not position.
#
# STEPS (largest-font line + keyword guard):
#   1. lines = page.extract_text_lines()   ← each: {"text": str, "chars": [ {...} ]}
#   2. if not lines: return None
#   3. keep candidates that LOOK like a scheme name:
#        - text contains "fund" or "scheme" (case-insensitive)  ← rejects callouts
#        - len(text.strip()) < 80                                ← rejects paragraphs
#   4. if candidates: return the one with the LARGEST font
#        font size of a line = max(c["size"] for c in line["chars"])
#      (size breaks ties between several "fund" lines — picks the big title over
#       a small "Fund Manager: ..." line)
#   5. else: return None   ← no scheme title on this page → pipeline uses fund_house
#
# Write detect_scheme_name() here:
def detect_scheme_name(page):
    lines = page.extract_text_lines()
    if not lines:
        return None

    def font_size(line):
        return max(c["size"] for c in line["chars"])

    candidates = [
        ln for ln in lines
        if ("fund" in ln["text"].lower() or "scheme" in ln["text"].lower())
        and len(ln["text"].strip()) < 80
    ]
    if not candidates:
        return None

    best = max(candidates, key=font_size)
    text = best["text"].strip()

    # Clip at "fund": the title line can carry trailing text (e.g. the subtitle
    # "An open ended equity scheme...") merged onto the same extracted line.
    # Keep everything up to and including "fund"; drop the rest.
    idx = text.lower().find("fund")
    if idx != -1:
        text = text[: idx + len("fund")]
    return text.strip()


# -----------------------------------------------------------------------------
# parse_pdf(path: str) -> list[tuple[int, str | None, str]]
#
# WHY: The pipeline needs every page's content together with its page number
# (for citations) AND its scheme name (for fund_name). This wraps the helpers
# above and walks the whole document.
#
# STEPS:
#   1. Open the PDF with pdfplumber.open(path) as a context manager
#   2. For each page, with its 1-based index:
#        - scheme  = detect_scheme_name(page)
#        - content = extract_page_content(page)
#        - append (page_number, scheme, content) to a results list
#   3. Return the list of (page_number, scheme, content) tuples
#
# NOTE: returning (page_number, scheme, content) keeps parsing decoupled from
# chunking — the chunker decides how to slice; the parser only extracts. A None
# scheme means "no header found"; the pipeline falls back to the fund house.
#
# Write parse_pdf() here:

def parse_pdf(path: str) -> list[tuple[int, str | None, str]]:

    with pdfplumber.open(path) as pdf:

        output_parsed = []
        for i, page in enumerate(pdf.pages, start=1):
            scheme_name = detect_scheme_name(page)
            output_parsed.append((i, scheme_name, extract_page_content(page)))
            # pdfplumber caches each page's parsed objects on the PDF object and
            # never frees them while iterating — memory climbs to the whole doc's
            # worth on a 189-page file. Flush this page's cache before the next so
            # peak memory stays ~one page, not the entire document (OOM guard).
            page.flush_cache()

    return output_parsed