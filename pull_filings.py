"""
pull_filings.py

Pulls Strategy Inc (formerly MicroStrategy, CIK 1050446) 8-K filings from
EDGAR and extracts weekly ATM common stock issuance data.

Run locally -- this hits data.sec.gov and www.sec.gov directly, which
isn't reachable from a sandboxed environment.

Usage:
    python pull_filings.py

Output:
    data/atm_issuance_raw.csv

EDGAR requires a real User-Agent with a contact email. Set EDGAR_USER_AGENT
before running; there is no default and the script exits without it.
"""

import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

CIK = "0001050446"  # Strategy Inc / MicroStrategy
SUBMISSIONS_URL = f"https://data.sec.gov/submissions/CIK{CIK}.json"
ARCHIVE_BASE = f"https://www.sec.gov/Archives/edgar/data/{int(CIK)}"

# SEC fair-access policy requires a real contact in the User-Agent. Sending a
# placeholder is worse than failing: it gets the whole IP range throttled and
# gives SEC no way to reach whoever is generating the traffic. So: no default.
USER_AGENT = os.environ.get("EDGAR_USER_AGENT")
if not USER_AGENT or "example.com" in USER_AGENT:
    sys.exit(
        "[ERROR] EDGAR_USER_AGENT is not set to a real contact.\n"
        "        SEC requires a User-Agent identifying who is making the request.\n"
        '        export EDGAR_USER_AGENT="Your Name your@email.com"'
    )

OUTPUT_PATH = Path(__file__).parent / "data" / "atm_issuance_raw.csv"

# Filings before this date are not pulled. Weekly ATM disclosure via 8-K is
# a relatively recent practice; adjust if you confirm earlier coverage.
EARLIEST_DATE = "2024-01-01"


def fetch_json(url: str) -> dict:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=30) as resp:
        raw_html = resp.read().decode("utf-8", errors="replace")
    return strip_html(raw_html)


def strip_html(raw_html: str) -> str:
    """Strip tags and collapse whitespace so regex matching works on plain
    text. Raw EDGAR HTML can split words like 'MSTR Shares' across inline
    tags (e.g. <font>MSTR</font> <font>Shares</font>), which would break
    phrase matching if tags weren't removed first.

    Table rows are converted to a single pipe-delimited line per <tr> so the
    table-format parser (see below) has a stable shape to match against,
    rather than losing all structure to whitespace collapse.
    """
    # Drop script/style blocks entirely -- their contents aren't filing text.
    text = re.sub(
        r"<(script|style)[^>]*>.*?</\1>", " ", raw_html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Convert each table cell boundary to " | " and each row to its own
    # line, before stripping remaining tags. This keeps "Security | Shares
    # Sold | Notional Value | Net Proceeds | Available" style rows intact
    # as single lines instead of one long blob.
    text = re.sub(r"<t[dh][^>]*>", " | ", text, flags=re.IGNORECASE)
    text = re.sub(r"</tr\s*>", "\n", text, flags=re.IGNORECASE)
    # Turn other block-level closing tags into paragraph breaks before
    # stripping remaining tags, so paragraph-level splitting later works.
    text = re.sub(r"</(p|div|br)\s*>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common HTML entities that matter for parsing (commas in
    # numbers, dashes used as "no value", non-breaking spaces).
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = text.replace("&#160;", " ").replace("&#8194;", " ")  # non-breaking/en spaces -> ASCII
    text = text.replace("&#8217;", "'").replace("&#8220;", '"').replace("&#8221;", '"')
    text = text.replace("&#8212;", "--").replace("&#8211;", "-")
    # Collapse runs of whitespace within a line, but preserve the line/
    # paragraph breaks inserted above.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" \| +", " | ", text)
    text = re.sub(r"\n[ \t]*\n+", "\n\n", text)
    return text.strip()


def get_8k_filings(submissions: dict) -> list[dict]:
    """Flatten the columnar filings.recent structure into row dicts,
    filtered to 8-K filings since EARLIEST_DATE."""
    recent = submissions["filings"]["recent"]
    rows = []
    n = len(recent["accessionNumber"])
    for i in range(n):
        form = recent["form"][i]
        if form != "8-K":
            continue
        filing_date = recent["filingDate"][i]
        if filing_date < EARLIEST_DATE:
            continue
        rows.append(
            {
                "accessionNumber": recent["accessionNumber"][i],
                "filingDate": filing_date,
                "primaryDocument": recent["primaryDocument"][i],
                "items": recent.get("items", [""] * n)[i],
            }
        )
    # NOTE: submissions.json only carries the most recent ~1000 filings or
    # ~1 year, whichever is more. If EARLIEST_DATE needs to reach further
    # back, this script will need to follow submissions["filings"]["files"]
    # for older archived JSON shards. Check the printed count against what
    # you expect before assuming full coverage.
    return rows


def filing_doc_url(accession_number: str, primary_document: str) -> str:
    acc_nodash = accession_number.replace("-", "")
    return f"{ARCHIVE_BASE}/{acc_nodash}/{primary_document}"


def looks_like_atm_update(text: str) -> bool:
    # Require one of three phrases as a necessary condition:
    #   "during period"            -- Format B/B-split table header
    #   "during the period between" -- Format A prose
    #   "atm program summary"      -- Format C multi-ATM table header
    # This filters out Sales Agreement amendments, BTC-only 8-Ks, equity
    # offering announcements, and other 8-Ks that mention "ATM" in passing
    # but have no weekly issuance data.
    lowered = text.lower()
    return (
        "during period" in lowered
        or "during the period between" in lowered
        or "atm program summary" in lowered
    )


# --- Parsing -----------------------------------------------------------------
# Strategy has used (at least) three disclosure formats across 2024-2026.
# Formats A and B are fully supported. Format C is intentionally not parsed
# (scope stop).
#
# FORMAT A -- prose (seen through ~mid 2025):
#   Two wording variants, both handled:
#   V1 (post-rebrand, ~Mar 2025): "sold an aggregate of X MSTR Shares under
#     the Common ATM ... approximately $Y million"
#   V2 (pre-rebrand, ~Oct-Nov 2024): "had sold an aggregate of X Shares under
#     the Sales Agreement ... approximately $Y billion"
#   Zero-sale weeks: "did not sell any shares/MSTR Shares" -- emitted as
#     shares=0 / proceeds=0 (format A_prose_zero).
#   Confirmed date range: Oct 2024 through ~March 2025.
#
# FORMAT B -- table (seen from ~late 2025 on), e.g. d122015d8k.htm:
#   "On March 9, 2026, Strategy Inc ("Strategy") announced an update with
#    respect to sales made under its at-the-market offering program ("ATM")
#    of the following securities:"
#   followed by a table with a header row "During Period March 2, 2026 to
#   March 8, 2026" and rows per security. MSTR row anchor is "MSTR Stock"
#   with "Class A Common Stock" on the same or next line. Data cells are
#   one value per line. Columns: Shares Sold, Notional Value (blank for
#   MSTR), Net Proceeds, Available. Zero-sale weeks (all dashes) are
#   emitted as shares=0 / proceeds=0.
#   Confirmed date range: Nov 2025 through 2026+.
#
# FORMAT C -- multi-ATM program summary table (~Apr-Oct 2025):
#   "ATM Program Summary" table with one row per ATM program (Common ATM /
#   MSTR ATM, STRK ATM, STRF ATM, etc.).  Only the MSTR common stock row
#   is extracted; preferred-stock rows are ignored.
#
#   Key structural facts (verified against Apr 14, Jun 16, Aug 18, Sep 8):
#   - Anchor: "| Common ATM" (Apr-Aug 2025) OR "| MSTR ATM" (Sep 2025+)
#   - Window: "| During Period" label on one line; date range on the NEXT
#     non-empty line as "Month D, YYYY to Month D, YYYY" (no "|" prefix).
#   - Shares: "| 959,712 MSTR Shares" (prose with suffix) or "| - " (zero)
#   - Columns after anchor (positional):
#       Shares Sold  |  [Notional Value -- present in Sep+, absent in Apr-May]
#       Net Proceeds  |  Available for Issuance
#     The Notional Value column appears in filings from ~Jun 2025 onward;
#     the Apr-May filings skip it.  We detect its presence by checking
#     whether the first post-anchor value cell contains "MSTR Shares" (=
#     shares column) vs a plain number (= notional, meaning shares was "-").
#   - Proceeds: "| $285.7 million" early (Apr-May); "| $200.5" later (no
#     "million" suffix -- the column header already says "in millions").
#   - Zero-sale weeks: shares cell is "| - " and proceeds cell is "| - ".
#     Emitted as shares=0 / proceeds=0 (format C_table_zero).
#
# FORMAT A zero-sale weeks: prose filings where Strategy sold no MSTR shares
#   use language like "did not sell any MSTR Shares". These pass the
#   looks_like_atm_update filter (they have a "During Period" in the BTC
#   section) but neither format matches -- logged to stderr, not an error.
#
# Both formats are tried on every filing. Whichever matches wins. If
# neither matches but the filing looked ATM-related, it's logged for
# manual review rather than silently dropped.

MAX_WINDOW_DAYS = 10  # reject windows longer than this as cumulative ranges


def _parse_date_loose(raw: str) -> "datetime | None":
    raw = raw.strip().rstrip(",")
    for fmt in ("%B %d, %Y", "%B %d %Y"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _window_span_ok(start_raw: str, end_raw: str, source_url: str) -> bool:
    start_dt = _parse_date_loose(start_raw)
    end_dt = _parse_date_loose(end_raw)
    if not (start_dt and end_dt):
        return True  # can't tell, don't block on a parse failure here
    span_days = (end_dt - start_dt).days
    if span_days > MAX_WINDOW_DAYS or span_days < 0:
        print(
            f"[skipped range] {source_url} :: {start_raw} - {end_raw} "
            f"(span {span_days}d, looks like a cumulative range, not a "
            f"weekly window)",
            file=sys.stderr,
        )
        return False
    return True


# FORMAT A regexes
WINDOW_RE_A = re.compile(
    r"during the period between\s+"
    r"([A-Za-z]+ \d{1,2},? \d{4})\s+and\s+"
    r"([A-Za-z]+ \d{1,2},? \d{4})",
    re.IGNORECASE,
)
# Post-rebrand wording (~late 2024 on): "MSTR Shares under the Common ATM"
MSTR_SHARES_RE_A = re.compile(
    r"sold an aggregate of\s+([\d,]+)\s+MSTR Shares\s+under the Common ATM"
    r".{0,200}?approximately\s*\$([\d,.]+)\s*(million|billion)",
    re.IGNORECASE | re.DOTALL,
)
# Pre-rebrand wording (~Oct-Nov 2024): "Shares under the Sales Agreement"
# Proceeds may be "$ 4.6 billion" (with space after $).
MSTR_SHARES_RE_A_V2 = re.compile(
    r"(?:had\s+)?sold an aggregate of\s+([\d,]+)\s+Shares\s+under the Sales Agreement"
    r".{0,300}?approximately\s*\$\s*([\d,.]+)\s*(million|billion)",
    re.IGNORECASE | re.DOTALL,
)
# Zero-sale prose: "did not sell any shares" or "did not sell any MSTR Shares"
ZERO_SALE_RE_A = re.compile(
    r"did not sell any (?:shares of class [AB] common stock|MSTR Shares|Shares)",
    re.IGNORECASE,
)


def parse_format_a(text: str, source_url: str) -> list[dict]:
    results = []
    paragraphs = re.split(r"\n\s*\n+", text)
    if len(paragraphs) <= 2 and text.lower().count("during the period between") > 1:
        # Raw HTML didn't preserve blank-line paragraph breaks -- fall back
        # to sentence-window splitting so date clauses stay near their own
        # sale figures rather than bleeding into neighboring tranches.
        sentences = re.split(r"(?<=[.])\s+(?=[A-Z])", text)
        paragraphs = [" ".join(sentences[i : i + 3]) for i in range(len(sentences))]

    for para in paragraphs:
        window_match = WINDOW_RE_A.search(para)
        if not window_match:
            continue
        window_start_raw, window_end_raw = window_match.groups()
        if not _window_span_ok(window_start_raw, window_end_raw, source_url):
            continue

        # Try both sale-wording variants; skip paragraphs that match neither
        # and don't contain the zero-sale phrase (likely preferred-stock or
        # BTC paragraphs that share the same window dates).
        sale_match = MSTR_SHARES_RE_A.search(para) or MSTR_SHARES_RE_A_V2.search(para)
        if not sale_match:
            if ZERO_SALE_RE_A.search(para):
                results.append(
                    {
                        "window_start_raw": window_start_raw,
                        "window_end_raw": window_end_raw,
                        "shares_sold": 0,
                        "net_proceeds_usd": 0.0,
                        "source_url": source_url,
                        "format": "A_prose_zero",
                    }
                )
            continue  # preferred-stock / BTC paragraph, skip

        shares_str, proceeds_str, unit = sale_match.groups()
        shares = int(shares_str.replace(",", ""))
        proceeds_raw = float(proceeds_str.replace(",", ""))
        multiplier = 1_000_000_000 if unit.lower() == "billion" else 1_000_000
        results.append(
            {
                "window_start_raw": window_start_raw,
                "window_end_raw": window_end_raw,
                "shares_sold": shares,
                "net_proceeds_usd": proceeds_raw * multiplier,
                "source_url": source_url,
                "format": "A_prose",
            }
        )
    return results


# FORMAT B regexes
# Header line e.g.: "During Period March 2, 2026 to March 8, 2026"
# (&#160; is decoded to ASCII space before this regex runs)
WINDOW_RE_B = re.compile(
    r"During Period\s+([A-Za-z]+ \d{1,2},? \d{4})\s+to\s+"
    r"([A-Za-z]+ \d{1,2},? \d{4})",
    re.IGNORECASE,
)
# Each table cell is its own line after strip_html (one <tr> cell per line).
# We anchor on the MSTR security name line, then collect number/dash tokens
# from the lines that follow to read: Shares Sold, Notional Value, Net
# Proceeds positionally. Collecting positionally keeps the columns aligned
# even when MSTR notional is blank ("--").
_MSTR_ANCHOR_RE = re.compile(r"MSTR Stock.*?Common Stock", re.IGNORECASE)
# A line that is purely a cell value: optional leading " | ", then either a
# number (with optional $ and commas) or a dash placeholder.
_CELL_VALUE_RE = re.compile(r"^\s*\|?\s*\$?\s*([\d,]+\.?\d*|---|--|—|-|0)\s*$")


def _to_float_or_zero(raw: str) -> float:
    raw = raw.strip()
    if raw in ("-", "--", "---", "—", ""):
        return 0.0
    return float(raw.replace(",", ""))


def _extract_mstr_row_tokens(lines: list[str], anchor_idx: int) -> list[str]:
    """Collect up to 4 value tokens from lines after the MSTR anchor line.

    Stops early if we hit another security-name line or 'Total', so we don't
    bleed into the next row. Returns the raw token strings."""
    tokens = []
    for ln in lines[anchor_idx + 1:]:
        stripped = ln.strip()
        if not stripped or stripped == "|":
            continue
        # Another security-name line or Total signals end of MSTR row.
        if re.search(r"\b(STR[CFKD]|Total)\b", stripped, re.IGNORECASE):
            break
        m = _CELL_VALUE_RE.match(stripped)
        if m:
            tokens.append(m.group(1))
            if len(tokens) == 4:
                break
    return tokens


_MSTR_STOCK_ONLY_RE = re.compile(r"^\s*\|?\s*MSTR Stock\s*$", re.IGNORECASE)


def parse_format_b(text: str, source_url: str) -> list[dict]:
    results = []
    window_match = WINDOW_RE_B.search(text)
    if not window_match:
        return results
    window_start_raw, window_end_raw = window_match.groups()
    if not _window_span_ok(window_start_raw, window_end_raw, source_url):
        return results

    lines = text.splitlines()

    # Primary anchor: "MSTR Stock ... Class A Common Stock" on one line.
    fmt = "B_table"
    anchor_idx = next(
        (i for i, ln in enumerate(lines) if _MSTR_ANCHOR_RE.search(ln)), None
    )

    if anchor_idx is None:
        # Fallback: "MSTR Stock" alone on a line (B-split layout where
        # "Class A Common Stock" is on a separate line). Used in late 2025
        # / early 2026 filings. Same token extraction as the primary path.
        split_idx = next(
            (i for i, ln in enumerate(lines) if _MSTR_STOCK_ONLY_RE.match(ln)), None
        )
        if split_idx is None:
            return results
        anchor_idx = split_idx
        fmt = "B_table_split"

    tokens = _extract_mstr_row_tokens(lines, anchor_idx)
    # Expect at least 3 tokens: shares, notional, net_proceeds.
    if len(tokens) < 3:
        print(
            f"[format B token underrun] {source_url} :: got {tokens!r}, "
            f"expected at least 3 (shares, notional, net_proceeds)",
            file=sys.stderr,
        )
        return results

    shares_raw, _notional_raw, proceeds_raw = tokens[0], tokens[1], tokens[2]
    shares = int(_to_float_or_zero(shares_raw))
    proceeds_millions = _to_float_or_zero(proceeds_raw)

    results.append(
        {
            "window_start_raw": window_start_raw,
            "window_end_raw": window_end_raw,
            "shares_sold": shares,
            "net_proceeds_usd": proceeds_millions * 1_000_000,
            "source_url": source_url,
            "format": fmt,
        }
    )
    return results


# FORMAT C regexes
# Window: either inline "| During Period June 30, 2025 to July 6, 2025"
# or split "| During Period\n  June 30, 2025 to July 6, 2025".
# We match both by searching for "During Period" and extracting dates from
# the same line if present, else from the next non-empty line.
_WINDOW_HEADER_C_RE = re.compile(r"\|\s*During Period", re.IGNORECASE)
_DATE_RANGE_C_RE = re.compile(
    r"([A-Za-z]+ \d{1,2},?\s+\d{4})\s+to\s+([A-Za-z]+ \d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)
# Anchor for MSTR common stock row:
#   "Common ATM"      (Apr-Aug 2025)
#   "2025 Common ATM" (some May 2025 filings -- year-prefixed variant)
#   "MSTR ATM"        (Sep 2025+)
_MSTR_ANCHOR_C_RE = re.compile(
    r"\|\s*(?:\d{4}\s+)?(?:Common ATM|MSTR ATM)\s*$", re.IGNORECASE
)
# Shares cell: "| 959,712 MSTR Shares" -- extract the integer
_SHARES_C_RE = re.compile(r"([\d,]+)\s+MSTR Shares", re.IGNORECASE)
# Proceeds cell: "| $285.7 million" (early) or "| $200.5" (later)
# Returns (value_str, unit_or_none) -- if no "million" suffix, column header
# already says "in millions" so unit is implicitly millions.
_PROCEEDS_C_RE = re.compile(
    r"\|\s*\$([\d,.]+)(?:\s*(million|billion))?\s*$", re.IGNORECASE
)
# Dash cell: "| - " or "| -- " (zero value placeholder)
_DASH_C_RE = re.compile(r"^\s*\|\s*-{1,3}\s*$")


def parse_format_c(text: str, source_url: str) -> list[dict]:
    """Parse Format C multi-ATM summary tables (Apr-Oct 2025)."""
    lines = text.splitlines()

    # --- Find the window dates from the "During Period" header ---------------
    window_start_raw = window_end_raw = None
    for i, ln in enumerate(lines):
        if _WINDOW_HEADER_C_RE.search(ln):
            # Dates may be inline (same line) or on the next non-empty line.
            m = _DATE_RANGE_C_RE.search(ln)
            if m:
                window_start_raw = m.group(1).strip()
                window_end_raw = m.group(2).strip()
            else:
                for j in range(i + 1, min(i + 6, len(lines))):
                    candidate = lines[j].strip()
                    if not candidate:
                        continue
                    m = _DATE_RANGE_C_RE.search(candidate)
                    if m:
                        window_start_raw = m.group(1).strip()
                        window_end_raw = m.group(2).strip()
                    break
            break  # only one During Period header per filing

    if not (window_start_raw and window_end_raw):
        return []
    if not _window_span_ok(window_start_raw, window_end_raw, source_url):
        return []

    # --- Find MSTR common stock anchor row -----------------------------------
    anchor_idx = None
    for i, ln in enumerate(lines):
        if _MSTR_ANCHOR_C_RE.search(ln):
            anchor_idx = i
            break

    if anchor_idx is None:
        return []

    # --- Extract value cells after anchor ------------------------------------
    # Collect non-empty, non-pipe-only lines until we hit the next ATM program
    # row or a "Total" line.  We want: Shares Sold, [Notional], Net Proceeds.
    #
    # Some filings split a shares cell across two consecutive lines:
    #   "| 797,008"      <- numeric part
    #   " MSTR Shares"   <- suffix on the next line (no leading "|")
    # We handle this by peeking at the next non-empty line when we see a
    # bare numeric cell and joining them before matching.
    _OTHER_PROGRAM_RE = re.compile(
        r"\|\s*(STR[CFKD] ATM|Total)\b", re.IGNORECASE
    )
    # Bare number cell (no "MSTR Shares" suffix yet): "| 797,008"
    _BARE_NUM_C_RE = re.compile(r"^\s*\|\s*([\d,]+)\s*$")

    shares = 0
    proceeds_usd = 0.0
    shares_found = False
    proceeds_found = False
    zero_sale = False

    post_anchor = [
        ln for ln in lines[anchor_idx + 1:]
        if ln.strip() and ln.strip() != "|"
    ]

    i_c = 0
    while i_c < len(post_anchor):
        stripped = post_anchor[i_c].strip()
        i_c += 1

        if _OTHER_PROGRAM_RE.search(stripped):
            break

        # Check for split shares cell: bare number followed by "MSTR Shares"
        if not shares_found:
            bm = _BARE_NUM_C_RE.match(stripped)
            if bm:
                # Peek at next line
                if i_c < len(post_anchor) and "MSTR Shares" in post_anchor[i_c]:
                    shares = int(bm.group(1).replace(",", ""))
                    shares_found = True
                    i_c += 1  # consume the suffix line
                    continue
                # Bare number not followed by "MSTR Shares" -- could be
                # notional or available; fall through to other checks.

            sm = _SHARES_C_RE.search(stripped)
            if sm:
                shares = int(sm.group(1).replace(",", ""))
                shares_found = True
                continue
            if _DASH_C_RE.match(stripped):
                zero_sale = True
                shares_found = True
                continue

        # Once shares are handled, look for proceeds.
        # Skip notional-value cells (they look like plain $ amounts too,
        # but the Available cell also has a dollar sign).  We distinguish
        # by only accepting the FIRST dollar-value cell after shares.
        if shares_found and not proceeds_found:
            pm = _PROCEEDS_C_RE.search(stripped)
            if pm:
                val = float(pm.group(1).replace(",", ""))
                unit = (pm.group(2) or "").lower()
                if unit == "billion":
                    proceeds_usd = val * 1_000_000_000
                else:
                    # No suffix or "million" -- column header says "in millions"
                    proceeds_usd = val * 1_000_000
                proceeds_found = True
                break
            if _DASH_C_RE.match(stripped):
                # Notional-value dash OR proceeds dash -- if shares was also
                # zero, this is a zero-sale week; otherwise skip (notional).
                if zero_sale:
                    proceeds_found = True
                    break
                # Skip this dash (it's the notional column) and keep looking
                continue

    if not shares_found:
        print(
            f"[format C shares not found] {source_url} :: anchor found but "
            f"could not extract shares cell",
            file=sys.stderr,
        )
        return []

    fmt = "C_table_zero" if (shares == 0 and proceeds_usd == 0.0) else "C_table"
    return [
        {
            "window_start_raw": window_start_raw,
            "window_end_raw": window_end_raw,
            "shares_sold": shares,
            "net_proceeds_usd": proceeds_usd,
            "source_url": source_url,
            "format": fmt,
        }
    ]


def parse_atm_text(text: str, source_url: str, filed_at: str) -> list[dict]:
    results = parse_format_a(text, source_url)
    if not results:
        results = parse_format_b(text, source_url)
    if not results:
        results = parse_format_c(text, source_url)

    for r in results:
        r["filed_at"] = filed_at

    if not results:
        print(
            f"[no MSTR common ATM row found] {source_url} -- matched ATM "
            f"keywords but no format (A prose / B table / C multi-ATM) "
            f"extracted an MSTR common-stock row. Could be a genuine "
            f"preferred-only week, an unseen format variant, or zero MSTR "
            f"shares sold that week -- check by eye.",
            file=sys.stderr,
        )
    return results


def main():
    print(f"Fetching filing list for CIK {CIK} ...")
    try:
        submissions = fetch_json(SUBMISSIONS_URL)
    except (HTTPError, URLError) as e:
        print(f"Failed to fetch submissions list: {e}", file=sys.stderr)
        print(
            "Check your network connection and that EDGAR_USER_AGENT "
            "contains a real contact email.",
            file=sys.stderr,
        )
        sys.exit(1)

    filings = get_8k_filings(submissions)
    print(f"Found {len(filings)} 8-K filings since {EARLIEST_DATE}.")
    print(
        "If this count looks low for the date range, submissions.json may "
        "only cover ~1 year/1000 filings -- check filings['files'] in the "
        "raw JSON for older shards.\n"
    )

    all_rows = []
    format_counts: dict[str, int] = {}
    for i, f in enumerate(filings):
        url = filing_doc_url(f["accessionNumber"], f["primaryDocument"])
        print(f"[{i+1}/{len(filings)}] {f['filingDate']} -- {url}")
        try:
            text = fetch_text(url)
        except (HTTPError, URLError) as e:
            print(f"  failed to fetch: {e}", file=sys.stderr)
            continue

        if not looks_like_atm_update(text):
            continue  # not an ATM-related 8-K, skip silently

        rows = parse_atm_text(text, url, f["filingDate"])
        for r in rows:
            fmt = r["format"]
            format_counts[fmt] = format_counts.get(fmt, 0) + 1
        all_rows.extend(rows)

        time.sleep(0.15)  # stay well under SEC's 10 req/s limit

    print(f"\nParsed {len(all_rows)} weekly ATM rows total.")
    for fmt, n in sorted(format_counts.items()):
        print(f"  {fmt}: {n}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "window_start_raw",
                "window_end_raw",
                "shares_sold",
                "net_proceeds_usd",
                "filed_at",
                "source_url",
                "format",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {OUTPUT_PATH}")
    print(
        "\nNext: spot-check a handful of rows from each format against "
        "source_url by eye before trusting this. Unparsed filings were "
        "logged above to stderr -- review those manually if the count is "
        "non-trivial."
    )


if __name__ == "__main__":
    main()
