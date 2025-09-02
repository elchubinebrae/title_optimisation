import os, time, re, itertools, collections, base64, requests, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from urllib.parse import quote, urlsplit, urlunsplit, urlencode, parse_qsl
import matplotlib.patheffects as pe
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import html
from collections import defaultdict

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 100

# -------------------------------
# CONFIG (hardcode creds if you want)
# -------------------------------

EBAY_CLIENT_ID = st.secrets["EBAY_CLIENT_ID"]
EBAY_CLIENT_SECRET = st.secrets["EBAY_CLIENT_SECRET"]
MARKETPLACE_ID = "EBAY_GB"   # change to EBAY_US, EBAY_DE, etc.

# -------------------------------
# TOKEN MANAGEMENT
# -------------------------------
EBAY_ACCESS_TOKEN = None
EBAY_TOKEN_EXPIRY = 0

CAMP_ID = "5339121284"  # e.g., "5338902233"


def render_help_popover():
    # Use st.popover if available; otherwise fall back to st.expander
    try:
        with st.popover("ℹ️ How do I read these charts?"):
            st.markdown(
                """
**Variant Lift vs Base (Δ)**
- **Bar length** = average change in rank vs the seed query (`Δ = base_rank − variant_rank`).  
  - Positive (right) = variant ranks higher than the seed.
  - Negative (left)  = variant ranks worse than the seed.
- **Label/color** = **Found rate** (how often your target items appeared at all).
  - Aim for **positive Δ** *and* **high found rate** → keeper phrases.

**Absolute Average Rank**
- **Lower bars** = closer to the top of search (better).  
- Use this to sanity-check when Δ feels abstract.

**Token / N-gram Importance**
- Estimated contribution of each token/phrase to rank lift across variants.
- Push **high-lift tokens** to the **front** of your title; demote low-lift ones to description/specs.
                """
            )
    except Exception:
        with st.expander("ℹ️ How do I read these charts?"):
            st.markdown(
                """
(Using expander fallback — same guidance as above.)
**Variant Lift vs Base (Δ)** … (same text)
                """
            )


def get_ebay_access_token(client_id: str, client_secret: str) -> str:
    global EBAY_ACCESS_TOKEN, EBAY_TOKEN_EXPIRY
    now = time.time()
    if EBAY_ACCESS_TOKEN and now < EBAY_TOKEN_EXPIRY:
        return EBAY_ACCESS_TOKEN

    credentials = f"{client_id}:{client_secret}"
    encoded = base64.b64encode(credentials.encode()).decode()
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {encoded}"
    }
    data = {
        "grant_type": "client_credentials",
        "scope": "https://api.ebay.com/oauth/api_scope"
    }
    r = requests.post("https://api.ebay.com/identity/v1/oauth2/token", headers=headers, data=data, timeout=25)
    r.raise_for_status()
    j = r.json()
    EBAY_ACCESS_TOKEN = j["access_token"]
    EBAY_TOKEN_EXPIRY = now + int(j["expires_in"]) - 60
    return EBAY_ACCESS_TOKEN


def compute_seller_strength(rows_df):
    df = rows_df.copy()
    df = df[df["rank"] < 150]  # ignore "not found"

    # Normalise rank → higher = better
    df["rank_score"] = 100 - df["rank"]

    grouped = df.groupby("seller").agg(
        avg_rank=("rank", "mean"),
        avg_score=("rank_score", "mean"),
        appearances=("variant", "count"),
        coverage=("variant", "nunique")
    ).reset_index()

    # Composite metric: avg_score × appearances
    grouped["seller_score"] = grouped["avg_score"] * grouped["appearances"]

    return grouped.sort_values("seller_score", ascending=False)


# -------------------------------
# UTILITIES
# -------------------------------
def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r'[^a-z0-9+\s-]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def tokenise(text: str):
    return [m.group(0).lower() for m in WORD_RE.finditer(str(text))]

def compute_token_lift(rows_df: pd.DataFrame,
                       title_col: str = "title",
                       min_support: int = 3,
                       max_doc_frac: float = 0.6) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: token, lift, support, present_mean, absent_mean
    lift = mean(1/rank | token present) - mean(1/rank | token absent)
    """
    df = rows_df.copy()
    if "rank" not in df or title_col not in df:
        raise ValueError("rows_df must include columns 'rank' and the title column.")
    df["y"] = 1.0 / pd.to_numeric(df["rank"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["y"])

    # Build presence lists per token
    present_vals = defaultdict(list)
    absent_vals  = defaultdict(list)

    titles_tokens = [set(tokenise(t)) for t in df[title_col].fillna("")]
    yvals = df["y"].to_numpy()

    all_tokens = set().union(*titles_tokens)
    n_docs = len(df)

    # Precompute doc freq
    dfreq = {tok: sum(tok in toks for toks in titles_tokens) for tok in all_tokens}

    # Filter tokens by support bounds
    keep = {tok for tok, c in dfreq.items()
            if c >= min_support and c <= int(max_doc_frac * n_docs)}

    # Accumulate y by presence/absence for kept tokens
    for toks, y in zip(titles_tokens, yvals):
        for tok in keep:
            if tok in toks:
                present_vals[tok].append(y)
            else:
                absent_vals[tok].append(y)

    rows = []
    for tok in keep:
        p = present_vals[tok]
        a = absent_vals[tok]
        if len(p) >= min_support and len(a) >= min_support:
            present_mean = float(np.mean(p))
            absent_mean  = float(np.mean(a))
            lift = present_mean - absent_mean
            rows.append(dict(
                token=tok,
                lift=lift,
                support=len(p),
                present_mean=present_mean,
                absent_mean=absent_mean
            ))

    out = pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)
    return out

def token_scores_from_lift(lift_df: pd.DataFrame) -> dict[str, float]:
    return {row.token: float(row.lift) for _, row in lift_df.iterrows()}

def ngrams(tokens, n=2):
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def dedupe_keep_order(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def generate_variants(title: str, max_variants=25):
    tokens = tokenise(title)
    bigrams = ngrams(tokens, 2)
    trigrams = ngrams(tokens, 3)

    cands = []
    cands += [' '.join(tokens)]
    cands += [' '.join(dedupe_keep_order(tokens))]
    cands += [' '.join([t for t in tokens if t not in {"hot","cold","wind"}])]
    cands += bigrams[:8] + trigrams[:6]
    cands.append(' '.join(tokens))
    cands.append(' '.join(dedupe_keep_order(bigrams[:3] + tokens)))

    for t in tokens[:6]:
        cands.append(' '.join([x for x in tokens if x != t]))

    cands = [normalize_text(c) for c in cands if len(c.split()) >= 2]
    cands = dedupe_keep_order(cands)[:max_variants]
    return cands

def ebay_search(keyword: str, limit=100):
    token = get_ebay_access_token(EBAY_CLIENT_ID, EBAY_CLIENT_SECRET)
    headers = {
        "Authorization": f"Bearer {token}",
        "X-EBAY-C-MARKETPLACE-ID": MARKETPLACE_ID,
    }
    params = {"q": keyword, "limit": str(limit)}
    r = requests.get("https://api.ebay.com/buy/browse/v1/item_summary/search", headers=headers, params=params, timeout=25)
    r.raise_for_status()
    data = r.json()
    items = data.get("itemSummaries", []) or []
    out = []
    for idx, it in enumerate(items, start=1):
        out.append({
            "rank": idx,
            "item_id": it.get("itemId"),
            "title": it.get("title",""),
            "price": (it.get("price") or {}).get("value"),
            "currency": (it.get("price") or {}).get("currency"),
            "seller": (it.get("seller") or {}).get("username"),
            "category": (it.get("categoryPath") or ""),
        })
    return out

def exact_match_ids(base_keyword: str, top_n=10):
    res = ebay_search(base_keyword, limit=100)
    ids = [r["item_id"] for r in res[:top_n] if r["item_id"]]
    base_pos = {r["item_id"]: r["rank"] for r in res if r.get("item_id")}
    base_meta = {
        r["item_id"]: {
            "seller": r.get("seller"),
            "title": r.get("title")
        }
        for r in res if r.get("item_id")
    }
    return ids, base_pos, base_meta, res


def evaluate_variants(
    base_title: str,
    variants: list[str],
    target_ids: list[str],
    base_pos: dict[str, int],
    base_meta: dict[str, dict],
    polite_delay: float = 0.3
) -> pd.DataFrame:
    """
    Evaluate how each variant ranks the target items.
    Includes seller so downstream 'seller strength' can be computed.
    """
    rows = []
    for v in variants:
        time.sleep(polite_delay)
        res = ebay_search(v, limit=100)

        # Build quick lookups from this variant's results
        rank_map   = {r["item_id"]: r["rank"]   for r in res if r.get("item_id")}
        seller_map = {r["item_id"]: r.get("seller") for r in res if r.get("item_id")}
        title_map  = {r["item_id"]: r.get("title") for r in res if r.get("item_id")}

        for tid in target_ids:
            vrank = rank_map.get(tid, None)
            brank = base_pos.get(tid, None)

            # penalties if missing
            if vrank is None: vrank = 150
            if brank is None: brank = 150

            delta = brank - vrank

            # prefer seller from current variant result; fallback to base cohort seller
            seller = seller_map.get(tid) or (base_meta.get(tid, {}).get("seller"))
            title  = title_map.get(tid)  or (base_meta.get(tid, {}).get("title"))

            rows.append({
                "variant": v,
                "target_id": tid,
                "seller": seller,
                "title": title,
                "base_rank": brank,
                "rank": vrank,
                "delta": delta
            })

    return pd.DataFrame(rows)


def token_importance(base_title: str, rows_df: pd.DataFrame):
    tokens = dedupe_keep_order(tokenise(base_title))
    extra = []
    for v in rows_df["variant"].unique():
        vt = tokenise(v)
        extra += ngrams(vt,2) + ngrams(vt,3)
    grams = [g for g,c in collections.Counter(extra).items() if c >= 3]
    feats = tokens + grams

    X, y = [], []
    for _, r in rows_df.iterrows():
        vtoks = set(tokenise(r["variant"]))
        vgrams = set(ngrams(tokenise(r["variant"]),2) + ngrams(tokenise(r["variant"]),3))
        row = []
        for f in feats:
            if " " in f:
                row.append(1 if f in vgrams else 0)
            else:
                row.append(1 if f in vtoks else 0)
        X.append(row)
        y.append(r["delta"])
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    if X.size == 0:
        return pd.DataFrame(columns=["feature","lift"])
    lam = 1.0
    XtX = X.T @ X + lam * np.eye(X.shape[1])
    coef = np.linalg.solve(XtX, X.T @ y)
    imp = pd.DataFrame({"feature": feats, "lift": coef})
    imp = imp.sort_values("lift", ascending=False).reset_index(drop=True)
    return imp

def compute_variant_summary(rows_df: pd.DataFrame):
    agg = rows_df.groupby("variant").agg(
        mean_delta=("delta","mean"),
        found_rate=("rank", lambda s: 100.0 * (s.ne(150).sum() / len(s))),
        avg_abs_rank=("rank","mean")
    ).reset_index()
    return agg.sort_values("mean_delta", ascending=False)

def wrap_label(text: str, words_per_line: int = 5, max_lines: int = 3) -> str:
    words = text.split()
    lines = [' '.join(words[i:i+words_per_line]) for i in range(0, len(words), words_per_line)]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] += '…'
    return '\n'.join(lines)

def wrap_labels(labels, words_per_line=5, max_lines=3):
    return [wrap_label(s, words_per_line, max_lines) for s in labels]

def st_theme():
    # Pull Streamlit theme so the chart matches light/dark + brand colors
    base   = st.get_option("theme.base") or "light"
    prim   = st.get_option("theme.primaryColor") or ("#4f46e5" if base=="light" else "#8b93ff")
    bg     = st.get_option("theme.backgroundColor") or ("#ffffff" if base=="light" else "#0e1117")
    sdbg   = st.get_option("theme.secondaryBackgroundColor") or ("#f5f5f9" if base=="light" else "#1a1f2b")
    txt    = st.get_option("theme.textColor") or ("#111827" if base=="light" else "#e5e7eb")
    grid   = "#e5e7eb" if base=="light" else "#2a3342"
    return {"base": base, "primary": prim, "bg": bg, "sbg": sdbg, "text": txt, "grid": grid}

def nice_number(x, step=0.5):
    # pad x-limit so bar-end labels don't get clipped
    if x >= 0:
        return np.ceil(x/step)*step
    return np.floor(x/step)*step

def epn_wrap(url: str, campid: str, customid: str | None = None) -> str:
    u = urlsplit(url)
    q = dict(parse_qsl(u.query, keep_blank_values=True))
    q["campid"] = campid
    if customid:
        q["customid"] = customid
    return urlunsplit((u.scheme, u.netloc, u.path, urlencode(q), u.fragment))

def seller_profile_url(seller: str) -> str:
    return f"https://www.ebay.co.uk/usr/{quote(str(seller))}"

def render_sellers_table(df, campid: str, customid: str = "top-sellers"):
    cols_wanted = ["seller", "avg_rank", "avg_score", "appearances", "coverage", "seller_score"]
    cols = [c for c in cols_wanted if c in df.columns]
    rows = []
    for i, row in df.head(20).reset_index(drop=True).iterrows():
        rank = i + 1
        seller = str(row["seller"])
        display = html.escape(seller)
        url = epn_wrap(seller_profile_url(seller), campid, customid)
        cells = [f"<td class='rank'>{rank}</td>",
                 f"<td class='seller'><a href='{url}' target='_blank' rel='noopener'>{display}</a></td>"]
        for c in cols[1:]:
            val = row[c]
            # make ints look nice
            if isinstance(val, (int, np.integer)) or (isinstance(val, (float, np.floating)) and float(val).is_integer()):
                cells.append(f"<td>{int(val)}</td>")
            else:
                cells.append(f"<td>{html.escape(str(val))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")

    # minimal, Streamlit-dark styled table
    html_table = f"""
    <style>
      .tbl-wrap {{
        background:#0E1117;border-radius:12px;overflow:hidden;border:1px solid #2a2f3a;
      }}
      .tbl {{ width:100%; border-collapse:separate; border-spacing:0; color:#e6e6e6; font-size:0.95rem; }}
      .tbl th, .tbl td {{ padding:10px 12px; border-bottom:1px solid #242935; }}
      .tbl th {{ text-align:left; background:#11151c; font-weight:600; color:#ffffff; }}
      .tbl tr:last-child td {{ border-bottom:none; }}
      .tbl tr:hover td {{ background:#151a22; }}
      .tbl .rank {{ width:64px; color:#cbd5e1; }}
      .tbl .seller a {{ color:#FF4B4B; text-decoration:none; }}
      .tbl .seller a:hover {{ text-decoration:underline; }}
      @media (max-width: 700px) {{
        .tbl th, .tbl td {{ padding:8px 10px; font-size:0.90rem; }}
      }}
    </style>
    <div class="tbl-wrap">
      <table class="tbl">
        <thead>
          <tr>
            <th>Seller&nbsp;Ranking</th>
            <th>seller</th>
            {''.join(f'<th>{html.escape(c)}</th>' for c in cols[1:])}
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>
    """
    return html_table

def _safe(col, fn, default=np.nan):
    try:
        return fn(col)
    except Exception:
        return default

def minmax(s: pd.Series):
    s = s.astype(float)
    vmin, vmax = np.nanmin(s.values), np.nanmax(s.values)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    return (s - vmin) / (vmax - vmin)

def item_url(item_id: str) -> str:
    return f"https://www.ebay.co.uk/itm/{item_id}"

def parse_target_id(t: str):
    """
    Accepts 'v1|<legacy_item_id>|<variation_id>' or just a plain number.
    Returns (item_id, variation_id or None).
    """
    if not isinstance(t, str):
        t = str(t)
    # Try the v1|...|... format
    parts = t.split("|")
    if len(parts) >= 3 and parts[0].lower() == "v1":
        legacy_item_id = parts[1]
        variation_id = parts[2] if parts[2] != "0" else None
        return legacy_item_id, variation_id
    # Fallback: extract the first long digit run
    m = re.search(r"\d{6,}", t)
    if m:
        return m.group(0), None
    return t, None  # last resort

def add_item_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    item_ids = []
    var_ids = []
    for val in df["target_id"]:
        iid, vid = parse_target_id(val)
        item_ids.append(iid)
        var_ids.append(vid)
    df["item_id"] = item_ids
    df["variation_id"] = var_ids
    return df

def pick_winner_minimal(rows_df: pd.DataFrame) -> str:
    """
    Minimal, robust winner:
    score = MRR + small bump for coverage across variants.
    Requires columns: item_id, variant, rank.
    """
    df = rows_df.copy()
    # ensure numeric ranks
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    g = df.groupby("item_id", dropna=False)
    mrr = g["rank"].apply(lambda s: np.mean(1.0 / s))
    coverage = g["variant"].nunique()
    best_rank = g["rank"].min()

    agg = pd.DataFrame({"mrr": mrr, "coverage": coverage, "best_rank": best_rank}).reset_index()
    agg["score"] = agg["mrr"] + 0.02 * agg["coverage"]
    winner = agg.sort_values(by=["score", "best_rank"], ascending=[False, True]).iloc[0]
    return str(winner["item_id"])

# new functions here
# ---------- helpers ----------
WORD_RE = re.compile(r"[A-Za-z0-9%+\-]+")  # keep things like 2600mAh, USB-C, 12V
TOK_RE  = re.compile(r"[A-Za-z0-9%+\-]+|[^A-Za-z0-9%+\-\s]+")

def tokenize_words(text: str):
    return [m.group(0) for m in WORD_RE.finditer(text)]

def build_token_scores(top_imp: pd.DataFrame, mode: str = "sum") -> dict[str, float]:
    """
    Map tokens to a lift score from token/n-gram features.

    Parameters
    ----------
    top_imp : DataFrame with columns ['feature', 'lift']
    mode : 'sum', 'avg', 'max', 'min'
        How to combine multiple lifts for the same token.

    Returns
    -------
    dict[str, float] : token -> combined lift
    """
    token_map: dict[str, list[float]] = {}

    for feat, lift in zip(top_imp["feature"], top_imp["lift"]):
        words = str(feat).lower().split()
        for w in words:
            token_map.setdefault(w, []).append(lift)

    scores: dict[str, float] = {}
    for tok, lifts in token_map.items():
        if mode == "sum":
            scores[tok] = sum(lifts)
        elif mode == "avg":
            scores[tok] = sum(lifts) / len(lifts)
        elif mode == "max":
            scores[tok] = max(lifts)
        elif mode == "min":
            scores[tok] = min(lifts)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    return scores


def highlight_title_html(title: str, scores: dict[str, float]) -> str:
    """
    Return HTML with each word wrapped in a span colored by lift:
      >0 = green badge, <0 = red badge, =0/unknown = neutral
    Keeps punctuation in place.
    """
    pieces = []
    for tok in TOK_RE.findall(title):
        raw = tok
        low = tok.lower()
        if WORD_RE.fullmatch(tok):
            lift = scores.get(low)
            if lift is None or abs(lift) < 1e-12:
                style = "background:#2a2f3a;color:#e6e6e6;border-radius:6px;padding:0 4px;"
                title_attr = "Neutral importance"
            elif lift > 0:
                # green-ish for dark mode
                style = "background:#0f5132;color:#d1fae5;border-radius:6px;padding:0 4px;"
                title_attr = f"Positive lift: {lift:.3f}"
            else:
                style = "background:#5c1a1a;color:#ffe4e6;border-radius:6px;padding:0 4px;"
                title_attr = f"Negative lift: {lift:.3f}"
            pieces.append(f"<span style='{style}' title='{html.escape(title_attr)}'>{html.escape(raw)}</span>")
        else:
            # punctuation/space
            pieces.append(html.escape(raw))
    return "".join(pieces)

def suggest_title(seed: str, scores: dict[str, float], max_len: int = 80) -> str:
    """
    Build a compact suggestion:
    - Start from seed words with positive lift, ranked by lift.
    - Add top missing positive tokens from the global scores.
    - Keep length <= max_len, de-duplicate.
    """
    seed_tokens = tokenize_words(seed)
    # preserve original casing where possible
    casing_map = {w.lower(): w for w in seed_tokens}

    pos_seed = sorted(
        [w for w in seed_tokens if scores.get(w.lower(), 0) > 0],
        key=lambda w: (-scores.get(w.lower(), 0), seed_tokens.index(w))
    )

    # bring in a few strong positives not already in the seed
    global_pos = [w for w, s in sorted(scores.items(), key=lambda x: -x[1]) if s > 0]
    missing_pos = [w for w in global_pos if w not in {t.lower() for t in seed_tokens}]

    # heuristic ordering: strongest seed terms first, then a couple of missing positives
    parts: list[str] = []
    used = set()

    def add_word(w: str):
        wl = w.lower()
        if wl in used:
            return
        candidate = (casing_map.get(wl, w.capitalize()))
        # try adding candidate; if too long, skip
        tentative = (" ".join(parts + [candidate])).strip()
        if len(tentative) <= max_len:
            parts.append(candidate)
            used.add(wl)

    for w in pos_seed:
        add_word(w)
        if len(" ".join(parts)) >= max_len: break

    for w in missing_pos:
        add_word(w)
        if len(" ".join(parts)) >= max_len: break

    # If we still have room, add numerics/units from seed (e.g., 2600mAh, USB)
    numericish = [w for w in seed_tokens if any(ch.isdigit() for ch in w)]
    for w in numericish:
        add_word(w)
        if len(" ".join(parts)) >= max_len: break

    # Final tidy: join and hard cut (safety)
    out = " ".join(parts).strip()
    return out[:max_len]



# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="eBay Title Optimizer", page_icon="💡", layout="wide")
st.title("🧪 eBay Title Optimizer (Streamlit)")
st.caption("Enter a seed title → generate variants → measure rank lift → find winning tokens. No fluff, just lift.")

with st.sidebar:
    st.subheader("Settings")
    MARKETPLACE_ID = st.selectbox("Marketplace", ["EBAY_GB","EBAY_US","EBAY_DE","EBAY_AU","EBAY_FR"], index=0)
    top_n = st.slider("Define exact-match cohort (top N from base)", 5, 40, 20)
    max_variants = st.slider("Max variants to test", 2, 20, 10)
    polite_delay = st.slider("Delay between calls (sec)", 0.1, 1.0, 0.5)
    st.divider()
    log_anonymous = st.checkbox("Log anonymous queries for trend detection (seed, marketplace, timestamp)", value=False)

seed = st.text_input(
    "Seed Product Title",
    value="Cordless Hair Dryer 2600mAh Portable USB Rechargeable Hair Dryer Cold Hot Wind"
)


if st.button("Run Experiment", type="primary"):

    target_ids, base_pos, base_meta, base_results = exact_match_ids(seed, top_n=top_n)

    if not target_ids:
        st.error("No items found for the base query. Try a different seed.")
        st.stop()

    # 2) Variants
    variants = generate_variants(seed, max_variants=max_variants)

    # 3) Evaluate
    #rows_df = evaluate_variants(seed, variants, target_ids, base_pos, polite_delay=polite_delay)
    rows_df = evaluate_variants(seed, variants, target_ids, base_pos, base_meta, polite_delay=polite_delay)

    lift_table = compute_token_lift(rows_df, title_col="title", min_support=3, max_doc_frac=0.6)
    scores = token_scores_from_lift(lift_table)
    
    # 4) Summaries
    summary_df = compute_variant_summary(rows_df)
    importance_df = token_importance(seed, rows_df)

    # ---- HELP POPOVER ----
    render_help_popover()

    # ---- Prep once ----
    dd  = summary_df.copy().sort_values("mean_delta", ascending=False)
    dd2 = dd.copy().sort_values("avg_abs_rank", ascending=True)
    top_imp = importance_df.head(15)

    # ---- Top row (2 columns) ----
    row1_c1, row1_c2 = st.columns(2)

    cmap = mcolors.LinearSegmentedColormap.from_list(
    "streamlit",
    ["#FF4B4B", "#6C63FF", "#FFFFFF"]
)

    with row1_c1:
        wrapped_labels = wrap_labels(dd["variant"].tolist(), words_per_line=5, max_lines=3)
        n = len(dd)
        fig_h = max(3.8, min(10, 0.45 * n + 1.6))

        fig1, ax1 = plt.subplots(figsize=(6.4, fig_h))

        # --- Background in Streamlit dark ---
        fig1.patch.set_facecolor("#0E1117")
        ax1.set_facecolor("#0E1117")


        # --- Bars with Streamlitty colours ---
        y_pos = range(n)
        bars = ax1.barh(
            y_pos,
            dd["mean_delta"],
            height=0.72,
            edgecolor="white",
            linewidth=1.2
        )

        # Smooth gradient: low = dark red, high = light pinkish
        for bar, fr in zip(bars, dd["found_rate"]):
            t = np.clip(fr/100.0, 0, 1)   # normalize 0–1
            bar.set_color(cmap(t))

        # --- Axes cleanup ---
        for sp in ["top", "right", "left", "bottom"]:
            ax1.spines[sp].set_visible(False)

        ax1.xaxis.grid(True, linestyle="--", linewidth=0.8, color="#444", alpha=0.6)
        ax1.set_axisbelow(True)

        ax1.tick_params(axis="x", colors="white", labelsize=9)
        ax1.tick_params(axis="y", length=0)

        # Y labels with white + glow
        ax1.set_yticks(list(y_pos))
        ax1.set_yticklabels(wrapped_labels, color="white", fontsize=10)
        for tl in ax1.get_yticklabels():
            tl.set_path_effects([pe.withStroke(linewidth=3, foreground="#0E1117", alpha=0.9)])

        # --- Bar-end % labels ---
        label_glow = [pe.withStroke(linewidth=3, foreground="#0E1117", alpha=0.9)]
        for y, (delta, fr) in enumerate(zip(dd["mean_delta"], dd["found_rate"])):
            x = delta + (0.35 if delta >= 0 else -0.35)
            ax1.text(
                x, y, f"{fr:.0f}%",
                va="center",
                ha="left" if delta >= 0 else "right",
                fontsize=9,
                color="white",
                path_effects=label_glow
            )

        # --- Title & labels ---
        ax1.set_xlabel("Δ vs base (↑ better)", fontsize=10, color="white")
        ax1.set_title("Variant Lift + Found-Rate", fontsize=13, color="white", pad=10)

        ax1.invert_yaxis()

        max_lines_in_labels = max((lbl.count("\n") + 1) for lbl in wrapped_labels) if wrapped_labels else 1
        left_margin = 0.22 + 0.06 * (max_lines_in_labels - 1)
        left_margin = min(0.50, left_margin)
        plt.subplots_adjust(left=left_margin, right=0.98, top=0.90, bottom=0.12)

        st.pyplot(fig1, use_container_width=True)


    # 2) Absolute Average Rank
    with row1_c2:
        # --- Prep labels ---
        wrapped_labels = wrap_labels(dd2["variant"].tolist(), words_per_line=5, max_lines=3)
        n = len(dd2)
        fig_h = max(3.8, min(10, 0.45 * n + 1.6))

        fig2, ax2 = plt.subplots(figsize=(6.4, fig_h), dpi=96)

        # --- Background (Streamlit dark) ---
        fig2.patch.set_facecolor("#0E1117")
        ax2.set_facecolor("#0E1117")

        # --- Bars ---
        y_pos = np.arange(n)
        values = dd2["avg_abs_rank"].to_numpy(dtype=float)

        bars = ax2.barh(
            y_pos,
            values,
            height=0.72,
            edgecolor="white",
            linewidth=1.2
        )

        # Streamlit-ish colormap (red -> purple -> white). Lower = better = brighter.
        # Normalize so min value is brightest (1.0), max darkest (0.0)
        vmin, vmax = float(values.min()) if n else 0.0, float(values.max()) if n else 1.0
        denom = (vmax - vmin) if vmax != vmin else 1.0
        norm_val = 1.0 - ((values - vmin) / denom)

        cmap = mcolors.LinearSegmentedColormap.from_list("streamlit", ["#FF4B4B", "#6C63FF", "#FFFFFF"])
        for bar, t in zip(bars, norm_val):
            bar.set_color(cmap(np.clip(t, 0, 1)))

        # --- Axes clean + grid ---
        for sp in ["top", "right", "left", "bottom"]:
            ax2.spines[sp].set_visible(False)

        ax2.xaxis.grid(True, linestyle="--", linewidth=0.8, color="#444", alpha=0.6)
        ax2.set_axisbelow(True)

        ax2.tick_params(axis="x", colors="white", labelsize=9)
        ax2.tick_params(axis="y", length=0)

        # --- Y labels (wrapped) with glow ---
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(wrapped_labels, color="white", fontsize=10)
        for tl in ax2.get_yticklabels():
            tl.set_path_effects([pe.withStroke(linewidth=3, foreground="#0E1117", alpha=0.95)])

        # --- Title / xlabel (white + centered) ---
        ax2.set_xlabel("Avg absolute rank (↓ better)", fontsize=10, color="white",
                    path_effects=[pe.withStroke(linewidth=3, foreground="#0E1117", alpha=0.95)])
        ax2.set_title("Absolute Rank by Variant", fontsize=13, color="white", pad=10)

        # Best at top (rank 1 near top)
        ax2.invert_yaxis()

        # --- Margins for wrapped labels ---
        max_lines = max((lbl.count("\n") + 1) for lbl in wrapped_labels) if wrapped_labels else 1
        left_margin = min(0.50, 0.22 + 0.06 * (max_lines - 1))
        plt.subplots_adjust(left=left_margin, right=0.985, top=0.90, bottom=0.12)


        # Render
        st.pyplot(fig2, use_container_width=True, dpi=96)
        plt.close(fig2)

    # ---- Bottom row (2 columns) ----
    row2_c1, row2_c2 = st.columns(2)

    # 3) Token Importance

    with row2_c1:
        # Wrap long labels
        wrapped_labels = wrap_labels(top_imp["feature"].tolist(), words_per_line=5, max_lines=3)
        n = len(top_imp)
        fig_h = max(3.8, min(10, 0.45 * n + 1.6))

        fig3, ax3 = plt.subplots(figsize=(6.4, fig_h), dpi=96)

        # --- Background (Streamlit dark) ---
        fig3.patch.set_facecolor("#0E1117")
        ax3.set_facecolor("#0E1117")

        # --- Bars ---
        y = np.arange(n)
        vals = top_imp["lift"].to_numpy(dtype=float)

        bars = ax3.barh(
            y, vals,
            height=0.72,
            edgecolor="white",
            linewidth=1.2
        )

        # Streamlit-y gradient (red -> purple -> white). Higher lift = brighter.
        vmin, vmax = (float(vals.min()) if n else 0.0, float(vals.max()) if n else 1.0)
        rng = (vmax - vmin) if vmax != vmin else 1.0
        norm = (vals - vmin) / rng  # 0..1, higher = brighter

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "streamlit", ["#FF4B4B", "#6C63FF", "#FFFFFF"]
        )
        for b, t in zip(bars, norm):
            b.set_color(cmap(np.clip(t, 0, 1)))

        # --- Axes cleanup & grid ---
        for sp in ["top", "right", "left", "bottom"]:
            ax3.spines[sp].set_visible(False)

        ax3.xaxis.grid(True, linestyle="--", linewidth=0.8, color="#444", alpha=0.6)
        ax3.set_axisbelow(True)

        ax3.tick_params(axis="x", colors="white", labelsize=9)
        ax3.tick_params(axis="y", length=0)

        # --- Y labels (wrapped) with glow ---
        ax3.set_yticks(y)
        ax3.set_yticklabels(wrapped_labels, color="white", fontsize=10)
        for tl in ax3.get_yticklabels():
            tl.set_path_effects([pe.withStroke(linewidth=3, foreground="#0E1117", alpha=0.95)])

        # --- Value labels at bar ends (optional but handy) ---
        label_glow = [pe.withStroke(linewidth=3, foreground="#0E1117", alpha=0.95)]
        for yi, v in zip(y, vals):
            ax3.text(
                v + (0.35 if v >= 0 else -0.35),
                yi,
                f"{v:.2f}",
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=9,
                color="white",
                path_effects=label_glow
            )

        # --- Titles & labels ---
        ax3.set_xlabel("Estimated lift", fontsize=10, color="white",
                    path_effects=[pe.withStroke(linewidth=3, foreground="#0E1117", alpha=0.95)])
        ax3.set_title("Token / N-gram Importance", fontsize=13, color="white", pad=10)

        # Highest at top
        ax3.invert_yaxis()

        # --- Margins for wrapped labels ---
        max_lines = max((lbl.count("\n") + 1) for lbl in wrapped_labels) if wrapped_labels else 1
        left_margin = min(0.50, 0.22 + 0.06 * (max_lines - 1))
        plt.subplots_adjust(left=left_margin, right=0.985, top=0.90, bottom=0.12)

        st.pyplot(fig3, use_container_width=True, dpi=96)
        plt.close(fig3)

    # 4) Rank Distribution Histogram
    with row2_c2:
        valid_ranks = rows_df[rows_df["rank"] < 150]["rank"]
        bins = [1, 10, 20, 30, 40, 50, 75, 100]

        fig4, ax4 = plt.subplots(figsize=(6.4, 4.8), dpi=96)

        # --- Background ---
        fig4.patch.set_facecolor("#0E1117")
        ax4.set_facecolor("#0E1117")

        # --- Histogram bars ---
        counts, edges, patches_hist = ax4.hist(
            valid_ranks,
            bins=bins,
            edgecolor="white",
            linewidth=1.2
        )

        # Streamlit-style colormap
        cmap = mcolors.LinearSegmentedColormap.from_list("streamlit", ["#FF4B4B", "#6C63FF", "#FFFFFF"])
        norm = mcolors.Normalize(vmin=0, vmax=counts.max() if len(counts) else 1)

        for c, p in zip(counts, patches_hist):
            p.set_facecolor(cmap(norm(c)))

        # --- Axes cleanup ---
        for sp in ["top", "right", "left", "bottom"]:
            ax4.spines[sp].set_visible(False)

        ax4.yaxis.grid(True, linestyle="--", linewidth=0.8, color="#444", alpha=0.6)
        ax4.set_axisbelow(True)

        ax4.tick_params(axis="x", colors="white", labelsize=9)
        ax4.tick_params(axis="y", colors="white", labelsize=9)

        # --- X ticks with custom labels ---
        ax4.set_xticks(bins)
        ax4.set_xticklabels(
            [f"1-{b}" if i == 0 else f"{bins[i-1]+1}-{b}" for i, b in enumerate(bins)],
            rotation=45,
            ha="right",
            color="white"
        )

        # --- Labels and title (white + glow) ---
        glow = [pe.withStroke(linewidth=3, foreground="#0E1117", alpha=0.95)]
        ax4.set_xlabel("Rank bucket", fontsize=10, color="white", path_effects=glow)
        ax4.set_ylabel("Frequency", fontsize=10, color="white", path_effects=glow)
        ax4.set_title("Rank Distribution", fontsize=13, color="white", pad=10, path_effects=glow)

        st.pyplot(fig4, use_container_width=True, dpi=96)
        plt.close(fig4)

    # ---- Seller Analysis ----
    st.subheader("Top Sellers Across Variants")


    seller_df = compute_seller_strength(rows_df)
    seller_df = seller_df.reset_index(drop = True)
    seller_df.index = seller_df.index + 1
    seller_df.index.name = "Seller Ranking"
    seller_df[["avg_rank", "avg_score"]] = (
    seller_df[["avg_rank", "avg_score"]]
        .round(0)
        .astype(int)
    )

    clean = add_item_id_columns(rows_df)

    winner_id = pick_winner_minimal(clean)

    # Build affiliate item URL
    best_aff_url = epn_wrap(f"https://www.ebay.co.uk/itm/{winner_id}", CAMP_ID, customid="best-listing")

    # Optional: grab the row(s) for that winner to display seller/title if you have them
    winner_rows = clean.loc[clean["item_id"] == winner_id]
    winner_seller = winner_rows["seller"].mode().iat[0] if not winner_rows["seller"].mode().empty else "—"

    st.markdown("### 🏆 Best Listing (overall)")
    st.markdown(
        f"**Item ID:** `{winner_id}`  \n"
        f"Seller: `{winner_seller}`  \n"
        f"[Open on eBay]({best_aff_url})"
    )

    # new markdown
    scores = build_token_scores(top_imp, mode="avg")

    original_html  = highlight_title_html(seed, scores)
    suggested_text = suggest_title(seed, scores, max_len=80)
    suggested_html = highlight_title_html(suggested_text, scores)

    with st.container():
        st.markdown("### 🏆 Best Listing & Title Suggestions")
        st.markdown("**Original (token-weighted):**", help="Green = positive lift; Red = negative lift; Grey = neutral/unknown.")
        st.markdown(f"<div style='font-size:1.05rem;line-height:1.7'>{original_html}</div>", unsafe_allow_html=True)

        st.markdown("**Suggested (≤ 80 chars):**")
        st.markdown(f"<div style='font-size:1.05rem;line-height:1.7'>{suggested_html}</div>", unsafe_allow_html=True)

        # tiny legend
        st.markdown(
            """
            <div style="font-size:0.85rem;color:#cbd5e1;margin-top:6px">
            <span style="background:#0f5132;color:#d1fae5;border-radius:6px;padding:0 6px;">positive</span>
            &nbsp;
            <span style="background:#5c1a1a;color:#ffe4e6;border-radius:6px;padding:0 6px;">negative</span>
            &nbsp;
            <span style="background:#2a2f3a;color:#e6e6e6;border-radius:6px;padding:0 6px;">neutral</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    col1, col2 = st.columns([1,1])

    with col1:
        st.markdown(render_sellers_table(seller_df, CAMP_ID), unsafe_allow_html=True)

    with col2:
        # --- Data (top 20, but chart title says Top 10 — tweak if needed)
        df_top = seller_df.head(20)
        sellers = df_top["seller"].tolist()
        scores = df_top["seller_score"].to_numpy(dtype=float)
        n = len(sellers)

        # Wrap long seller names
        wrapped_labels = wrap_labels(sellers, words_per_line=4, max_lines=3)

        fig, ax = plt.subplots(figsize=(6.4, max(3.8, min(10, 0.42 * n + 1.6))), dpi=96)

        # --- Background ---
        fig.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#0E1117")

        # --- Bars ---
        y = np.arange(n)
        bars = ax.barh(
            y, scores,
            height=0.72,
            edgecolor="white",
            linewidth=1.2
        )

        # Streamlit colour map (higher score = brighter)
        vmin, vmax = float(scores.min()) if n else 0.0, float(scores.max()) if n else 1.0
        rng = (vmax - vmin) if vmax != vmin else 1.0
        norm = (scores - vmin) / rng

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "streamlit", ["#FF4B4B", "#6C63FF", "#FFFFFF"]
        )
        for b, t in zip(bars, norm):
            b.set_color(cmap(np.clip(t, 0, 1)))

        # --- Axes cleanup ---
        for sp in ["top", "right", "left", "bottom"]:
            ax.spines[sp].set_visible(False)

        ax.xaxis.grid(True, linestyle="--", linewidth=0.8, color="#444", alpha=0.6)
        ax.set_axisbelow(True)

        ax.tick_params(axis="x", colors="white", labelsize=9)
        ax.tick_params(axis="y", length=0)

        # --- Y labels with glow ---
        ax.set_yticks(y)
        ax.set_yticklabels(wrapped_labels, color="white", fontsize=10)
        for tl in ax.get_yticklabels():
            tl.set_path_effects([pe.withStroke(linewidth=3, foreground="#0E1117", alpha=0.95)])

        # --- Bar-end labels (seller_score) ---
        glow = [pe.withStroke(linewidth=3, foreground="#0E1117", alpha=0.95)]
        for yi, v in zip(y, scores):
            ax.text(
                v + (0.35 if v >= 0 else -0.35),
                yi,
                f"{v:.0f}",   # no decimals, adjust if you want
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=9,
                color="white",
                path_effects=glow
            )

        # --- Labels & Title ---
        ax.set_xlabel("Seller Score", fontsize=10, color="white", path_effects=glow)
        ax.set_title("Top Sellers", fontsize=13, color="white", pad=10, path_effects=glow)

        # Put best at top
        ax.invert_yaxis()

        # --- Margins for wrapped labels ---
        max_lines = max((lbl.count("\n") + 1) for lbl in wrapped_labels) if wrapped_labels else 1
        left_margin = min(0.50, 0.22 + 0.06 * (max_lines - 1))
        plt.subplots_adjust(left=left_margin, right=0.985, top=0.90, bottom=0.12)

        st.pyplot(fig, use_container_width=True, dpi=96)
        plt.close(fig)



