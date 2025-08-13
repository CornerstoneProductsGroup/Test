from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import pdfplumber, re, os
import re as _re
from rapidfuzz import fuzz, process
from pathlib import Path
from vendor_map import normalize_key
from pypdf import PdfReader, PdfWriter

MODEL_TOKEN_PAT = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-/_\.]{1,29}")

@dataclass
class Candidate:
    value: str
    kind: str
    anchor_text: str
    anchor_dist: float
    bbox: Tuple[float, float, float, float]

def _find_band(words, x_pad=14, below_px=150):
    # Find "Model #" or "Model Number" on a single baseline and return a narrow vertical band below it.
    lower = [(i, (w.get('text','') or '').strip().lower(), w) for i, w in enumerate(words)]
    for i, txt, w0 in lower:
        if txt == "model":
            # look for "#" or "number" to the right on same baseline
            baseline = (w0.get('top',0)+w0.get('bottom',0))/2.0
            candidates = [("#", "Model #"), ("number", "Model Number")]
            for needle, label in candidates:
                nxt = None
                for _, t2, w2 in lower:
                    if abs(((w2.get('top',0)+w2.get('bottom',0))/2.0) - baseline) <= 4.0 and w2.get('x0',0) >= w0.get('x1',0) - 2 and (w2.get('x0',0)-w0.get('x1',0)) <= 120:
                        if t2 == needle:
                            nxt = w2
                            break
                if nxt is not None:
                    x0 = min(w0.get('x0',0), nxt.get('x0',0)) - x_pad
                    x1 = max(w0.get('x1',0), nxt.get('x1',0)) + x_pad
                    anchor_bottom = max(w0.get('bottom',0), nxt.get('bottom',0))
                    band_y1 = anchor_bottom + below_px
                    return (x0, x1, anchor_bottom, band_y1, label)
    return None

def _extract_model_below(words) -> Optional[Candidate]:
    band = _find_band(words)
    if not band:
        return None
    bx0, bx1, ay, by, label = band
    below = [w for w in words if ay - 0.5 <= w.get('top',0) <= by and bx0 <= w.get('x0',0) and w.get('x1',0) <= bx1]
    below.sort(key=lambda w: (w.get('top',0.0), w.get('x0',0.0)))
    for w in below:
        txt = (w.get('text') or '').strip()
        if not txt:
            continue
        low = txt.lower()
        if low in ('model', '#', 'number', 'model #', 'model number'):
            continue
        m = MODEL_TOKEN_PAT.fullmatch(txt) or MODEL_TOKEN_PAT.search(txt)
        if m:
            val = m.group(0)
            return Candidate(value=val, kind='model', anchor_text=label, anchor_dist=0.0,
                             bbox=(w.get('x0',0.0), w.get('top',0.0), w.get('x1',0.0), w.get('bottom',0.0)))
    return None

def extract_candidates_from_page(page) -> List[Candidate]:
    words = page.extract_words(extra_attrs=['size']) or []
    c = _extract_model_below(words)
    return [c] if c else []

def score_candidate(val: str, vendor_map: Dict[str,str], idx_keys) -> float:
    """Score with map-aware cleaning and canonical checks:
    - extract letter-leading subtokens (e.g., from '11UP-CLEANER-XTRA6506389')
    - exact match on normalize_key or canonical_key (A-Z0-9 only)
    - otherwise try token_set_ratio (>=90) or partial_ratio (>=95)
    """
    pattern_score = 0.9
    dict_score = 0.0

    def canon(s: str) -> str:
        # normalize_key then strip non-alphanumerics
        return _re.sub(r'[^A-Z0-9]+', '', normalize_key(s))

    # Build candidate strings
    subs = _re.findall(r"[A-Za-z][A-Za-z0-9\-/_\.]{1,29}", val or "")
    candidates = [val] + subs

    # Heuristic: UPROOT variants (UP-... vs UPROOT ...)
    extra = []
    for c in list(candidates):
        if c and c.upper().startswith("UP") and not c.upper().startswith("UPROOT"):
            rest = c[2:].lstrip(" -_/")
            extra.append("UPROOT " + rest)
    candidates += extra

    keys = idx_keys
    canon_map = { _re.sub(r'[^A-Z0-9]+', '', k): k for k in keys }

    # Exact membership (normalized)
    for c in candidates:
        nk = normalize_key(c)
        if nk in vendor_map:
            dict_score = 1.0
            return 0.55*dict_score + 0.45*pattern_score

    # Canonical membership (A-Z0-9 only)
    for c in candidates:
        ck = canon(c)
        if ck in canon_map and canon_map[ck] in vendor_map:
            dict_score = 1.0
            return 0.55*dict_score + 0.45*pattern_score

    # Fuzzy fallback: token_set_ratio then partial_ratio
    from rapidfuzz import fuzz, process
    probe = max(subs, key=len) if subs else (val or "")
    nprobe = normalize_key(probe)

    best1 = process.extractOne(nprobe, keys, scorer=fuzz.token_set_ratio)
    if best1 and best1[1] >= 90:
        dict_score = 0.9
        return 0.55*dict_score + 0.45*pattern_score

    best2 = process.extractOne(nprobe, keys, scorer=fuzz.partial_ratio)
    if best2 and best2[1] >= 95:
        dict_score = 0.9
        return 0.55*dict_score + 0.45*pattern_score

    return 0.55*dict_score + 0.45*pattern_score

    # Partial match (substring) fallback at a high bar
    from rapidfuzz import fuzz, process
    # Use the longest subtoken if present, else the original
    probe = max(subs, key=len) if subs else (val or "")
    nval = normalize_key(probe)
    best = process.extractOne(nval, idx_keys, scorer=fuzz.partial_ratio)
    if best and best[1] >= 95:
        dict_score = 0.9
    return 0.55*dict_score + 0.45*pattern_score

def choose_best_vendor(cands: List[Candidate], vendor_map: Dict[str,str], threshold=0.88):
    if not cands:
        return None, None, 0.0, None
    idx_keys = list(vendor_map.keys())

    def canon(s: str) -> str:
        return _re.sub(r'[^A-Z0-9]+', '', normalize_key(s))

    canon_map = { _re.sub(r'[^A-Z0-9]+', '', k): k for k in idx_keys }

    best = None
    best_score = -1.0
    best_vendor = None

    from rapidfuzz import fuzz, process

    for c in cands:
        s = score_candidate(c.value, vendor_map, idx_keys)
        nkey = normalize_key(c.value)
        v = vendor_map.get(nkey)

        if not v:
            # Canonical exact match
            ck = canon(c.value)
            k2 = canon_map.get(ck)
            if k2:
                v = vendor_map.get(k2)

        if not v:
            # Fuzzy options on the strongest probe
            subs = _re.findall(r"[A-Za-z][A-Za-z0-9\-/_\.]{1,29}", c.value or "")
            probe = max(subs, key=len) if subs else (c.value or "")
            nprobe = normalize_key(probe)

            m1 = process.extractOne(nprobe, idx_keys, scorer=fuzz.token_set_ratio)
            m2 = process.extractOne(nprobe, idx_keys, scorer=fuzz.partial_ratio)

            pick = None
            if m1 and m1[1] >= 90:
                pick = m1
            if (not pick) and m2 and m2[1] >= 95:
                pick = m2
            if pick:
                v = vendor_map.get(pick[0])

        if s > best_score:
            best_score = s
            best = c
            best_vendor = v

    if best_score >= threshold and best_vendor:
        return best_vendor, best, best_score, None
    else:
        return None, best, best_score, 'LOW_CONFIDENCE'


def split_pdf_to_vendors(pdf_path: str, out_dir: str, vendor_map: Dict[str,str], threshold=0.88):
    os.makedirs(out_dir, exist_ok=True)
    report_rows = []
    review_rows = []
    page_exports = {}

    rdr = PdfReader(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            cands = extract_candidates_from_page(page)
            vendor, best, score, flag = choose_best_vendor(cands, vendor_map, threshold=threshold)
            report_rows.append({
                'page': i+1,
                'vendor': vendor or '',
                'score': round(score, 3),
                'best_value': best.value if best else '',
                'best_kind': best.kind if best else '',
                'anchor': best.anchor_text if best else '',
                'flag': flag or ''
            })
            if vendor and not flag:
                page_exports.setdefault(vendor, []).append(i)
            else:
                review_rows.append({
                    'page': i+1,
                    'score': round(score,3),
                    'best_value': best.value if best else '',
                    'best_kind': best.kind if best else '',
                    'anchor': best.anchor_text if best else '',
                })

    out_pdfs = {}
    master_name = os.environ.get('HD_MASTER_NAME', 'Batch')
    for vendor, pages in page_exports.items():
        w = PdfWriter()
        for p in pages:
            w.add_page(rdr.pages[p])
        vend_dir = Path(out_dir)/vendor
        vend_dir.mkdir(parents=True, exist_ok=True)
        out_path = vend_dir / f"{master_name} {vendor}.pdf"
        with open(out_path, 'wb') as f:
            w.write(f)
        out_pdfs[vendor] = str(out_path)

    return report_rows, review_rows, out_pdfs
