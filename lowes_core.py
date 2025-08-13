from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import pdfplumber, re, os
import re as _re
from rapidfuzz import fuzz, process
from pathlib import Path
from vendor_map import normalize_key
from pypdf import PdfReader, PdfWriter

SOS_PO_PAT = re.compile(r"PO\s*#\s*[:\-]?\s*(\d{6,})", re.I)
SLIP_PO_PAT = re.compile(r"PO\s*Number\s*:\s*(\d{6,})", re.I)
def _extract_po_from_text(text: str):
    if not text:
        return None
    m = SOS_PO_PAT.search(text)
    if m:
        return m.group(1)
    m2 = SLIP_PO_PAT.search(text)
    if m2:
        return m2.group(1)
    return None
def _is_sos_page(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    # Require 'sos' plus one of these shipping markers to avoid false positives
    return (' sos' in t or '\nsos' in t) and ('ship unit count' in t or 'sscc' in t or 'sos for' in t)


MODEL_TOKEN_PAT = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-/_\.]{1,29}")

@dataclass
class Candidate:
    value: str
    kind: str
    anchor_text: str
    anchor_dist: float
    bbox: Tuple[float, float, float, float]

def _find_band(words, x_pad_left=60, x_pad_right=220, below_px=180):
    """Locate a "Model #" or "Model Number" header and return a vertical capture band below it."""
    lower = [(i, (w.get('text','') or '').strip().lower(), w) for i, w in enumerate(words)]
    for i, txt, w0 in lower:
        # accept 'model', 'model#', or 'model #' as the left token
        if txt in ('model', 'model#', 'model #'):
            baseline = (w0.get('top',0)+w0.get('bottom',0))/2.0
            for needle, label in (('#', 'Model #'), ('number', 'Model Number')):
                nxt = None
                for _, t2, w2 in lower:
                    if abs(((w2.get('top',0)+w2.get('bottom',0))/2.0) - baseline) <= 4.0                                    and w2.get('x0',0) >= w0.get('x1',0) - 2                                    and (w2.get('x0',0)-w0.get('x1',0)) <= 120:
                        if t2 == needle:
                            nxt = w2
                            break
                if nxt is not None:
                    x0 = min(w0.get('x0',0), nxt.get('x0',0)) - x_pad_left
                    x1 = max(w0.get('x1',0), nxt.get('x1',0)) + x_pad_right
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



def _page_text(page):
    try:
        t = page.extract_text() or ""
        return t
    except Exception:
        return ""

_PO_PACK_PAT = re.compile(r"PO\s*Number\s*:\s*(\d{6,})", re.IGNORECASE)
_PO_SOS_PAT = re.compile(r"PO\s*#\s*[:\-]?\s*(\d{6,})", re.IGNORECASE)

def _extract_po_packing(page) -> Optional[str]:
    txt = _page_text(page)
    m = _PO_PACK_PAT.search(txt)
    if m:
        return m.group(1)
    return None

def _extract_po_sos(page) -> Optional[str]:
    txt = _page_text(page)
    # quick SOS page check
    if "SOS" not in txt.upper():
        return None
    m = _PO_SOS_PAT.search(txt)
    if m:
        return m.group(1)
    return None

def _is_sos_page(page) -> bool:
    txt = _page_text(page).upper()
    return (" SOS" in txt or "SOS " in txt) and ("PO #" in txt or "PO#".upper() in txt)


def split_pdf_to_vendors(pdf_path: str, out_dir: str, vendor_map: Dict[str,str], threshold=0.88):
    os.makedirs(out_dir, exist_ok=True)
    report_rows = []
    review_rows = []
    page_exports = {}  # temp export list if we want to do adjacency (not needed post-pass)

    rdr = PdfReader(pdf_path)

    # First pass: collect candidates, vendor by PO, and SOS pages by PO
    vendor_by_po: Dict[str, str] = {}
    pack_pages_by_po: Dict[str, list] = {}
    sos_pages_by_po: Dict[str, list] = {}
    page_meta: Dict[int, dict] = {}

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # Detect SOS vs packing
            po_sos = _extract_po_sos(page)
            if po_sos:
                sos_pages_by_po.setdefault(po_sos, []).append(i)
                page_meta[i] = {'type': 'SOS', 'po': po_sos}
                # We'll add to report later
                continue

            # Packing page: extract model from band and PO Number
            words = page.extract_words(extra_attrs=['size']) or []
            c = _extract_model_below(words)
            cands = [c] if c else []
            vendor, best, score, flag = choose_best_vendor(cands, vendor_map, threshold=threshold)

            po = _extract_po_packing(page)
            page_meta[i] = {'type': 'PACK', 'po': po}

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
                pack_pages_by_po.setdefault(po or f'PAGE{i}', []).append(i)
                if po:
                    vendor_by_po[po] = vendor
            else:
                review_rows.append({
                    'page': i+1,
                    'score': round(score,3),
                    'best_value': best.value if best else '',
                    'best_kind': best.kind if best else '',
                    'anchor': best.anchor_text if best else '',
                })

    # Second pass: attach SOS pages by PO to vendor files
    out_pdfs: Dict[str, str] = {}
    master_name = os.environ.get('HD_MASTER_NAME', 'Batch')

    # Merge pack + SOS per PO
    handled_sos_pages = set()
    for po, pack_pages in pack_pages_by_po.items():
        vend = None
        if po and po in vendor_by_po:
            vend = vendor_by_po[po]
        elif isinstance(po, str) and po.startswith('PAGE'):
            # packing page without PO (rare): vendor already chosen
            if report_rows:
                # Find the row for this page to get vendor
                pg = int(po[4:])
                for r in report_rows:
                    if r['page'] == pg+1 and r.get('vendor'):
                        vend = r['vendor']
                        break
        if not vend:
            continue
        # pages to export: packing + any sos for this PO
        pages = list(pack_pages)
        if po and po in sos_pages_by_po:
            pages.extend(sos_pages_by_po[po])
            handled_sos_pages.update(sos_pages_by_po[po])
        pages = sorted(set(pages))
        if not pages:
            continue
        w = PdfWriter()
        for p in pages:
            w.add_page(rdr.pages[p])
        vend_dir = Path(out_dir)/vend
        vend_dir.mkdir(parents=True, exist_ok=True)
        out_path = vend_dir / f"{master_name} {vend}.pdf"
        with open(out_path, 'wb') as f:
            w.write(f)
        out_pdfs[vend] = str(out_path)

    # Any remaining SOS pages (with PO but missing packing/vendor) -> send to review
    for po, sos_pages in sos_pages_by_po.items():
        if any(p in handled_sos_pages for p in sos_pages):
            continue
        # if we can infer vendor by po from earlier, route; else review
        vend = vendor_by_po.get(po)
        if vend:
            pages = sorted(set(sos_pages))
            w = PdfWriter()
            for p in pages:
                w.add_page(rdr.pages[p])
            vend_dir = Path(out_dir)/vend
            vend_dir.mkdir(parents=True, exist_ok=True)
            out_path = vend_dir / f"{master_name} {vend}.pdf"
            # append to existing file if exists
            if vend in out_pdfs and Path(out_pdfs[vend]).exists():
                from pypdf import PdfReader as _R, PdfWriter as _W
                _r_old = _R(out_pdfs[vend])
                _w = _W()
                for pg in _r_old.pages:
                    _w.add_page(pg)
                for p in pages:
                    _w.add_page(rdr.pages[p])
                with open(out_pdfs[vend], 'wb') as f:
                    _w.write(f)
            else:
                with open(out_path, 'wb') as f:
                    w.write(f)
                out_pdfs[vend] = str(out_path)
        else:
            # add each SOS page to review with reason
            for p in sos_pages:
                review_rows.append({
                    'page': p+1,
                    'score': 0.0,
                    'best_value': '',
                    'best_kind': '',
                    'anchor': 'SOS',
                    'flag': 'SOS_UNMATCHED_PO'
                })

    # Report rows for SOS pages (optional): add info rows for transparency
    for po, sos_pages in sos_pages_by_po.items():
        for p in sos_pages:
            report_rows.append({
                'page': p+1,
                'vendor': vendor_by_po.get(po, ''),
                'score': 1.0 if po in vendor_by_po else 0.0,
                'best_value': f"PO#{po}",
                'best_kind': 'SOS',
                'anchor': 'PO #',
                'flag': '' if po in vendor_by_po else 'SOS_UNMATCHED_PO'
            })

    # Sort report by page for readability
    try:
        report_rows.sort(key=lambda r: int(r['page']))
    except Exception:
        pass

    return report_rows, review_rows, out_pdfs
