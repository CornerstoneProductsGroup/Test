from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import pdfplumber, re, os
from rapidfuzz import fuzz, process
from pathlib import Path
from vendor_map import normalize_key

MODEL_PAT = re.compile(r"[A-Z0-9][A-Z0-9\-\/\.]{1,29}", re.I)
SKU_PAT   = re.compile(r"\b\d{5,12}\b")

# Home Depot-specific anchors with (search_width, search_height) in PDF units
ANCHORS_HD = [
    ("Model #",        260, 100,  "model"),
    ("Model#",         260, 100,  "model"),
    ("Model Number",   280, 110,  "model"),
    ("Model No.",      260, 100,  "model"),
    ("Model No",       260, 100,  "model"),
    ("Model",          240, 100,  "model"),
    ("M/N",            220, 90,   "model"),
    ("Mfr #",          240, 100,  "model"),
    ("SKU #",          240, 100,  "sku"),
    ("Store SKU #",    280, 110,  "sku"),
    ("Store SKU",      280, 110,  "sku"),
    ("SKU:",           220, 90,   "sku"),
]

@dataclass
class Candidate:
    value: str
    kind: str  # model|sku
    anchor_text: str
    anchor_dist: float
    bbox: Tuple[float, float, float, float]

def _find_anchor_boxes(words, anchor_text: str, dx: float, dy: float):
    boxes = []
    at = anchor_text.lower()
    for w in words:
        if at in w.get("text", "").lower():
            x0, y0, x1, y1 = w["x0"], w["top"], w["x1"], w["bottom"]
            boxes.append((x1, y0, x1 + dx, y0 + dy, (x0, y0, x1, y1)))
    return boxes

def _tokens_in_box(words, box):
    x0, y0, x1, y1 = box
    return [w for w in words if (x0 <= w["x0"] <= x1 and y0 <= w["top"] <= y1)]

def _match_candidates(tokens, kind: str):
    pat = MODEL_PAT if kind == "model" else SKU_PAT
    out = []
    for t in tokens:
        m = pat.search(t.get("text", ""))
        if m:
            out.append((m.group(0).strip(), t))
    return out

def _distance(ax_box, token):
    # distance between right edge of anchor and token left/top
    ax0, ay0, ax1, ay1 = ax_box
    tx, ty = token["x0"], token["top"]
    dx = max(0.0, tx - ax1)
    dy = max(0.0, ty - ay0)
    return (dx**2 + dy**2) ** 0.5

def score_candidate(val: str, anchor_dist: float, kind: str, vendor_map: Dict[str,str], idx_keys) -> float:
    # pattern score by kind
    pattern_score = 0.9 if kind == "model" else 0.85
    # distance score (closer is better)
    dist_score = max(0.0, min(1.0, 1.0 - (anchor_dist / 180.0)))
    # dictionary score
    dict_score = 0.0
    nval = normalize_key(val)
    if nval in vendor_map:
        dict_score = 1.0
    else:
        best = process.extractOne(nval, idx_keys, scorer=fuzz.token_set_ratio)
        if best and best[1] >= 92:
            dict_score = 0.85
    return 0.5*dict_score + 0.3*pattern_score + 0.2*dist_score

def extract_candidates_from_page(page) -> List[Candidate]:
    words = page.extract_words(extra_attrs=["size"]) or []
    # Home Depot packing slip: use only "Model Number" label -> value directly below
    cands: List[Candidate] = []
    c = _extract_model_below(words)
    if c:
        cands.append(c)
    return cands

def extract_all_tokens(page):
    r"""Return a list of word tokens (text, x0, top, x1, bottom)."""
    ws = page.extract_words(extra_attrs=["size"]) or []
    return ws

def dictionary_first_fallback(words, vendor_map):
    r"""Scan all tokens; if any normalized token matches a map key (exact), return a Candidate with zero distance."""
    idx_keys = set(vendor_map.keys())
    best = None
    for w in words:
        text = (w.get("text") or "").strip()
        if not text:
            continue
        # Tokenize by non-word breaks to catch model-like pieces
        parts = re.findall(r"[A-Z0-9][A-Z0-9\-_/\.]{1,29}", text, flags=re.I)
        for p in parts:
            n = normalize_key(p)
            if n in idx_keys:
                # fabricate a candidate with strong anchor-like score
                return Candidate(value=p, kind="model", anchor_text="DICT_MATCH", anchor_dist=0.0,
                                 bbox=(w.get("x0",0.0), w.get("top",0.0), w.get("x1",0.0), w.get("bottom",0.0)))
    return None


def choose_best_vendor(cands: List[Candidate], vendor_map: Dict[str,str], threshold=0.88):
    if not cands:
        return None, None, 0.0, None
    idx_keys = list(vendor_map.keys())
    best = None
    best_score = -1.0
    best_vendor = None
    for c in cands:
        s = score_candidate(c.value, c.anchor_dist, c.kind, vendor_map, idx_keys)
        if s > best_score:
            best_score = s
            # map by normalized value if possible
            nkey = normalize_key(c.value)
            v = vendor_map.get(nkey)
            if not v:
                # fuzzy resolve to a key then map
                match = process.extractOne(nkey, idx_keys, scorer=fuzz.token_set_ratio)
                if match and match[1] >= 92:
                    v = vendor_map.get(match[0])
            best = c
            best_vendor = v
    if best_score >= threshold and best_vendor:
        return best_vendor, best, best_score, None
    else:
        return None, best, best_score, "LOW_CONFIDENCE"

def split_pdf_to_vendors(pdf_path: str, out_dir: str, vendor_map: Dict[str,str], threshold=0.88):
    os.makedirs(out_dir, exist_ok=True)
    report_rows = []
    review_rows = []
    page_exports = {}  # vendor -> list of (page_num, page_obj)

    from pypdf import PdfReader, PdfWriter
    rdr = PdfReader(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            cands = extract_candidates_from_page(page)
            if not cands:
                # try dictionary-first fallback
                dfb = dictionary_first_fallback(extract_all_tokens(page), vendor_map)
                if dfb:
                    cands = [dfb]
            vendor, best, score, flag = choose_best_vendor(cands, vendor_map, threshold=threshold)
            # Collect a small debug snippet of candidate values
            _cand_vals = ", ".join(sorted({c.value for c in cands})[:5])
            # Report
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

    # Write preliminary per-vendor PDFs
    out_pdfs = {}
    for vendor, pages in page_exports.items():
        w = PdfWriter()
        src = PdfReader(pdf_path)
        for p in pages:
            w.add_page(src.pages[p])
        vend_dir = Path(out_dir)/vendor
        vend_dir.mkdir(parents=True, exist_ok=True)
        out_path = vend_dir / f"{vendor}.pdf"
        with open(out_path, 'wb') as f:
            w.write(f)
        out_pdfs[vendor] = str(out_path)

    return report_rows, review_rows, out_pdfs
