
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import pdfplumber, re, os
from rapidfuzz import fuzz, process
from pathlib import Path
from vendor_map import normalize_key

# ---- Patterns ----
MODEL_PAT = re.compile(r"[A-Z0-9][A-Z0-9\-\/\.]{1,29}", re.I)
SKU_PAT   = re.compile(r"\b\d{5,12}\b")

@dataclass
class Candidate:
    value: str
    kind: str  # model|sku
    anchor_text: str
    anchor_dist: float
    bbox: Tuple[float, float, float, float]

# ===== Home Depot: precise "Model Number" column extraction =====

def _find_model_label_band(words, x_pad=12, below_px=140):
    """Find the 'Model Number' label built from tokens 'Model' + 'Number' on the same line,
    then return a tight vertical band directly beneath that label within which the model value appears."""
    # Find any token 'Model'
    model_idxs = [i for i,w in enumerate(words) if (w.get('text','') or '').strip().lower() == 'model']
    for i in model_idxs:
        w1 = words[i]
        y_mid = (w1['top'] + w1['bottom']) / 2.0
        # Find 'Number' token to the right on same baseline
        near = [w2 for w2 in words
                if abs(((w2.get('top',0)+w2.get('bottom',0))/2.0) - y_mid) <= 4.0
                and (w2.get('text','') or '').strip().lower() == 'number'
                and w2.get('x0',0) >= w1.get('x1',0) - 2
                and (w2.get('x0',0) - w1.get('x1',0)) <= 60]
        if near:
            w2 = min(near, key=lambda w: w.get('x0',0.0))
            x0 = min(w1['x0'], w2['x0']) - x_pad
            x1 = max(w1['x1'], w2['x1']) + x_pad
            anchor_bottom = max(w1['bottom'], w2['bottom'])
            band_y1 = anchor_bottom + below_px
            return (x0, x1, anchor_bottom, band_y1)
    return None

_MODEL_EXACT_PAT = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-/_\.]{1,29}")

def _extract_model_below(words) -> Optional[Candidate]:
    band = _find_model_label_band(words)
    if not band:
        return None
    bx0, bx1, ay, by = band
    # Consider tokens within the narrow vertical band below the label
    below = [w for w in words if w.get('top',0) >= ay - 0.5 and w.get('top',0) <= by and w.get('x0',0) >= bx0 and w.get('x1',0) <= bx1]
    # Sort by vertical position, then x
    below.sort(key=lambda w: (w.get('top',0.0), w.get('x0',0.0)))
    for w in below:
        txt = (w.get('text') or '').strip()
        if not txt:
            continue
        low = txt.lower()
        if low in ('model','number','model number'):
            continue
        m = _MODEL_EXACT_PAT.fullmatch(txt) or _MODEL_EXACT_PAT.search(txt)
        if m:
            val = m.group(0)
            return Candidate(value=val, kind='model', anchor_text='Model Number', anchor_dist=0.0,
                             bbox=(w.get('x0',0.0), w.get('top',0.0), w.get('x1',0.0), w.get('bottom',0.0)))
    return None

# ===== Candidate scoring & vendor resolution =====

def score_candidate(val: str, anchor_dist: float, kind: str, vendor_map: Dict[str,str], idx_keys) -> float:
    pattern_score = 0.9 if kind == 'model' else 0.85
    dist_score = max(0.0, min(1.0, 1.0 - (anchor_dist / 180.0)))
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
    words = page.extract_words(extra_attrs=['size']) or []
    cands: List[Candidate] = []
    c = _extract_model_below(words)
    if c:
        cands.append(c)
    return cands

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
            nkey = normalize_key(c.value)
            v = vendor_map.get(nkey)
            if not v:
                match = process.extractOne(nkey, idx_keys, scorer=fuzz.token_set_ratio)
                if match and match[1] >= 92:
                    v = vendor_map.get(match[0])
            best = c
            best_vendor = v
    if best_score >= threshold and best_vendor:
        return best_vendor, best, best_score, None
    else:
        return None, best, best_score, 'LOW_CONFIDENCE'

import os as _os
def split_pdf_to_vendors(pdf_path: str, out_dir: str, vendor_map: Dict[str,str], threshold=0.88):
    master_name = _os.environ.get('HD_MASTER_NAME', 'Batch')
    os.makedirs(out_dir, exist_ok=True)
    report_rows = []
    review_rows = []
    page_exports = {}  # vendor -> list of page indices

    from pypdf import PdfReader, PdfWriter
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

    # Write preliminary per-vendor PDFs
    out_pdfs = {}
    for vendor, pages in page_exports.items():
        w = PdfWriter()
        src = PdfReader(pdf_path)
        for p in pages:
            w.add_page(src.pages[p])
        vend_dir = Path(out_dir)/vendor
        vend_dir.mkdir(parents=True, exist_ok=True)
        out_path = vend_dir / f"{master_name} {vendor}.pdf"
        with open(out_path, 'wb') as f:
            w.write(f)
        out_pdfs[vendor] = str(out_path)

    return report_rows, review_rows, out_pdfs
