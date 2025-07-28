import fitz  # PyMuPDF
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering

# --- Hardcoded Weights from train.py ---
WEIGHTS = {
    'font_size_diff': 0.5775,
    'is_bold': 3.0672,
    'gap_above': 0.0030,
    'word_count': -0.3946,
    'ends_in_period': -3.7670,
    'is_short_line': -1.3743,
}
INTERCEPT = 1.6011

# WEIGHTS = {
#     'font_size_diff': 0.5019,
#     'is_bold': 3.0546,
#     'gap_above': 0.0016,
#     'word_count': -0.4218,
#     'ends_in_period': -2.7970,
#     'is_short_line': -1.4046,
# }
# INTERCEPT = 1.7762

# --- Helper: Normalize text ---
def _clean_text(text):
    cleaned = re.sub(r'[^a-z0-9\s]', '', text.lower())
    return re.sub(r'\s+', ' ', cleaned).strip()

def detect_underlined_lines(pdf_path, lines):
    """
    Scan each page's drawing objects for horizontal lines immediately beneath text.
    return a set of (page, y0).
    """
    doc = fitz.open(pdf_path)
    underlined = set()
    # Group lines by page for faster lookup
    by_page = {}
    for L in lines:
        by_page.setdefault(L["page"], []).append(L)

    for page_num, page_lines in by_page.items():
        page = doc.load_page(page_num-1)
        # collect all horizontal segments
        draws = page.get_drawings()
        for d in draws:
            for item in d["items"]:
                op = item[0]
                if op == "l":                                   # line
                    (x0, y0), (x1, y1) = item[1], item[2]
                    if abs(y1 - y0) < 1.0:  # horizontal
                        # any line whose text-y0 is just above this
                        for L in page_lines:
                            if 0 < abs(y0 - L["y1"]) < (L["font_size"]*0.3):
                                underlined.add((L["page"], L["y0"]))
                                print(L["text"] +  " has been underlined")
                elif op == "re":                                # rectangle operator
                    x, y_top, w, y = item[1]
                    h = y - y_top
                    if h >= 0 and h < 2:                     # very short height, likely underline       
                        # rectangle’s top edge is at y+h
                        for L in page_lines:
                            if 0 <= abs(L["y1"] - y_top) < (L["font_size"]*0.3):
                                underlined.add((L["page"], L["y0"]))
                                print(L["text"] +  " has been underlined")
        page = None
    doc.close()
    return underlined

# --- Main extraction: line-wise grouping with table detection ---
def extract_and_group_lines(pdf_path):
    doc = fitz.open(pdf_path)
    all_lines = []

    for page_num, page in enumerate(doc, start=1):
        try:
            blocks = page.get_text("dict").get("blocks", [])
            tables = page.find_tables().tables
            table_areas = [t.bbox for t in tables]
        except Exception as e:
            print(f"Skipping page {page_num+1} due to MuPDF error: {e}")
            continue

        spans = []
        for block in blocks:
            if block.get("type") == 0:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        bbox = fitz.Rect(span.get("bbox"))
                        in_table = any(bbox.intersects(tb) for tb in table_areas)
                        is_bold = "bold" in span.get("font", "").lower() or span.get("flags", 0) & 2

                        spans.append({
                            'text': text,
                            'page': page_num,
                            'font_size': round(span.get("size"), 2),
                            'is_bold': is_bold,
                            'bbox': bbox,
                            'y0': bbox.y0,
                            'in_table': in_table
                        })

        if not spans:
            continue

        # Sort spans by vertical position
        spans.sort(key=lambda x: x["y0"])
        lines_on_page = []
        current_line_spans = [spans[0]]
        for span in spans[1:]:
            if abs(span["y0"] - current_line_spans[-1]["y0"]) < 2:
                current_line_spans.append(span)
            else:
                lines_on_page.append(current_line_spans)
                current_line_spans = [span]
        lines_on_page.append(current_line_spans)

        last_y0 = None
        for group in lines_on_page:
            y0 = group[0]['y0']
            gap = y0 - last_y0 if last_y0 is not None else 0
            last_y0 = y0

            line_text = " ".join(s["text"] for s in group)
            word_count = len(line_text.split())
            all_lines.append({
                "text": line_text,
                "page": group[0]['page'],
                "font_size": round(np.mean([s["font_size"] for s in group]), 2),
                "is_bold": any(s["is_bold"] for s in group),
                "gap_above": round(gap, 2),
                "y0": y0,
                "y1": group[0]["bbox"].y1, 
                "word_count": word_count,
                "ends_in_period": int(line_text.strip().endswith(".")),
                "is_short_line": int(word_count <= 2),
                "in_table": any(s["in_table"] for s in group)
            })

    doc.close()

    page_maxY = {}
    for line in all_lines:                                  # mark last lines
        p, y0 = line["page"], line["y0"]
        page_maxY[p] = max(page_maxY.get(p, y0), y0)

    for line in all_lines:                                  # add is_last_line feature
        line["is_last_line"] = (line["y0"] == page_maxY[line["page"]])

    return all_lines

# --- Classify Headings ---
# In 1B/src/extractor_utils.py

def classify_heading_candidates(lines):
    if not lines:
        return []
    font_sizes = [l["font_size"] for l in lines if not l["in_table"]]
    if not font_sizes:
        return []

    body_size = Counter(font_sizes).most_common(1)[0][0]
    candidates = []

    for line in lines:
        if line["in_table"] or line["is_last_line"]:
            continue
        if line['y0'] > 600:
            continue
            
        features = {
            'font_size_diff': line["font_size"] - body_size,
            'is_bold': int(line["is_bold"]),
            'gap_above': line["gap_above"],
            'word_count': line["word_count"],
            'ends_in_period': line["ends_in_period"],
            'is_short_line': line["is_short_line"],
        }
        score = INTERCEPT + sum(features[k] * WEIGHTS.get(k, 0) for k in features)
        
        if score > 1:
            # --- THIS IS THE FIX ---
            # We now include the y0 value directly in the candidate object
            candidates.append({
                "text": line["text"],
                "page": line["page"],
                "font_size": line["font_size"],
                "is_bold": line["is_bold"],
                "y0": line["y0"], # Keep the vertical position
                "score": score
            })

    return candidates

# --- Cluster and Assign H1/H2/H3 ---
def cluster_pages_and_build_outline(headings, lines, underlined):
    if not headings:
        return []

    pages = sorted({l["page"] for l in lines})
    n_pages = len(pages)
    page_to_index = {p: i for i, p in enumerate(pages)}
    all_sizes = [l["font_size"] for l in lines]
    top_sizes = sorted(set(all_sizes), reverse=True)[:5]
    size_to_idx = {s: i for i, s in enumerate(top_sizes)}

    X = np.zeros((n_pages, len(top_sizes) + 1))
    stats = {p: {'counts': Counter(), 'total': 0, 'bold': 0} for p in pages}
    for l in lines:
        p = l["page"]
        stats[p]['counts'][l["font_size"]] += 1
        stats[p]['total'] += 1
        stats[p]['bold'] += int(l["is_bold"])

    for p, s in stats.items():
        i = page_to_index[p]
        total = s['total'] or 1
        for size, count in s['counts'].items():
            if size in size_to_idx:
                X[i, size_to_idx[size]] = count / total
        X[i, len(top_sizes)] = s['bold'] / total

    # Bypass clustering if only 1 page
    if n_pages == 1:
        seg_dict = {0: [pages[0]]}
    else:
        conn = sparse.lil_matrix((n_pages, n_pages))
        for i in range(n_pages - 1):
            conn[i, i + 1] = 1
            conn[i + 1, i] = 1

        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.4,
            linkage='average',
            metric='euclidean',
            connectivity=conn.tocsr()
        )
        labels = model.fit_predict(X)
        seg_dict = defaultdict(list)
        for p, lbl in zip(pages, labels):
            seg_dict[lbl].append(p)

    page_segments = list(seg_dict.values())
    outline = []
    headings_by_page = defaultdict(list)
    for h in headings:
        headings_by_page[h['page']].append(h)

    for seg in page_segments:
        # collect all headings in this segment
        seg_heads = [h for p in seg for h in headings_by_page.get(p, [])]
        # attach underline flag
        for h in seg_heads:
            h["_underlined"] = int((h["page"], h["y0"]) in underlined)

        # sort by font_size, bold, then underline as tiebreak
        seg_heads.sort(key=lambda h: (
            h["font_size"],
            int(h.get("is_bold", 0)),
            h["_underlined"]
        ), reverse=True)

        # pick distinct (font_size, bold) groups in that order
        seen = set()
        top_groups = []
        for h in seg_heads:
            grp = (h["font_size"], int(h.get("is_bold",0)), h["_underlined"])
            if grp not in seen:
                seen.add(grp)
                top_groups.append(grp)
            if len(top_groups) == 3:
                break

        # map the top 3 groups to H1,H2,H3
        level_map = {grp: f"H{i+1}" for i, grp in enumerate(top_groups)}

        # emit
        for h in seg_heads:
            group = (h["font_size"], int(h["is_bold"]), h["_underlined"])
            lvl = level_map.get(group, "H3")
            outline.append({
                "level": lvl,
                "text": h["text"],
                "page": h["page"],
                "y0": h["y0"],
                "score": h.get("score")
            })
                  
    return outline


# --- Final Output ---
# In 1B/src/extractor_utils.py

# In 1B/src/extractor_utils.py

# In 1B/src/extractor_utils.py

def get_final_outline(pdf_path, return_all_lines=False):
    """Main function for 1A logic. Can optionally return all lines for 1B."""
    lines = extract_and_group_lines(pdf_path)
    underlined = detect_underlined_lines(pdf_path, lines)
    
    title = Path(pdf_path).stem
    page1_lines = [l for l in lines if l['page'] == 1 and not l['in_table']]
    if page1_lines:
        best_fit_for_title = max(page1_lines, key = lambda L: (L["font_size"], -L["y0"]))
        if best_fit_for_title["y0"] < 550:
            title = best_fit_for_title["text"]

    # --- THIS IS THE FIX ---
    # The candidates now correctly contain the y0 value from the start.
    # No need for the extra, buggy loop here.
    heading_candidates = classify_heading_candidates(lines)
    
    raw_outline = cluster_pages_and_build_outline(heading_candidates, lines, underlined)

    filtered = [o for o in raw_outline if _clean_text(o["text"]) != _clean_text(title)]
    
    if return_all_lines:
        return title, filtered, lines

    sorted_outline = sorted(filtered, key=lambda o: (o['page'], o.get('y0', 0)))
    for item in sorted_outline:
        item.pop('y0', None)
        item.pop('score', None) # Clean up score for final 1A output

    return {'title': title, 'outline': sorted_outline}


# --- CLI Entrypoint ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract structured outline from PDF.")
    parser.add_argument('--input', required=True, help='Input PDF path')
    parser.add_argument('--output', required=True, help='Output JSON path')
    args = parser.parse_args()

    result = get_final_outline(args.input)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"Extracted outline written to {args.output}")
