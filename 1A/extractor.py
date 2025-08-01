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

def is_noise_line(text):
    # Line is mostly punctuation or contains long stretches of dots
    return bool(re.match(r"^[. \t]{5,}.*\d{1,3}$", text.strip())) or \
           bool(re.match(r"^.*\.{5,}.*\d{1,3}$", text.strip()))

# Detect underlined lines
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
                                # print(L["text"] +  " has been underlined")
                elif op == "re":                                # rectangle operator
                    x, y_top, w, y = item[1]
                    h = y - y_top
                    if h >= 0 and h < 2:                     # very short height, likely underline       
                        # rectangle’s top edge is at y+h
                        for L in page_lines:
                            if 0 <= abs(L["y1"] - y_top) < (L["font_size"]*0.3):
                                underlined.add((L["page"], L["y0"]))
                                # print(L["text"] +  " has been underlined")
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
            table_areas = [fitz.Rect(t.bbox) for t in tables]
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
                        raw_text = span.get("text", "").strip()

                        # Default: assume it's in table if intersects with any table bbox
                        in_table = any(tb.contains(bbox) for tb in table_areas)

                        # Heuristic override: looks like a sentence (likely not a table cell)
                        looks_like_sentence = (
                            raw_text and
                            raw_text[0].isupper() and
                            raw_text[-1] in ".?" and
                            len(raw_text.split()) >= 5  # at least 5 words
                        )

                        is_link_like = bool(re.search(r"(https?://|www\.|mailto:|\.com|\.org|\|)", raw_text, re.IGNORECASE))
                        is_date_like = bool(re.search(r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b", raw_text))
                        is_phone_like = bool(re.search(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,5}[-.\s]?\d{3,5}\b", raw_text))
                        is_email_like = bool(re.search(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", raw_text))
                        is_timestamp_like = bool(re.search(r"\b\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM|am|pm)?\b", raw_text))
                        is_page_number = bool(re.fullmatch(r"Page\s*\d+|\d+\s*/\s*\d+|\d{1,3}", raw_text.strip()))
                        is_punct_deco = bool(re.fullmatch(r"[-=_*.•●·▪️◆■]+", raw_text.strip()))
                        is_file_path = bool(re.search(r"\w+\.(pdf|docx?|pptx?|xls|txt|csv)", raw_text, re.IGNORECASE))

                        if any([
                            is_link_like, is_date_like, is_phone_like, is_email_like,
                            is_timestamp_like, is_page_number, is_punct_deco, is_file_path
                        ]):
                            continue

                        # Override table flag if it's clearly a sentence and not a link
                        if in_table and (looks_like_sentence and not is_link_like):
                            in_table = False
                        is_bold = "bold" in span.get("font", "").lower() or span.get("flags", 0) & 2

                        # if (is_bold): print(text, round(span.get("size"), 2), in_table, page_num, bbox)

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
                "y1": group[0]["bbox"].y1,                   # group[0]["bbox"].y1
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

    # --- Smart header/footer removal based on text frequency, only if >=2 pages ---
    pages = {l["page"] for l in all_lines}
    if len(pages) > 2:
        # 1) Count how often each exact line text appears (excluding page 1)
        text_freq = Counter()
        page_of_text = defaultdict(set)
        for l in all_lines:
            if l["page"] == 1:
                continue
            t = l["text"].strip()
            if t:
                text_freq[t] += 1
                page_of_text[t].add(l["page"])

        # 2) Identify “repeated” texts appearing on ≥50% of pages (page 2+)
        num_pages = max(pages)
        threshold = max(2, int((num_pages - 1) * 0.5))  # at least 2 occurrences
        repeated = {t for t, cnt in text_freq.items() if cnt >= threshold}

        # 3) Filter out only those lines whose text is in `repeated`
        cleaned = []
        for l in all_lines:
            if l["page"] == 1 or l["text"].strip() not in repeated:
                cleaned.append(l)
        all_lines = cleaned
    # else: single‐page PDF → skip filtering
    return all_lines



    # --- Merge multiple lines that likely belong to same visual heading ---
def merge_lines_with_underline(all_lines, underlined_set):
    merged_lines = []
    i = 0
    while i < len(all_lines):
        current = all_lines[i]
        # if current['y0'] < 100 and current['page'] > 1:
        #     i += 1
        #     continue

        if is_noise_line(current['text']):
            i += 1
            continue
        
        combined = current.copy()

        # print(combined['text'])

        j = i + 1
        while j < len(all_lines):
            next_line = all_lines[j]
            # if (next_line['y0']<100 and next_line['page']>1):
            #     j+=1
            #     continue

            if is_noise_line(next_line['text']):
                j += 1
                continue

            gap = next_line['y0'] - current['y0']
            same_page = current['page'] == next_line['page']
            same_bold = current['is_bold'] == next_line['is_bold']
            similar_size = abs(current['font_size'] - next_line['font_size']) < 0.5

            # Allow wider gap if it's on the first page
            max_gap = 60 if current['page'] == 1 else 20
            # max_gap=20
            # print(current['page'], current['text'], max_gap)

            can_merge = (
                same_page and
                similar_size and
                same_bold and
                0 < gap < max_gap and
                current["in_table"]==next_line["in_table"]
            )


            if not can_merge:
                break

            combined['text'] += ' ' + next_line['text']
            combined['word_count'] += next_line['word_count']
            combined['ends_in_period'] = int(next_line['text'].strip().endswith("."))
            combined['is_short_line'] = int(combined['word_count'] <= 2)
            combined['in_table'] = combined['in_table'] or next_line['in_table']
            current = next_line
            j += 1
        # print(combined['text'])
        merged_lines.append(combined)
        i = j

    return merged_lines

# --- Classify Headings ---
def classify_heading_candidates(lines):
    if not lines:
        return []
    font_sizes = [l["font_size"] for l in lines if not l["in_table"]]
    if not font_sizes:
        return []

    body_size = Counter(font_sizes).most_common(1)[0][0]
    candidates = []

    for line in lines:
        if line["in_table"] or line["is_last_line"]:        # skip table lines and last lines of page
            continue
        if line['y0'] > 615:                                # skip footers
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
            candidates.append({
                "text": line["text"],
                "page": line["page"],
                "font_size": line["font_size"],
                "is_bold": (int(line["is_bold"])),
                "score": score,
                "y0": line["y0"]
            })

    unique = {}
    for c in candidates:
        key = (c["text"], c["page"])
        if key not in unique or c["score"] > unique[key]["score"]:
            unique[key] = c
    unique_candidates = list(unique.values())

    # print("\n=== Detected Heading Candidates ===")
    # for c in unique_candidates:
    #     print(f"Page {c['page']} | Size: {c['font_size']} | Score: {round(c['score'], 2)} | Text: {c['text']}")

    return unique_candidates

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
        seg_sizes = sorted({h['font_size'] for p in seg for h in headings_by_page.get(p, [])}, reverse=True)
        level_map = {s: f"H{i+1}" for i, s in enumerate(seg_sizes[:3])}
        temp = []
        for p in seg:
            for h in headings_by_page.get(p, []):
                lvl = level_map.get(h["font_size"], "H3")
                temp.append({
                "level": lvl,
                "text":  h["text"],
                "page":  h["page"],
                "score": h["score"],
                "demoted": 0,
                "font_size": h["font_size"],
                "is_bold":  int(h.get("is_bold",0)),
                "underlined": int((h["page"],h["y0"]) in underlined)
                })
        
        for level in ["H1","H2","H3"]:
            bucket = [x for x in temp if x["level"] == level]
            n = len(bucket)
            if n > 1:
                for i in range(n):
                    x1 = bucket[i]
                    for j in range(i, n):
                        x2 = bucket[j]
                        if x1["font_size"] == x2["font_size"] and x1["is_bold"] != x2["is_bold"]:
                            if x1["is_bold"] and x2["demoted"] == 0:
                                x2["level"] = f"H{min(int(level[1])+1, 3)}"         # demoted
                                x2["demoted"] = 1
                            elif x2["is_bold"]:
                                x1["level"] = f"H{min(int(level[1])+1, 3)}"
                                x1["demoted"] = 1
                                break

                        elif x1["font_size"] == x2["font_size"] and x1["underlined"] != x2["underlined"]:
                            if x1["underlined"] and x2["demoted"] == 0:
                                x2["level"] = f"H{min(int(level[1])+1, 3)}"
                                x2["demoted"] = 1
                            elif x2["underlined"]:
                                x1["level"] = f"H{min(int(level[1])+1, 3)}"
                                x1["demoted"] = 1
                                break

        outline.extend(temp)
                  
    return outline

def find_first_content_page(all_lines, min_words=1):
    page_word_counts = defaultdict(int)
    for line in all_lines:
        if not is_noise_line(line["text"]):
            page_word_counts[line["page"]] += len(line["text"].split())

    # Sort by page number and return the first with enough words
    for page in sorted(page_word_counts):
        if page_word_counts[page] >= min_words:
            return page
    return 1  # fallback if no good page is found

# --- Final Output ---
def get_final_outline(pdf_path):
    pdf = fitz.open(pdf_path)
    page_count = pdf.page_count
    pdf.close()
    page_offset = 1 if page_count > 1 else 0

    all_lines_before_merge = extract_and_group_lines(pdf_path)
    underlined = detect_underlined_lines(pdf_path, all_lines_before_merge)
    all_lines = merge_lines_with_underline(all_lines_before_merge, underlined)

    # Detect Title
    first_text_page = find_first_content_page(all_lines)
    page1_lines = [l for l in all_lines if l["page"] == first_text_page and not l["in_table"]]
    with open('output.txt', 'w', encoding='utf-8') as fout:            # DEBUGGING
        for line in page1_lines:
            fout.write(line['text'] + '\n')
    title = ""
    if page1_lines:
        # title is line with greatest font size. if tie, then upper line
        best_fit_for_title = max(page1_lines, key = lambda L: (L["font_size"], -L["y0"]))
        if best_fit_for_title["y0"] < 550:
            title = best_fit_for_title["text"]

    headings = classify_heading_candidates(all_lines)
    outline = cluster_pages_and_build_outline(headings, all_lines, underlined)

    # Filter title out
    filtered = [o for o in outline if _clean_text(o["text"]) != _clean_text(title)]
    sorted_outline = sorted(filtered, key=lambda x: x["page"])

    # taking care of page indexing
    for o in sorted_outline:
        o['page'] = o['page'] - page_offset

    cleaned_outline = [ {"level": o["level"], "text": o["text"], "page": o["page"]} for o in sorted_outline]

    return {
        "title": title,
        "outline": cleaned_outline
    }

# --- CLI Entrypoint ---
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Extract structured outline from PDF.")
#     parser.add_argument('--input', required=True, help='Input PDF path')
#     parser.add_argument('--output', required=True, help='Output JSON path')
#     args = parser.parse_args()

#     result = get_final_outline(args.input)

#     with open(args.output, 'w', encoding='utf-8') as f:
#         json.dump(result, f, indent=4, ensure_ascii=False)

#     # print(f"Extracted outline written to {args.output}")


if __name__ == '__main__':
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")

    for pdf_path in input_dir.glob("*.pdf"):
        try:
            result = get_final_outline(str(pdf_path))
            output_path = output_dir / (pdf_path.stem + ".json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            print(f"Processed {pdf_path.name} -> {output_path.name}")
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
