import fitz  # PyMuPDF
import argparse
import json
import re
from collections import Counter
import numpy as np
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering
from pathlib import Path

# --- Data Extraction & Grouping ---
def extract_spans(pdf_path):
    """Extracts and cleans all text spans from the PDF."""
    doc = fitz.open(pdf_path)
    all_spans = []
    for page_num, page in enumerate(doc, start=1):
        try:
            text_dict = json.loads(page.get_text("json"))
            print(f"Page {page_num} block types:", set(block.get("type") for block in text_dict.get("blocks", [])))
        except json.JSONDecodeError:
            continue
        
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            # 💡 Detect if it's table-like
            lines = block.get("lines", [])
            if len(lines) >= 2:
                spans_per_line = [len(line.get("spans", [])) for line in lines]
                if all(count >= 3 for count in spans_per_line):  # heuristic: 3+ spans per line
                    continue

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    
                    text = span.get("text", '').strip()
                    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
                    if not text:
                        continue
                    
                    font_name = span.get("font", "").lower()
                    is_bold = 'bold' in font_name or bool(span.get("flags", 0) & 2)
                    
                    all_spans.append({
                        'page': page_num, 'text': text, 'font_size': round(span.get("size"), 2),
                        'is_bold': is_bold, 'bbox': span.get("bbox"), 'y0': span.get("bbox")[1]
                    })
    doc.close()
    return all_spans

def group_spans_into_lines(spans):
    """Groups text spans into consolidated lines."""
    if not spans: return []
    spans.sort(key=lambda s: (s['page'], s['y0']))
    
    lines = []
    current_line = [spans[0]]
    for span in spans[1:]:
        if span['page'] == current_line[-1]['page'] and abs(span['y0'] - current_line[-1]['y0']) < 2:
            current_line.append(span)
        else:
            lines.append(current_line)
            current_line = [span]
    lines.append(current_line)

    consolidated = []
    last_y0, last_page = None, None
    for group in lines:
        page = group[0]['page']
        y0 = group[0]['y0']
        gap = y0 - last_y0 if last_page == page and last_y0 is not None else 0
        
        consolidated.append({
            'page': page,
            'text': ' '.join(s['text'] for s in group),
            'font_size': round(sum(s['font_size'] for s in group) / len(group), 2),
            'is_bold': any(s['is_bold'] for s in group),
            'y0': y0,
            'gap_above': round(gap, 2)
        })
        last_y0, last_page = y0, page
    return consolidated

# --- Tier 1: Explicit Structure Extraction -

# --- Tier 2: Heuristic Engine ---
def score_and_classify(lines):
    """Identifies heading candidates using a rule-based scoring system."""
    if not lines: return []
    sizes = [L['font_size'] for L in lines]
    body_size = Counter(sizes).most_common(1)[0][0]
    
    candidates = []
    for L in lines:
        text = L['text']
        # High-confidence rule for numbered headings
        match = re.match(r'^(\d+(?:\.\d+)*)\s', text)
        if match:
            depth = match.group(1).count('.') + 1
            candidates.append({
                'text': text, 'page': L['page'], 'font_size': L['font_size'],
                'score': float('inf'), 'explicit_level': f'H{min(depth, 3)}'
            })
            continue

        if re.search(r'\.{5,}', text) or len(text) < 3:
            continue
            
        score = 0
        if L['font_size'] > body_size: score += (L['font_size'] - body_size) * 2
        if L['is_bold']: score += 5
        if L['gap_above'] > 12: score += 5
        
        if score > 5:
            candidates.append({
                'text': text, 'page': L['page'], 'font_size': L['font_size'], 'score': round(score, 2)
            })
    return candidates

def cluster_pages_and_build_outline(headings, lines):
    """Clusters pages by layout and determines heading levels by section."""
    if not headings: return []
    
    pages = sorted({L['page'] for L in lines})
    n_pages = len(pages)
    if n_pages <= 1: # No need to cluster a single page
        page_segments = [pages]
    else:
        # Create page "fingerprints" based on layout
        all_sizes = [L['font_size'] for L in lines]
        top_sizes = sorted(set(all_sizes), reverse=True)[:5]
        size_to_idx = {s: i for i, s in enumerate(top_sizes)}
        page_to_index = {p: i for i, p in enumerate(pages)}
        
        X = np.zeros((n_pages, len(top_sizes) + 2))
        stats = {p: {'counts': Counter(), 'total': 0, 'bold': 0} for p in pages}

        for L in lines:
            p_stats = stats[L['page']]
            p_stats['counts'][L['font_size']] += 1
            p_stats['total'] += 1
            p_stats['bold'] += int(L['is_bold'])

        for p, st in stats.items():
            i = page_to_index[p]
            total = st['total'] or 1
            for s, cnt in st['counts'].items():
                if s in size_to_idx:
                    X[i, size_to_idx[s]] = cnt / total
            X[i, len(top_sizes)] = st['bold'] / total
        
        # Build connectivity matrix to only allow adjacent pages to cluster
        conn = sparse.lil_matrix((n_pages, n_pages))
        for i in range(n_pages - 1):
            conn[i, i + 1] = 1
            conn[i + 1, i] = 1
        
        model = AgglomerativeClustering(
            n_clusters=None, metric='euclidean', linkage='average',
            distance_threshold=0.5, # This is the main tunable parameter
            connectivity=conn.tocsr()
        )
        labels = model.fit_predict(X)
        
        seg_dict = {}
        for p, lbl in zip(pages, labels):
            seg_dict.setdefault(lbl, []).append(p)
        page_segments = [sorted(seg_dict[lbl]) for lbl in sorted(seg_dict)]

    # Determine H1/H2/H3 levels for each segment
    outline = []
    headings_by_page = {p: [] for p in pages}
    for h in headings:
        headings_by_page.get(h['page'], []).append(h)

    for seg in page_segments:
        seg_sizes = sorted({h['font_size'] for p in seg for h in headings_by_page.get(p, [])}, reverse=True)
        level_map = {size: f'H{i+1}' for i, size in enumerate(seg_sizes[:3])}
        
        for p in seg:
            for h in headings_by_page.get(p, []):
                lvl = h.get('explicit_level') or level_map.get(h['font_size'])
                if lvl:
                    outline.append({'level': lvl, 'text': h['text'], 'page': p})
    return outline

def get_final_outline(pdf_path):
    """Main function to orchestrate the entire extraction process."""
    # --- Step 1: Extract all line data from the PDF ---
    spans = extract_spans(pdf_path)
    lines = group_spans_into_lines(spans)

    # --- Step 2: Determine the Document Title Visually ---
    # We find the title based on the text from page 1, regardless of other methods.
    title = Path(pdf_path).stem # Default title is the filename
    page1_lines = [h for h in lines if h['page'] == 1]
    if page1_lines:
        # The title is likely the text with the largest font size on the first page.
        # We check the top few lines to be safe.
        top_lines = sorted(page1_lines, key=lambda x: x['y0'])[:5]
        if top_lines:
            title = max(top_lines, key=lambda x: x['font_size'])['text']

    # --- Step 4: If no TOC, run the Heuristic Engine ---
    headings = score_and_classify(lines)
    
    # Build the outline and filter out the line that we identified as the title
    raw_outline = cluster_pages_and_build_outline(headings, lines)
    filtered = [o for o in raw_outline if o['text'].strip().lower() != title.strip().lower()]
    sorted_outline = sorted(filtered, key=lambda o: (o['page'], o['text']))

    return {'title': title, 'outline': sorted_outline}

# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract a structured outline from a PDF.')
    parser.add_argument('--input', required=True, help='Path to input PDF')
    parser.add_argument('--output', required=True, help='Path to output JSON')
    args = parser.parse_args()
    
    result = get_final_outline(args.input)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f'Successfully wrote outline to {args.output}')