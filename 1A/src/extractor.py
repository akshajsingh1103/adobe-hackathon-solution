import fitz  # PyMuPDF
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering

# --- Data Extraction & Grouping (No changes needed here) ---
def extract_and_group_lines(pdf_path):
    """Extracts and consolidates all text lines from the PDF with their properties."""
    doc = fitz.open(pdf_path)
    all_lines = []
    for page_num, page in enumerate(doc, start=1):
        page_width = page.rect.width
        try:
            blocks = page.get_text("dict").get("blocks", [])
        except Exception: continue
            
        spans = []
        for block in blocks:
            if block.get("type") == 0:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", '').strip()
                        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
                        if not text: continue
                        
                        font_name = span.get("font", "").lower()
                        is_bold = 'bold' in font_name or bool(span.get("flags", 0) & 2)
                        
                        spans.append({
                            'page': page_num, 'text': text, 'font_size': round(span.get("size"), 2),
                            'is_bold': is_bold, 'bbox': span.get("bbox"), 'y0': span.get("bbox")[1]
                        })
        
        if not spans: continue
        spans.sort(key=lambda s: s['y0'])

        lines_on_page = []
        if spans:
            current_line_spans = [spans[0]]
            for span in spans[1:]:
                if abs(span['y0'] - current_line_spans[-1]['y0']) < 2:
                    current_line_spans.append(span)
                else:
                    lines_on_page.append(current_line_spans)
                    current_line_spans = [span]
            lines_on_page.append(current_line_spans)

        last_y0 = None
        for group in lines_on_page:
            y0 = group[0]['y0']
            gap = y0 - last_y0 if last_y0 is not None else 0
            x0 = min(s['bbox'][0] for s in group)
            x1 = max(s['bbox'][2] for s in group)
            mid_x = (x0 + x1) / 2

            all_lines.append({
                'text': ' '.join(s['text'] for s in group), 'page': page_num,
                'font_size': round(sum(s['font_size'] for s in group) / len(group), 2),
                'is_bold': any(s['is_bold'] for s in group), 'gap_above': round(gap, 2),
                'mid_x': mid_x, 'page_width': page_width,
                'y0': y0  # BUG FIX: Added the y0 key back
            })
            last_y0 = y0
    doc.close()
    return all_lines

# --- UPGRADED CLASSIFIER USING TRAINED WEIGHTS ---
def classify_heading_candidates(lines):
    """Identifies heading candidates using our pre-trained, hardcoded weights."""
    # --- PASTE THE LATEST, BEST WEIGHTS FROM YOUR train.py SCRIPT HERE ---
    WEIGHTS = {
        'font_size_diff': 0.6218,
        'is_bold': 3.1237,
        'gap_above': 0.0020,
        'word_count': -0.2557,
        'ends_in_period': -3.9561,
    }
    INTERCEPT = 0.4064
    # -----------------------------------------------------------------

    if not lines: return []
    font_sizes = [L['font_size'] for L in lines]
    if not font_sizes: return []
    body_size = Counter(font_sizes).most_common(1)[0][0]
    
    candidates = []
    for line in lines:
        text = line['text']
        features = {
            'font_size_diff': line['font_size'] - body_size,
            'is_bold': int(line['is_bold']),
            'gap_above': line['gap_above'],
            'word_count': len(text.split()),
            'ends_in_period': int(text.endswith('.'))
        }
        
        score = INTERCEPT
        for feature, weight in WEIGHTS.items():
            score += features.get(feature, 0) * weight
        
        if score > 0:
            candidates.append({'text': text, 'page': line['page'], 'font_size': line['font_size'], 'score': score})
            
    return candidates

# --- THE CLUSTERING ENGINE FOR STRUCTURAL ANALYSIS ---
def cluster_pages_and_build_outline(headings, lines):
    """Clusters pages by layout and determines heading levels by section."""
    if not headings: return []
    
    pages = sorted({L['page'] for L in lines})
    n_pages = len(pages)
    if n_pages <= 1:
        page_segments = [pages]
    else:
        # Create page "fingerprints"
        all_sizes = [L['font_size'] for L in lines]
        top_sizes = sorted(set(all_sizes), reverse=True)[:5]
        size_to_idx = {s: i for i, s in enumerate(top_sizes)}
        page_to_index = {p: i for i, p in enumerate(pages)}
        
        X = np.zeros((n_pages, len(top_sizes) + 1)) # Features: font distribution + bold ratio
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
        
        conn = sparse.lil_matrix((n_pages, n_pages))
        for i in range(n_pages - 1):
            conn[i, i + 1] = 1
            conn[i + 1, i] = 1
        
        model = AgglomerativeClustering(
            n_clusters=None, metric='euclidean', linkage='average',
            distance_threshold=0.4, # Tunable parameter: lower means more, smaller sections
            connectivity=conn.tocsr()
        )
        labels = model.fit_predict(X)
        
        seg_dict = {}
        for p, lbl in zip(pages, labels):
            seg_dict.setdefault(lbl, []).append(p)
        page_segments = [sorted(seg_dict[lbl]) for lbl in sorted(seg_dict)]

    # Determine H1/H2/H3 levels for each segment
    outline = []
    headings_by_page = defaultdict(list)
    for h in headings:
        headings_by_page[h['page']].append(h)

    for seg in page_segments:
        seg_sizes = sorted({h['font_size'] for p in seg for h in headings_by_page.get(p, [])}, reverse=True)
        level_map = {size: f'H{i+1}' for i, size in enumerate(seg_sizes[:3])}
        
        for p in seg:
            for h in headings_by_page.get(p, []):
                lvl = level_map.get(h['font_size'])
                if lvl:
                    outline.append({'level': lvl, 'text': h['text'], 'page': p})
    return outline

def get_final_outline(pdf_path):
    """Main function orchestrating the entire extraction process."""
    all_lines = extract_and_group_lines(pdf_path)
    
    # Determine Title visually from the top of the first page
    title = Path(pdf_path).stem
    page1_lines = [l for l in all_lines if l['page'] == 1]
    if page1_lines:
        top_lines = sorted(page1_lines, key=lambda x: x['y0'])[:5]
        if top_lines:
            title = max(top_lines, key=lambda x: x['font_size'])['text']

    # Use our trained model to find all heading candidates
    heading_candidates = classify_heading_candidates(all_lines)
    
    # Use clustering to build the final outline and assign levels
    raw_outline = cluster_pages_and_build_outline(heading_candidates, all_lines)
    
    # Filter out the title and sort
    filtered = [o for o in raw_outline if o['text'].strip().lower() != title.strip().lower()]
    sorted_outline = sorted(filtered, key=lambda o: o['page'])

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
