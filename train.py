import fitz
import json
import re
from collections import Counter
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def _clean_text(text):
    cleaned = re.sub(r'[^a-z0-9\s]', '', text.lower())
    return re.sub(r'\s+', ' ', cleaned).strip()

def extract_blocks(pdf_path):
    pdf = fitz.open(pdf_path)
    all_spans = []
    for page_num, page in enumerate(pdf, start=1):
        try:
            text_dict = json.loads(page.get_text("json"))
            tables = page.find_tables().tables
            table_bboxes = [t.bbox for t in tables]
        except Exception:
            continue

        for block in text_dict.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", '').strip()
                    text = re.sub(r'[\x00-\x1F\x7F-\x9F\u2000-\u200F\u2028-\u202F]', '', text)
                    if not text:
                        continue
                    bbox = fitz.Rect(span.get("bbox"))
                    is_bold = "bold" in span.get("font", "").lower()
                    in_table = any(bbox.intersects(tb) for tb in table_bboxes)

                    all_spans.append({
                        "page": page_num,
                        "text": text,
                        "font_size": round(span.get("size"), 2),
                        "is_bold": is_bold,
                        "y0": bbox.y0,
                        "in_table": in_table
                    })
    pdf.close()
    return all_spans

def compute_vertical_gaps(spans):
    new_spans = []
    last_y0 = None
    last_page = None
    spans.sort(key=lambda s: (s["page"], s["y0"]))
    for span in spans:
        page = span["page"]
        y0 = span["y0"]
        gap = y0 - last_y0 if last_page == page and last_y0 is not None else 0
        span["gap_above"] = round(gap, 2)
        new_spans.append(span)
        last_y0 = y0
        last_page = page
    return new_spans

def group_spans_into_lines(spans):
    lines = []
    if not spans: return lines
    current_line_spans = [spans[0]]
    for i in range(1, len(spans)):
        if spans[i]["page"] == spans[i - 1]["page"] and abs(spans[i]["y0"] - spans[i - 1]["y0"]) < 2:
            current_line_spans.append(spans[i])
        else:
            lines.append(current_line_spans)
            current_line_spans = [spans[i]]
    lines.append(current_line_spans)

    consolidated = []
    for group in lines:
        full_text = " ".join(s["text"] for s in group)
        first = group[0]
        word_count = len(full_text.split())
        consolidated.append({
            "text": full_text,
            "page": first["page"],
            "font_size": round(sum(s["font_size"] for s in group) / len(group), 2),
            "is_bold": any(s["is_bold"] for s in group),
            "gap_above": first["gap_above"],
            "y0": first["y0"],
            "in_table": any(s["in_table"] for s in group)
        })
    return consolidated

def create_training_data(pdf_path, ground_truth_json_path):
    with open(ground_truth_json_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    true_headings = {_clean_text(item['text']) for item in ground_truth.get('outline', [])}
    true_title = _clean_text(ground_truth.get('title', ''))

    spans = extract_blocks(pdf_path)
    spans_with_gaps = compute_vertical_gaps(spans)
    all_lines = group_spans_into_lines(spans_with_gaps)
    all_lines = [l for l in all_lines if not l['in_table']]

    font_sizes = [l['font_size'] for l in all_lines]
    if not font_sizes:
        return pd.DataFrame()
    body_text_size = Counter(font_sizes).most_common(1)[0][0]

    data = []
    for line in all_lines:
        cleaned = _clean_text(line['text'])
        is_heading = int(cleaned in true_headings or cleaned == true_title)
        word_count = len(line['text'].split())
        features = {
            'font_size_diff': line['font_size'] - body_text_size,
            'is_bold': int(line['is_bold']),
            'gap_above': line['gap_above'],
            'word_count': word_count,
            'ends_in_period': int(line['text'].endswith('.')),
            'is_short_line': int(word_count <= 2),
            'is_heading': is_heading
        }
        data.append(features)
    return pd.DataFrame(data)

def train_and_get_weights(df):
    if df.empty:
        print("No training data.")
        return

    features = ['font_size_diff', 'is_bold', 'gap_above', 'word_count', 'ends_in_period', 'is_short_line']
    X = df[features]
    y = df['is_heading']

    model = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
    model.fit(X, y)

    print("\nClassification Report:")
    print(classification_report(y, model.predict(X)))

    print("\n--- Learned Weights ---")
    print("WEIGHTS = {")
    for f, w in zip(features, model.coef_[0]):
        print(f"    '{f}': {w:.4f},")
    print("}")
    print(f"INTERCEPT = {model.intercept_[0]:.4f}")
    print("------------------------")

if __name__ == "__main__":
    DATA_DIR = "1A/dataset/"
    all_data = []

    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".pdf"):
            pdf_path = os.path.join(DATA_DIR, fname)
            json_path = pdf_path.replace(".pdf", ".json")
            if os.path.exists(json_path):
                df = create_training_data(pdf_path, json_path)
                all_data.append(df)

    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        train_and_get_weights(df)
    else:
        print("No training data found.")
