import fitz
import json
import re
from collections import Counter
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression

def _clean_text(text):
    """Normalizes text for robust comparison."""
    # Convert to lowercase and remove all non-alphanumeric characters
    cleaned = re.sub(r'[^a-z0-9\s]', '', text.lower())
    # Consolidate multiple spaces into one
    return re.sub(r'\s+', ' ', cleaned).strip()

# (Helper functions: extract_blocks, compute_vertical_gaps, group_spans_into_lines)
def extract_blocks(pdf_path):
    pdf = fitz.open(pdf_path)
    all_spans = []
    # --- BUG FIX: Page numbers must start from 1 ---
    for page_num, page in enumerate(pdf, start=1):
        try:
            # First, get the JSON string
            json_string = page.get_text("json")
            # Then, parse it into a Python dictionary
            text_dict = json.loads(json_string)
        except json.JSONDecodeError:
            continue # Skip pages that fail to parse

        for block in text_dict.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", '').strip()
                    text = re.sub(r'[\x00-\x1F\x7F-\x9F\u2000-\u200F\u2028-\u202F]', '', text)
                    if not text:
                        continue
                    
                    is_bold = "bold" in span.get("font", "").lower()
                    all_spans.append({
                        "page": page_num, 
                        "text": text, 
                        "font_size": round(span.get("size"), 2), 
                        "is_bold": is_bold, 
                        "y0": span.get("bbox")[1]
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
        new_span = span.copy()
        new_span["gap_above"] = round(gap, 2)
        new_spans.append(new_span)
        last_y0 = y0
        last_page = page
    return new_spans

def group_spans_into_lines(spans):
    lines = []
    if not spans: return lines
    current_line_spans = [spans[0]]
    for i in range(1, len(spans)):
        current_span, prev_span = spans[i], current_line_spans[-1]
        if current_span["page"] == prev_span["page"] and abs(current_span["y0"] - prev_span["y0"]) < 2:
            current_line_spans.append(current_span)
        else:
            lines.append(current_line_spans)
            current_line_spans = [current_span]
    lines.append(current_line_spans)
    
    consolidated_lines = []
    for line_spans in lines:
        full_text = " ".join(s["text"] for s in line_spans)
        first_span = line_spans[0]
        consolidated_lines.append({
            "text": full_text, "page": first_span["page"], "font_size": first_span["font_size"], 
            "is_bold": first_span["is_bold"], "y0": first_span["y0"], "gap_above": first_span["gap_above"]
        })
    return consolidated_lines

# In train.py, replace the old function with this one:

def create_training_data(pdf_path, ground_truth_json_path):
    """Creates a labeled dataset for training our model with robust, fuzzy matching."""
    print(f"Processing {os.path.basename(pdf_path)}...")
    with open(ground_truth_json_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    # Clean the ground truth text once using our new helper
    true_headings = {_clean_text(item['text']) for item in ground_truth.get('outline', [])}
    true_title = _clean_text(ground_truth.get('title', ''))
    
    spans = extract_blocks(pdf_path)
    spans_with_gaps = compute_vertical_gaps(spans)
    all_lines = group_spans_into_lines(spans_with_gaps)
    
    font_sizes = [line['font_size'] for line in all_lines if line['text']]
    if not font_sizes: return pd.DataFrame()
    body_text_size = Counter(font_sizes).most_common(1)[0][0]
    
    data = []
    for line in all_lines:
        # Clean the extracted text from the PDF in the same way
        cleaned_text = _clean_text(line['text'])
        
        # Now, the comparison is much more reliable
        is_heading = 1 if (cleaned_text in true_headings or cleaned_text == true_title) else 0
        
        features = {
            'font_size_diff': line['font_size'] - body_text_size,
            'is_bold': int(line['is_bold']),
            'gap_above': line['gap_above'],
            'word_count': len(line['text'].strip().split()),
            'ends_in_period': int(line['text'].strip().endswith('.')),
            'is_heading': is_heading
        }
        data.append(features)
        
    return pd.DataFrame(data)

def train_and_get_weights(df):
    """Trains the model and prints the weights to be hardcoded."""
    if df.empty:
        print("DataFrame is empty. Cannot train.")
        return

    features = ['font_size_diff', 'is_bold', 'gap_above', 'word_count', 'ends_in_period']
    X = df[features]
    y = df['is_heading']

    model = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
    model.fit(X, y)

    # Print the weights for us to copy
    weights = dict(zip(features, model.coef_[0]))
    print("\n--- 🧠 Learned Model Weights ---")
    print("Copy these into your extractor.py script!")
    print("\nWEIGHTS = {")
    for feature, weight in weights.items():
        print(f"    '{feature}': {weight:.4f},")
    print("}")
    print(f"\nINTERCEPT = {model.intercept_[0]:.4f}")
    print("---------------------------------")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    DATA_DIR = "1A/dataset/" 
    
    all_pdfs = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    all_dataframes = []

    print("--- 📊 Step 1: Creating Training Dataset ---")
    for pdf_file in all_pdfs:
        pdf_path = os.path.join(DATA_DIR, pdf_file)
        json_file = pdf_file.replace(".pdf", ".json")
        json_path = os.path.join(DATA_DIR, json_file)

        if os.path.exists(json_path):
            df = create_training_data(pdf_path, json_path)
            all_dataframes.append(df)
        else:
            print(f"Warning: Missing ground truth for {pdf_file}. Skipping.")
    
    if not all_dataframes:
        print("\nNo training data could be created.")
    else:
        training_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"\n✅ Successfully created a dataset with {len(training_df)} total lines.")
        
        print("\n--- 🧠 Step 2: Training Model to Find Weights ---")
        train_and_get_weights(training_df)