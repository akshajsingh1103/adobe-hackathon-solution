import fitz  # PyMuPDF
import argparse
import json
import re
from collections import Counter

# (Your friend's excellent data extraction functions - no changes needed)
def extract_blocks(pdf_path):
    pdf = fitz.open(pdf_path)
    all_spans = []
    for page_num, page in enumerate(pdf, start=1):
        text_dict = page.get_text("dict")
        page_spans = []
        for block in text_dict.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", '').strip()
                    text = re.sub(r'[\x00-\x1F\x7F-\x9F\u200d\u200c\u200b\u202a-\u202e]', '', text)
                    if not text:
                        continue
                    
                    is_bold = "bold" in span.get("font", "").lower()
                    
                    page_spans.append({
                        "page": page_num,
                        "text": text,
                        "font_size": round(span.get("size"), 2),
                        "is_bold": is_bold,
                        "bbox": span.get("bbox"),
                        "y0": span.get("bbox")[1]
                    })
        page_spans.sort(key=lambda s: s["y0"])
        all_spans.extend(page_spans)
    pdf.close()
    return all_spans

def compute_vertical_gaps(spans):
    new_spans = []
    last_y0 = None
    last_page = None
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

# --- NEW LOGIC STARTS HERE ---

def group_spans_into_lines(spans):
    """Groups text spans into lines based on their vertical position."""
    lines = []
    if not spans:
        return lines

    current_line_spans = [spans[0]]
    for i in range(1, len(spans)):
        current_span = spans[i]
        prev_span = current_line_spans[-1]
        
        # Check if spans are on the same page and are vertically aligned (within a small tolerance)
        if current_span["page"] == prev_span["page"] and abs(current_span["y0"] - prev_span["y0"]) < 2:
            current_line_spans.append(current_span)
        else:
            lines.append(current_line_spans)
            current_line_spans = [current_span]
    lines.append(current_line_spans)

    # Consolidate grouped spans into single line objects
    consolidated_lines = []
    for line_spans in lines:
        full_text = " ".join(s["text"] for s in line_spans)
        # Use properties of the first span as representative for the line
        first_span = line_spans[0]
        consolidated_lines.append({
            "text": full_text,
            "page": first_span["page"],
            "font_size": first_span["font_size"],
            "is_bold": first_span["is_bold"],
            "y0": first_span["y0"],
            "gap_above": first_span["gap_above"]
        })
    return consolidated_lines

def classify_headings(lines):
    """Scores and classifies lines to identify headings."""
    if not lines:
        return [], None

    # Step 1: Find the most common font size to identify body text
    font_sizes = [line['font_size'] for line in lines if line['text']]
    if not font_sizes:
        return [], None
    body_text_size = Counter(font_sizes).most_common(1)[0][0]

    # Step 2: Score each line to determine if it's a heading candidate
    headings = []
    for line in lines:
        score = 0
        # Score based on font size (larger than body text is a strong indicator)
        if line["font_size"] > body_text_size:
            score += (line["font_size"] - body_text_size) * 2

        # Score for being bold
        if line["is_bold"]:
            score += 5

        # Score for having significant space above it
        if line["gap_above"] > 10:  # 10 is a heuristic, might need tuning
            score += 5

        # Lines that start with numbers (e.g., "1. Introduction") are likely headings
        if re.match(r'^\d+(\.\d+)*\s', line['text']):
            score += 10

        # If the score is high enough, consider it a heading
        if score > 5:
            headings.append({
                "text": line["text"],
                "page": line["page"],
                "font_size": line["font_size"],
                "score": score
            })
    
    return headings, body_text_size


def generate_final_json(pdf_path, headings):
    """Formats the classified headings into the required final JSON structure."""
    if not headings:
        return {"title": "Unknown", "outline": []}

    # Step 1: Identify the document title
    doc = fitz.open(pdf_path)
    title = doc.metadata.get('title', '').strip()
    doc.close()
    
    # If metadata title is missing or generic, use the highest-scored heading on page 1
    if not title or len(title) < 4:
        page1_headings = [h for h in headings if h["page"] == 1]
        if page1_headings:
            title = max(page1_headings, key=lambda x: x['score'])['text']
        else: # Fallback if no headings on page 1
             title = max(headings, key=lambda x: x['score'])['text']


    # Step 2: Determine H1, H2, H3 based on font sizes
    heading_font_sizes = sorted(list(set(h['font_size'] for h in headings)), reverse=True)
    
    size_to_level = {}
    if len(heading_font_sizes) > 0:
        size_to_level[heading_font_sizes[0]] = "H1"
    if len(heading_font_sizes) > 1:
        size_to_level[heading_font_sizes[1]] = "H2"
    if len(heading_font_sizes) > 2:
        size_to_level[heading_font_sizes[2]] = "H3"
        
    # Step 3: Build the final outline
    outline = []
    for heading in headings:
        # Exclude the title from the outline if it's found there
        if heading['text'].lower() == title.lower():
            continue
            
        font_size = heading['font_size']
        if font_size in size_to_level:
            outline.append({
                "level": size_to_level[font_size],
                "text": heading['text'],
                "page": heading['page']
            })

    return {"title": title, "outline": outline}


def main(input_pdf, output_json):
    # The main pipeline orchestrating all steps
    spans = extract_blocks(input_pdf)
    spans_with_gaps = compute_vertical_gaps(spans)
    lines = group_spans_into_lines(spans_with_gaps)
    headings, _ = classify_headings(lines)
    final_output = generate_final_json(input_pdf, headings)

    # saving the final structured JSON
    with open(output_json, 'w', encoding='utf-8') as file:
        json.dump(final_output, file, ensure_ascii=False, indent=2)
    
    print(f"Successfully created outline for {input_pdf} at {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a structured outline from a PDF.")
    parser.add_argument("--input", required=True, help="Input PDF file path")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    args = parser.parse_args()
    main(args.input, args.output)