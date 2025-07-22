import fitz
import argparse
import json


def extract_blocks(pdf_path):
    """
    Extracts text spans from each page of the PDF, returning a list of span records.
    record: {page, text, font_size, is_bold, bbox, y0 (top y-coord)}
    """
    pdf = fitz.open(pdf_path)
    all_spans = []

    for page_num, page in enumerate(pdf, start=1):
        text_dict = page.get_text("dict")
        page_spans = []
        for block in text_dict.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", '').strip()
                    if not text:
                        continue

                    size = span.get("size")
                    flags = span.get("flags")             # bold, italic, underline, etc
                    bbox = span.get("bbox")               # bounding box coords

                    is_bold = bool(flags & 2)             # flag bit 2 - bold
                    y0 = bbox[1]
                    page_spans.append({
                        "page": page_num,
                        "text": text,
                        "font_size": size,
                        "is_bold": is_bold,
                        "bbox": bbox,
                        "y0": y0
                    })

        # Sort by vertical position (y0 ascending)
        page_spans.sort(key=lambda s: s["y0"])
        all_spans.extend(page_spans)

    return all_spans


def compute_vertical_gaps(spans):
    """
    Given a list of spans sorted by page and y0, compute the vertical gap to the previous span.
    Returns a new list of spans with an added 'gap_above' field.
    """
    new_spans = []
    last_y0 = None
    last_page = None

    for span in spans:
        page = span["page"]
        y0 = span["y0"]
        if last_page == page and last_y0 is not None:           # same page
            gap = y0 - last_y0
        else:                              # no gaps between spans of different pages
            gap = 0

        new_span = span.copy()
        new_span["gap_above"] = gap
        new_spans.append(new_span)
        last_y0 = y0
        last_page = page

    return new_spans


def main(input_pdf, output_json):
    spans = extract_blocks(input_pdf)
    spans_with_gaps = compute_vertical_gaps(spans)

    # saving spans to json
    with open(output_json, 'w', encoding='utf-8') as file:
        json.dump(spans_with_gaps, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and store PDF text spans with layout features")

    # parsing input and output file paths
    parser.add_argument("--input", required=True, help="Input PDF file path")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    args = parser.parse_args()
    main(args.input, args.output)