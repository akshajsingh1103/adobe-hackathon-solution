# pdf_structure_parser.py using pdfminer.six for semantic extraction

import argparse
import json
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar
from collections import defaultdict, Counter
import re

class PDFMinerStructureParser:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.lines = []
        self.headings = []

    def extract_lines(self):
        for page_num, layout in enumerate(extract_pages(self.pdf_path), start=1):
            for element in layout:
                if isinstance(element, LTTextContainer):
                    for text_line in element:
                        line_text = text_line.get_text().strip()
                        if not line_text:
                            continue
                        font_sizes = [char.size for char in text_line if isinstance(char, LTChar)]
                        fonts = [char.fontname for char in text_line if isinstance(char, LTChar)]
                        is_bold = any("Bold" in font for font in fonts)
                        avg_font_size = round(sum(font_sizes)/len(font_sizes), 2) if font_sizes else 0

                        self.lines.append({
                            "text": re.sub(r'\s+', ' ', line_text),
                            "page": page_num,
                            "font_size": avg_font_size,
                            "is_bold": is_bold
                        })

    def classify_headings(self):
        font_sizes = [line['font_size'] for line in self.lines]
        body_size = Counter(font_sizes).most_common(1)[0][0]

        for line in self.lines:
            score = 0
            if line['font_size'] > body_size:
                score += (line['font_size'] - body_size) * 2
            if line['is_bold']:
                score += 3
            if re.match(r'^\d+(\.\d+)*\s', line['text']):
                score += 2
            if len(line['text'].split()) < 3:
                score -= 1

            if score >= 5:
                line['type'] = 'heading'
                self.headings.append(line)
            else:
                line['type'] = 'paragraph'

    def get_title(self):
        page1 = [h for h in self.headings if h['page'] == 1]
        if page1:
            return page1[0]['text']
        elif self.headings:
            return self.headings[0]['text']
        return "Unknown"

    def format_output(self):
        title = self.get_title()
        sizes = sorted(set(h['font_size'] for h in self.headings), reverse=True)
        size_to_level = {sz: f"H{i+1}" for i, sz in enumerate(sizes)}

        outline = []
        for h in self.headings:
            if h['text'].lower() == title.lower():
                continue
            outline.append({
                "level": size_to_level.get(h['font_size'], "H3"),
                "text": h['text'],
                "page": h['page']
            })

        return {
            "title": title,
            "outline": outline
        }

    def run(self, output_path=None):
        self.extract_lines()
        self.classify_headings()
        result = self.format_output()
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input PDF")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    args = parser.parse_args()

    parser = PDFMinerStructureParser(args.input)
    result = parser.run(output_path=args.output)
    print(f"Saved output to {args.output}")
