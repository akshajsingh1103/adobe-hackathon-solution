import json
import os
import datetime
import re
import sys
# import io
import argparse
import math
from sentence_transformers import SentenceTransformer, util

# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdout = open("debug_log.txt", "w", encoding="utf-8")  # Redirects all prints


# This tells Python to also look for files in the current script's directory (src)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now we can correctly import the functions from our 1A utility file
from extractor_utils import get_final_outline

# In 1B/src/main.py

def chunk_document_into_sections(pdf_path, title, outline, all_lines):
    """
    Groups a document's lines into sections based on ONLY H1 headings.
    The content of an H1 section runs until the next H1.
    """
    if not outline or not any(h.get('level') == 'H1' for h in outline):
        # If there are no H1s, treat the whole document as one section
        full_text = " ".join(line['text'] for line in all_lines)
        return [{"section_title": title, "page_number": 1, "content": full_text, "document_name": os.path.basename(pdf_path)}]

    sections = []
    # --- THIS IS THE FIX ---
    # First, find only the H1 headings to use as our main boundaries
    h1_headings = sorted(
        [h for h in outline if h.get('level') == 'H1'], 
        key=lambda x: (x['page'], x.get('y0', 0))
    )

    for i, current_h1 in enumerate(h1_headings):
        section_text = []
        start_page = current_h1['page']
        start_y0 = current_h1.get('y0', 0)

        # The end boundary is the start of the NEXT H1
        end_page, end_y0 = float('inf'), float('inf')
        if i + 1 < len(h1_headings):
            next_h1 = h1_headings[i+1]
            end_page = next_h1['page']
            end_y0 = next_h1.get('y0', 0)

        # Collect all lines that fall between this H1 and the next one
        for line in all_lines:
            # We don't need to check if the line is a heading, just if it's in the boundary
            is_after_start = line['page'] > start_page or (line['page'] == start_page and line.get('y0', 0) > start_y0)
            is_before_end = line['page'] < end_page or (line['page'] == end_page and line.get('y0', 0) < end_y0)
            
            if is_after_start and is_before_end:
                section_text.append(line['text'])
        
        sections.append({
            "section_title": current_h1['text'],
            "page_number": current_h1['page'],
            "content": " ".join(section_text).strip(),
            "document_name": os.path.basename(pdf_path)
        })
        
    return sections

def deconstruct_pdfs(pdf_dir):
    """Uses our 1A logic to process all PDFs and create a master list of sections."""
    print("--- Step 1: Deconstructing PDFs into sections ---")
    all_sections = []
    
    if not os.path.isdir(pdf_dir):
        print(f"FATAL ERROR: The PDF directory was not found at '{pdf_dir}'")
        return []

    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"FATAL ERROR: No PDF files found in '{pdf_dir}'")
        return []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"  - Processing {pdf_file}...")
        title, outline, all_lines = get_final_outline(pdf_path, return_all_lines=True)
        doc_sections = chunk_document_into_sections(pdf_path, title, outline, all_lines)
        all_sections.extend(doc_sections)
    
    print(f"Found {len(all_sections)} total sections across all documents.")
    return all_sections

def run_semantic_search(query, sections, model):
    """Uses a sentence transformer to find and rank the most relevant sections."""
    print("\n--- Step 2: Running Semantic Search ---")
    if not sections: return []
        
    # --- UPGRADE: Create a smarter corpus by combining titles and content ---
    corpus = [f"{s['section_title']}: {s['content']}" for s in sections if s['content']]
    corpus_map = [i for i, section in enumerate(sections) if section['content']]

    if not corpus: return []

    corpus_embeddings = model.encode(corpus, convert_to_tensor=True, show_progress_bar=True)
    query_embedding = model.encode(query, convert_to_tensor=True)
    print("\n>>> Debug: Corpus being searched against:")
    for i, c in enumerate(corpus[:5]):
        print(f"  [{i+1}] {c[:200]}...\n")

    
    # Increase top_k to get a better pool of candidates for sub-section analysis
    search_results = util.semantic_search(query_embedding, corpus_embeddings, top_k=10)[0]
    print("\n>>> Debug: Top matched sections with scores:")
    for r in search_results[:5]:
        matched_section = sections[corpus_map[r['corpus_id']]]
        print(f"  Score: {r['score']:.4f} | Section: {matched_section['section_title']} (Page {matched_section['page_number']})")
    
    ranked_sections = []
    for result in search_results:
        original_index = corpus_map[result['corpus_id']]
        section = sections[original_index]
        section['relevance_score'] = result['score']
        ranked_sections.append(section)
        
    print(f"Found {len(ranked_sections)} relevant sections.")
    return ranked_sections

def perform_subsection_analysis(query, top_sections, model):
    """For each top section, finds the most relevant 'Refined Text'."""
    print("\n--- Step 3: Performing Sub-section Analysis ---")
    
    for section in top_sections:
        content = section['content']
        sentences = re.split(r'(?<=[.!?])\s+', content)
        sentences = [s.strip() for s in sentences if len(s.split()) > 5]

        if not sentences:
            section['refined_text'] = section['content'][:500]
            continue

        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
        query_embedding = model.encode(query, convert_to_tensor=True)
        
        # --- UPGRADE: Find the top 3 sentences and join them for a better summary ---
        best_sentence_results = util.semantic_search(query_embedding, sentence_embeddings, top_k=3)[0]
        print(f"\n→ From Section: {section['section_title']} (Page {section['page_number']})")
        print("Top Sentences Picked:")
        for result in best_sentence_results:
            sent = sentences[result['corpus_id']]
            print(f"  [{result['score']:.4f}] {sent}")

        
        if best_sentence_results:
            # Sort the best sentences by their original order in the text
            top_indices = sorted([result['corpus_id'] for result in best_sentence_results])
            refined_text = " ".join(sentences[i] for i in top_indices)
            section['refined_text'] = refined_text
        else:
            section['refined_text'] = section['content'][:500]

    return top_sections

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Performs persona-driven analysis on a collection of PDFs.")
    parser.add_argument('--input_dir', default="1B/input", help='Path to the input directory')
    parser.add_argument('--output_dir', default="1B/output", help='Path to the output directory')
    args = parser.parse_args()

    PDF_DIR = os.path.join(args.input_dir, "PDFs")
    QUERY_FILE = os.path.join(args.input_dir, "challenge1b2_input.json")
    OUTPUT_FILE = os.path.join(args.output_dir, "challenge1b2_output.json")
    os.makedirs(args.output_dir, exist_ok=True)

    print("--- Loading user query ---")
    try:
        with open(QUERY_FILE, 'r', encoding='utf-8') as f:
            query_data = json.load(f)
    except FileNotFoundError:
        print(f"\nFATAL ERROR: Could not find the query file at '{QUERY_FILE}'.")
        exit()
    
    # --- UPGRADE: Robustly handle the query format ---
    persona = query_data.get("persona", "") # The key is 'persona', not 'persona_definition'
    job_to_be_done = query_data.get("job_to_be_done", {})
    if isinstance(job_to_be_done, dict):
        task = job_to_be_done.get("task", "")
    else:
        task = str(job_to_be_done)
    user_query = f"{persona}: {task}"
    # --------------------------------------------------
    
    print(f"Query: {user_query}")

    print("\n--- Loading Semantic Model (this may take a moment on first run) ---")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    all_sections = deconstruct_pdfs(PDF_DIR)
    
    if all_sections:
        top_ranked_sections = run_semantic_search(user_query, all_sections, model)
        final_results_with_subsections = perform_subsection_analysis(user_query, top_ranked_sections, model)

        # We only want the top 5 for the final output
        final_top_5 = final_results_with_subsections[:5]

        output_data = {
            "metadata": { "input_documents": [os.path.basename(f) for f in os.listdir(PDF_DIR) if f.endswith('.pdf')], "persona": persona, "job_to_be_done": job_to_be_done, "processing_timestamp": datetime.datetime.now().isoformat() },
            "extracted_sections": [], "subsection_analysis": []
        }

        for rank, section in enumerate(final_top_5, start=1):
            output_data["extracted_sections"].append({ "document": section["document_name"], "page_number": section["page_number"], "section_title": section["section_title"], "importance_rank": rank })
            output_data["subsection_analysis"].append({ "document": section["document_name"], "refined_text": section.get("refined_text", ""), "page_number": section["page_number"] })

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        
        print(f"\n✅ Round 1B processing complete. Output saved to {OUTPUT_FILE}")
    else:
        print("\n--- No sections found to process. Halting execution. ---")
