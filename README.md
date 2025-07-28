# 🚀 Round 1A: PDF Outline Extractor

A self-contained command-line tool that ingests PDFs, detects titles and hierarchical headings (H1–H3), and outputs structured JSON outlines.

---

## 🧠 Our Approach

- **Data Extraction:** Uses PyMuPDF to extract all text lines, capturing features like font size, boldness, position, and word count.
- **Feature Engineering:** Contextual features such as font size difference and layout cues are computed.
- **Data-Driven Classification:** A lightweight Logistic Regression model (weights predicted using logistic regression and then hardcoded) predicts headings without runtime model loading.
- **Hierarchical Structuring:** Clusters pages by layout and assigns heading levels based on font size relative to section.

---

## ⚙️ How to Run

### Local Execution

Setup:

~~~bash
cd 1A
python -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows
pip install -r requirements.txt
~~~

Run (for a single PDF):

~~~bash
python extractor.py --input path/to/your.pdf --output path/to/output.json
~~~

**Note:** The CLI mode is currently commented out. By default, the script runs in batch mode processing all PDFs in `/app/input` and outputs JSONs to `/app/output` (for Docker usage).

---

### Docker Execution

The container expects PDFs mounted inside `/app/input` and writes JSONs to `/app/output`.

Build the image:

~~~bash
cd 1A
docker build --platform linux/amd64 -t adobe-hackathon-1a .
~~~

Run the container:

~~~bash
docker run --rm \
  -v $(pwd)/path/to/your/input_folder:/app/input \
  -v $(pwd)/path/to/your/output_folder:/app/output \
  --network none \
  adobe-hackathon-1a
~~~

Replace `path/to/your/input_folder` and `path/to/your/output_folder` with your actual paths.

---

## 🗂️ Repository Structure
~~~
1A/
├── Dockerfile
├── requirements.txt
└── extractor.py
~~~
---

## Notes

- The script creates JSON outline files named after each PDF in the output directory.
- Input and output directories **are not created by the code**; they must exist before running.
- The Docker container requires you to mount your local folders appropriately.

---
---

## 🗂️ Repository Structure


# Round 1B: Semantic Section Retriever

This module performs persona-driven semantic analysis on a collection of PDFs by extracting and ranking the most relevant sections based on a natural language query.

---

## 🧠 Our Approach

This solution builds directly on the 1A extractor to create a powerful, multi-stage semantic search engine.

- **PDF Deconstruction:** Processes every PDF in the collection, breaking them down into a master list of logical sections (a heading plus all the text that follows it).
- **Semantic Embedding:** Uses the sentence-transformers library with the `all-MiniLM-L6-v2` model to convert the user's query and every document section into embeddings.
- **Relevance Ranking:** Performs semantic search using cosine similarity to find document sections closest to the user's query.
- **Sub-section Analysis:** For top-ranked sections, identifies and extracts the 3 most relevant sentences combined as the "Refined Text."

---

## ⚙️ How to Run 1B

### Local Execution

#### Setup:

~~~ bash
# Navigate to the 1B directory
cd 1B

# Create and activate a virtual environment
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install sentence-transformers torch  # If not included in requirements.txt
~~~

#### Prepare Input:

- Manually create the input directory and PDFs folder:

~~~ bash
mkdir -p input/PDFs
~~~

- Place **all your PDF files** inside the `1B/input/PDFs/` folder.
- Place your **query file** as `1B/input/challenge1b_input2.json`.

> **Note:** The script expects the input folder structure as `/app/input/PDFs/` and the query file inside `/app/input/` named `challenge1b_input2.json`.

- The output directory `1B/output` will be created automatically by the script during runtime.

#### Run:

~~~ bash
# From the root of the repository or inside the 1B directory
python src/main.py --input_dir input --output_dir output
~~~

The results will be saved to:

~~~
1B/output/challenge1b2_output.json
~~~

---

### Docker Execution

The Docker container expects input in `/app/input` and outputs results to `/app/output`.

#### Build the Image:

~~~ bash
# From inside the 1B directory
docker build --platform linux/amd64 -t adobe-hackathon-1b .
~~~

#### Run the Container:

Replace `path/to/your/Collection1` with the absolute path to your input folder, and `path/to/your/output_folder` with your desired output path:

~~~ bash
docker run --rm \
  -v /absolute/path/to/Collection1:/app/input \
  -v /absolute/path/to/output_folder:/app/output \
  --network none \
  adobe-hackathon-1b
~~~

- Your input folder must have this structure:

~~~
input/
├── PDFs/                     # PDF files here
└── challenge1b_input2.json   # Query JSON file here
~~~

- After running, check the output JSON here:

~~~
output/challenge1b2_output.json
~~~

---

### Folder Structure Summary

~~~
1B/
├── Dockerfile
├── requirements.txt
├── src/
│   ├── extractor_utils.py
│   ├── main.py
├── input/                     # YOU must create and populate before running
│   ├── PDFs/                  # PDF files go here
│   └── challenge1b_input2.json
└── output/                    # Created automatically by the script
~~~
