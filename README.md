# adobe-hackathon-solution

First code commit(rishi): to run extractor.py, you have to mention the pathnames of the input pdf and output json. Use the following command:
python .\1A\src\extractor.py --input .\1A\dataset\<pdf file>.pdf --output .\1A\output\<json file>.json4

---

# Second Commit(akshaj)
## How to Run

This project uses a Python script to extract a structured outline from a PDF file.

1.  **Clone the Repository**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Set Up The Environment**
    Create and activate a virtual environment.
    ```powershell
    # Create the venv
    python -m venv venv
    # Activate on Windows PowerShell
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies**
    Install the required libraries from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You can create this file by running `pip freeze > requirements.txt` while your venv is active.)*

4.  **Execute the Script**
    To process a PDF and generate the final structured outline, run the following command from the project's root directory.

    ```bash
    python 1A/src/extractor.py --input "1A/dataset/South of France - Cities.pdf" --output "1A/output/final_outline.json"
    ```
    This will read the input PDF i.e. pdfname.pdf from the destination and create the `final_outline.json` file in the output folder.
    ---
