import fitz  # PyMuPDF

doc = fitz.open(r"C:\Users\Mayank\OneDrive\Desktop\adobe\adobe-hackathon-solution\1A\dataset\file06.pdf")
with open("output.txt", "wb") as out:
    for page in doc:
        text = page.get_text().encode("utf8")
        out.write(text)
        out.write(bytes((12,)))  # form feed: new page
