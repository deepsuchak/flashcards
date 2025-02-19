# from flask import Flask, request, render_template, jsonify, send_file
# import fitz  # PyMuPDF
# from groq import Groq
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# import json
# import re

# # Load environment variables from .env file
# from dotenv import load_dotenv
# load_dotenv()

# app = Flask(__name__)

# def process_pdf(pdf_file):
#     """Process PDF using PyMuPDF and extract text."""
#     try:
#         doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
#         text = ""
#         for page in doc:
#             text += page.get_text()
#         document = Document(page_content=text, metadata={})
#         return [document]
#     except Exception as e:
#         raise Exception(f"Error processing PDF: {str(e)}")
#     finally:
#         if 'doc' in locals():
#             doc.close()

# def split_document(documents, chunk_size=2000, chunk_overlap=50):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap
#     )
#     return text_splitter.split_documents(documents)

# def generate_with_groq(prompt):
#     """Generate response using GROQ client."""
#     client = Groq(api_key=os.getenv("GROQ_API_KEY"))
#     try:
#         completion = client.chat.completions.create(
#         model="llama-3.3-70b-versatile",
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": f"{prompt}"},
#                 ],
#             }
#         ],
#         temperature=0,
#         max_completion_tokens=1024,
#         top_p=1,
#         stream=False,
#         stop=None,
#     )
#         return completion.choices[0].message.content
#     except Exception as e:
#         raise Exception(f"API call failed: {str(e)}")

# def parse_flashcard(raw_flashcard):
#     """Parse a single flashcard text into structured format."""
#     flashcard = {}
#     current_section = None
#     options_text = []
    
#     lines = raw_flashcard.strip().split("\n")
#     for line in lines:
#         line = line.strip()
#         if not line or line == "---":
#             continue
            
#         if "**Type**:" in line:
#             flashcard["type"] = line.split(":", 1)[1].strip()
#         elif "**Content**:" in line:
#             flashcard["content"] = line.split(":", 1)[1].strip()
#         elif "**Options**:" in line:
#             current_section = "options"
#             options_text.append(line.split(":", 1)[1].strip())
#         elif "**Answer**:" in line:
#             current_section = "answer"
#             flashcard["answer"] = line.split(":", 1)[1].strip()
#         elif current_section == "options":
#             options_text.append(line.strip())
    
#     # Process options for multiple choice questions
#     if flashcard.get("type", "").lower() == "multiple choice":
#         # Join all options text and split by common option patterns
#         options_str = " ".join(options_text)
#         # Extract options that start with A), B), C), etc.
#         options = re.findall(r'[A-D]\)[\s]*.+?(?=[A-D]\)|$)', options_str)
#         if not options:
#             # Try alternative format with just letters
#             options = re.findall(r'[A-D]\.[\s]*.+?(?=[A-D]\.|$)', options_str)
#         if not options:
#             # Fall back to simple splitting by common delimiters
#             options = [opt.strip() for opt in options_str.split(',') if opt.strip()]
        
#         flashcard["options"] = [opt.strip() for opt in options if opt.strip()]
#     else:
#         flashcard["options"] = []
    
#     return flashcard

# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         uploaded_files = request.files.getlist("pdf_files")
#         flashcards_by_document = {}

#         for uploaded_file in uploaded_files:
#             file_name = uploaded_file.filename
#             try:
#                 parsed_documents = process_pdf(uploaded_file)
#                 if not parsed_documents:
#                     flashcards_by_document[file_name] = {"error": f"No content was extracted from the document: {file_name}"}
#                     continue

#                 docs = split_document(parsed_documents)
#                 flashcards = []
#                 for doc in docs:
#                     chunk = doc.page_content
#                     prompt = f"""
#                         Generate diverse flashcards from the given content using the following formats:
#                         Q&A: Ask insightful questions with detailed answers.
#                         True/False: Create statements for evaluation.
#                         Fill-in-the-Blank: Omit key terms for completion.
#                         Multiple Choice: Provide a question with 3-4 options only and the correct answer.
#                         For Multiple Choice questions, format options as:
#                         A) First option
#                         B) Second option
#                         C) Third option
#                         D) Fourth option

#                         Content:
#                         {chunk}
                        
#                         Output only the flashcards in the format below. Do not include any explanation, reasoning, or meta-thinking.
#                         Format for each flashcard:
#                         ---
#                         **Type**: [Question and Answer / True/False / Fill-in-the-Blank / Multiple Choice]
#                         **Content**: [The actual flashcard content with the question, statement, or description]
#                         **Options**: [For Multiple Choice only: list options as A) Option 1, B) Option 2, etc.]
#                         **Answer**: [The answer or explanation corresponding to the flashcard]
#                         ---
#                     """
#                     try:
#                         response = generate_with_groq(prompt)
#                         # Split the response into individual flashcards
#                         raw_flashcards = response.strip().split("---")
#                         for raw_flashcard in raw_flashcards:
#                             if raw_flashcard.strip():
#                                 flashcard = parse_flashcard(raw_flashcard)
#                                 if flashcard:  # Only add if parsing was successful
#                                     flashcards.append(flashcard)
#                     except Exception as e:
#                         flashcards_by_document[file_name] = {"error": f"Error generating flashcard for {file_name}: {str(e)}"}
#                         continue

#                 flashcards_by_document[file_name] = flashcards
#             except Exception as e:
#                 flashcards_by_document[file_name] = {"error": f"An error occurred processing {file_name}: {str(e)}"}

#         # Save flashcards to a JSON file
#         with open("flashcards.json", "w") as f:
#             json.dump(flashcards_by_document, f, indent=4)

#         return jsonify(flashcards_by_document)

#     return render_template("index.html")

# @app.route("/download_flashcards", methods=["GET"])
# def download_flashcards():
#     return send_file("flashcards.json", as_attachment=True)

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, render_template, jsonify, send_file
import fitz  # PyMuPDF
from groq import Groq
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import json
import re
import concurrent.futures

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

def process_pdf(pdf_file):
    """Process PDF using PyMuPDF and extract text."""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return [Document(page_content=text, metadata={})]
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")
    finally:
        if 'doc' in locals():
            doc.close()

def split_document(documents, chunk_size=2000, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

def generate_with_groq(prompt):
    """Generate response using GROQ client."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_completion_tokens=2048
        )
        return completion.choices[0].message.content
    except Exception as e:
        raise Exception(f"API call failed: {str(e)}")

def parse_flashcard(raw_flashcard):
    """Parse a single flashcard text into structured format."""
    flashcard = {}
    current_section = None
    options_text = []
    
    lines = raw_flashcard.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line or line == "---":
            continue
            
        if "**Type**:" in line:
            flashcard["type"] = line.split(":", 1)[1].strip()
        elif "**Content**:" in line:
            flashcard["content"] = line.split(":", 1)[1].strip()
        elif "**Options**:" in line:
            current_section = "options"
            options_text.append(line.split(":", 1)[1].strip())
        elif "**Answer**:" in line:
            current_section = "answer"
            flashcard["answer"] = line.split(":", 1)[1].strip()
        elif current_section == "options":
            options_text.append(line.strip())
    
    if flashcard.get("type", "").lower() == "multiple choice":
        options_str = " ".join(options_text)
        options = re.findall(r'[A-D]\)[\s]*.+?(?=[A-D]\)|$)', options_str)
        if not options:
            options = re.findall(r'[A-D]\.[\s]*.+?(?=[A-D]\.|$)', options_str)
        if not options:
            options = [opt.strip() for opt in options_str.split(',') if opt.strip()]
        
        flashcard["options"] = [opt.strip() for opt in options if opt.strip()]
    else:
        flashcard["options"] = []
    
    return flashcard

def process_chunk(doc):
    """Process a single text chunk to generate flashcards."""
    prompt = f"""Generate diverse and high-quality flashcards based on the given content, ensuring clarity, specificity, and variety. Follow these structured formats:

                - **Q&A:** Pose insightful questions with well-explained answers.  
                - **True/False:** Create statements that challenge understanding.  
                - **Fill-in-the-Blank:** Remove key terms to reinforce recall.  
                - **Multiple Choice:** Provide a question with exactly **four options**, formatted as:  
                A) First option  
                B) Second option  
                C) Third option  
                D) Fourth option  

                ### **Instructions:**  
                - Ensure the language is **clear, precise, and easy to understand**. Avoid jargon unless the topic requires it.  
                - Use **varied question structures** to promote deeper comprehension.  
                - Keep flashcards **concise yet informative**, focusing on key concepts.  
                - Maintain the exact format below **without modifications**.  
                - Do not include explanations, reasoning, or meta-thinking—**only structured flashcards**.  

                ### **Flashcard Format:**  
                ---
                **Type**: [Question and Answer / True/False / Fill-in-the-Blank / Multiple Choice]  
                **Content**: [The actual flashcard content with the question, statement, or description]  
                **Options**: [For Multiple Choice only: list options as A) Option 1, B) Option 2, etc.]  
                **Answer**: [The answer or explanation corresponding to the flashcard]  
                ---

                **Content:**  
                {doc.page_content}
                """
    # prompt = f"""
    #     Generate diverse flashcards from the given content using the following formats:
    #     Q&A: Ask insightful questions with detailed answers.
    #     True/False: Create statements for evaluation.
    #     Fill-in-the-Blank: Omit key terms for completion.
    #     Multiple Choice: Provide a question with 3-4 options only and the correct answer.
    #     For Multiple Choice questions, format options as:
    #     A) First option
    #     B) Second option
    #     C) Third option
    #     D) Fourth option

    #     Content:
    #     {doc.page_content}
        
    #     Output only the flashcards in the format below. Do not include any explanation, reasoning, or meta-thinking.
    #     Format for each flashcard:
    #     ---
    #     **Type**: [Question and Answer / True/False / Fill-in-the-Blank / Multiple Choice]
    #     **Content**: [The actual flashcard content with the question, statement, or description]
    #     **Options**: [For Multiple Choice only: list options as A) Option 1, B) Option 2, etc.]
    #     **Answer**: [The answer or explanation corresponding to the flashcard]
    #     ---
    # """
    try:
        response = generate_with_groq(prompt)
        return [parse_flashcard(rf) for rf in response.strip().split("---") if rf.strip()]
    except Exception as e:
        return []

def process_single_file(uploaded_file):
    """Process a single PDF file with parallel chunk processing."""
    try:
        parsed_docs = process_pdf(uploaded_file)
        if not parsed_docs:
            return {"error": f"No content extracted from {uploaded_file.filename}"}
        
        docs = split_document(parsed_docs)
        flashcards = []
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_chunk, doc) for doc in docs]
            for future in concurrent.futures.as_completed(futures):
                flashcards.extend(future.result())
        
        return flashcards
    except Exception as e:
        return {"error": f"Error processing {uploaded_file.filename}: {str(e)}"}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded_files = request.files.getlist("pdf_files")
        flashcards_by_document = {}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_single_file, uf): uf.filename for uf in uploaded_files}
            for future in concurrent.futures.as_completed(futures):
                fname = futures[future]
                try:
                    flashcards_by_document[fname] = future.result()
                except Exception as e:
                    flashcards_by_document[fname] = {"error": str(e)}

        with open("flashcards.json", "w") as f:
            json.dump(flashcards_by_document, f, indent=4)

        return jsonify(flashcards_by_document)

    return render_template("index.html")

@app.route("/download_flashcards", methods=["GET"])
def download_flashcards():
    return send_file("flashcards.json", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)