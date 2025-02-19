# import streamlit as st
# import nest_asyncio
# from openai import OpenAI
# from llama_parse import LlamaParse
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from dotenv import load_dotenv
# import json
# import os

# # Apply nest_asyncio to handle the running event loop
# nest_asyncio.apply()

# # Load environment variables from .env file
# load_dotenv()

# def initialize_parser():
#     return LlamaParse(
#         api_key=os.getenv("LLAMA_API_KEY"),
#         result_type="markdown",
#         language="en",
#         verbose=True,
#         is_formatting_instruction=False,
#         parsing_instruction= """  The document is educational material, such as a textbook, presentation, or class notes.
#         Your task is to process the document and provide concise summaries of its contents. The summaries should be structured to assist with generating flashcards. Focus on the following:

#         1. **Summarization**:
#            - Summarize each section of the document in 2-3 sentences, preserving the key points and main ideas.
#            - Condense the content while retaining important terms, definitions, and concepts.

#         2. **Structure**:
#            - Use headings and subheadings to organize the summaries, reflecting the structure of the document.
#            - If the document has bullet points or numbered lists, incorporate their essence into the summaries.

#         3. **Tables**:
#            - If tables are present, summarize their content by highlighting key data trends or insights.

#         4. **Mathematical Equations**:
#            - Summarize equations briefly, focusing on their significance or what they represent.

#         5. **Key Highlights**:
#            - Pay special attention to bolded, italicized, or underlined text, and ensure that critical terms or phrases are included in the summary.

#         The goal is to create compact summaries that represent the essence of each section, enabling the generation of diverse flashcards without needing to process the entire document content verbatim.
#         """
#     )

# def process_document(parser, input_file):
#     parsed_content = parser.load_data(input_file)
#     if not isinstance(parsed_content, list):
#         parsed_content = [parsed_content]

#     documents = []
#     for content in parsed_content:
#         text = content if isinstance(content, str) else str(content)
#         doc = Document(page_content=text, metadata={})
#         documents.append(doc)

#     return documents

# def split_document(documents, chunk_size=4000, chunk_overlap=10):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap
#     )
#     return text_splitter.split_documents(documents)

# def generate_with_openrouter(prompt, model):
#     """Generate response using OpenRouter API with OpenAI client."""
#     client = OpenAI(
#         base_url="https://openrouter.ai/api/v1",
#         api_key=os.getenv("OPENROUTER_API_KEY")
#     )
    
#     try:
#         completion = client.chat.completions.create(
#             extra_headers={
#                 "HTTP-Referer": os.getenv('APP_DOMAIN', 'http://localhost:8501'),
#                 "X-Title": "Flashcard Generator"
#             },
#             model=model,
#             messages=[{"role": "user", "content": prompt}]
#         )
#         return completion.choices[0].message.content
#     except Exception as e:
#         raise Exception(f"API call failed: {str(e)}")

# # Streamlit app
# st.title("Flashcard Generator")

# # Multiple file uploads
# uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# if uploaded_files:
#     flashcards_by_document = {}

#     try:
#         parser = initialize_parser()

#         for uploaded_file in uploaded_files:
#             file_name = uploaded_file.name
#             st.info(f"Processing the uploaded document: {file_name}")

#             # Save the uploaded file temporarily
#             input_file = f"uploaded_{file_name}"
#             with open(input_file, "wb") as f:
#                 f.write(uploaded_file.read())

#             # Process the document
#             parsed_documents = process_document(parser, input_file)
            
#             if not parsed_documents:
#                 st.error(f"No content was extracted from the document: {file_name}")
#                 continue

#             docs = split_document(parsed_documents)

#             st.info(f"Generating flashcards for {file_name}...")
#             flashcards = []

#             for doc in docs:
#                 chunk = doc.page_content
#                 prompt = f"""
#                 Generate diverse flashcards from the given content using the following formats:

#                 Q&A: Ask insightful questions with detailed answers.
#                 True/False: Create statements for evaluation.
#                 Fill-in-the-Blank: Omit key terms for completion.
#                 Multiple Choice: Provide a question with 3-4 options and the correct answer.
                
#                 Content:
#                 {chunk}
                
#                 Format for each flashcard:
#                 ---
#                 **Type**: [Question and Answer / True/False / Fill-in-the-Blank / Multiple Choice]

#                 **Content**: 
#                 [The actual flashcard content with the question, statement, or description]

#                 **Answer**: 
#                 [The answer or explanation corresponding to the flashcard]
#                 ---
#                 """
#                 try:
#                     response = generate_with_openrouter(prompt, model="qwen/qwen-2-7b-instruct")
#                     flashcards.append(response.strip())
#                 except Exception as e:
#                     st.error(f"Error generating flashcard for {file_name}: {str(e)}")
#                     continue

#             # Save flashcards for the document
#             flashcards_by_document[file_name] = flashcards

#             if flashcards:
#                 st.success(f"Flashcards generated for {file_name}")
#             else:
#                 st.error(f"No flashcards generated for {file_name}")

#         # Display and download options for all flashcards
#         st.subheader("Generated Flashcards by Document")
#         for file_name, flashcards in flashcards_by_document.items():
#             st.markdown(f"### {file_name}")
#             for i, flashcard in enumerate(flashcards, start=1):
#                 with st.expander(f"{file_name} - Flashcard {i}"):
#                     st.markdown(flashcard)

#         # Download JSON file
#         st.download_button(
#             label="Download Flashcards as JSON",
#             data=json.dumps(flashcards_by_document, indent=4),
#             file_name="flashcards.json",
#             mime="application/json"
#         )

#     except Exception as e:
#         st.error(f"An error occurred: {str(e)}")
#         st.error("Please make sure your PDF files are valid and try again.")
import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI
from groq import Groq
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import json
import os

# Load environment variables from .env file
load_dotenv()

def process_pdf(pdf_file):
    """Process PDF using PyMuPDF and extract text."""
    try:
        # Open the PDF file
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        
        # Extract text from each page
        for page in doc:
            text += page.get_text()
        
        # Create a Document object
        document = Document(page_content=text, metadata={})
        return [document]
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

def generate_with_groq(prompt, model):
    """Generate response using GROQ client."""
    client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    try:
        completion = client.chat.completions.create(
            messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
            model=model,
            reasoning_format="hidden"
        )
        return completion.choices[0].message.content
    except Exception as e:
        raise Exception(f"API call failed: {str(e)}")
# Streamlit app
st.title("Flashcard Generator")

# Multiple file uploads
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    flashcards_by_document = {}

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        st.info(f"Processing the uploaded document: {file_name}")

        try:
            # Process the PDF
            parsed_documents = process_pdf(uploaded_file)
            
            if not parsed_documents:
                st.error(f"No content was extracted from the document: {file_name}")
                continue

            docs = split_document(parsed_documents)

            st.info(f"Generating flashcards for {file_name}...")
            flashcards = []

            for doc in docs:
                chunk = doc.page_content
                prompt = f"""
            
                    Generate diverse flashcards from the given content using the following formats:

                    Q&A: Ask insightful questions with detailed answers.
                    True/False: Create statements for evaluation.
                    Fill-in-the-Blank: Omit key terms for completion.
                    Multiple Choice: Provide a question with 3-4 options and the correct answer.

                    Content:
                    {chunk}

                    Output only the flashcards in the format below. Do not include any explanation, reasoning, or meta-thinking.

                    Format for each flashcard:
                    ---
                    **Type**: [Question and Answer / True/False / Fill-in-the-Blank / Multiple Choice]

                    **Content**: 
                    [The actual flashcard content with the question, statement, or description]

                    **Answer**: 
                    [The answer or explanation corresponding to the flashcard]
                    ---
                """
                try:
                    response = generate_with_groq(prompt, model="deepseek-r1-distill-llama-70b")
                    flashcards.append(response.strip())
                except Exception as e:
                    st.error(f"Error generating flashcard for {file_name}: {str(e)}")
                    continue

            # Save flashcards for the document
            flashcards_by_document[file_name] = flashcards

            if flashcards:
                st.success(f"Flashcards generated for {file_name}")
            else:
                st.error(f"No flashcards generated for {file_name}")

        except Exception as e:
            st.error(f"An error occurred processing {file_name}: {str(e)}")
            continue

    # Display and download options for all flashcards
    st.subheader("Generated Flashcards by Document")
    for file_name, flashcards in flashcards_by_document.items():
        st.markdown(f"### {file_name}")
        for i, flashcard in enumerate(flashcards, start=1):
            with st.expander(f"{file_name} - Flashcard {i}"):
                st.markdown(flashcard)

    # Download JSON file
    st.download_button(
        label="Download Flashcards as JSON",
        data=json.dumps(flashcards_by_document, indent=4),
        file_name="flashcards.json",
        mime="application/json"
    )