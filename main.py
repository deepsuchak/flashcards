
# import streamlit as st
# from openai import OpenAI
# import nest_asyncio
# from llama_parse import LlamaParse
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from dotenv import load_dotenv
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
#         parsing_instruction="""
#         The document is educational material, such as a textbook, presentation, or class notes.
#         Your task is to extract and organize the information in a structured format to assist with generating flashcards. Focus on the following:

#         1. **Text Content**:
#            - Extract all textual content, preserving headings, subheadings, and bullet points.
#            - Group content under its respective headings to maintain context.

#         2. **Tables**:
#            - Parse all tables and include their content in a structured format (e.g., row and column data).
#            - Include captions or titles of the tables if available.

#         3. **Images and Figures**:
#            - Identify and describe the content of any images, figures, or diagrams.
#            - Use captions or nearby text as context for describing images.

#         4. **Key Highlights**:
#            - Pay special attention to bolded, italicized, or underlined text as it may indicate important terms or concepts.

#         5. **Mathematical Equations** (if any):
#            - Extract equations and format them appropriately (e.g., LaTeX or plain text).

#         6. **Other Features**:
#            - Include lists, bullet points, and numbered items as they appear in the document.
#            - Maintain the logical flow and structure of the content.

#         Organize the parsed output logically, with clear separations between different content types (e.g., headings, tables, images). This structure will help an LLM generate meaningful flashcards.
#         """
#     )

# def process_document(parser, input_file):
#     # Parse the document and convert to LangChain Document format
#     parsed_content = parser.load_data(input_file)
#     # Convert the parsed content to a list if it isn't already
#     if not isinstance(parsed_content, list):
#         parsed_content = [parsed_content]
    
#     # Convert to LangChain Document format
#     documents = []
#     for content in parsed_content:
#         # Handle both string and dictionary-like objects
#         if isinstance(content, str):
#             text = content
#         else:
#             # If the content is a dictionary-like object, try to get the text content
#             # Adjust this based on the actual structure of your parsed content
#             text = str(content)  # Convert the entire content to string if needed
        
#         doc = Document(page_content=text, metadata={})
#         documents.append(doc)
    
#     return documents

# def split_document(documents, chunk_size=2000, chunk_overlap=100):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap
#     )
#     return text_splitter.split_documents(documents)

# # Streamlit app
# st.title("Flashcard Generator")

# # File upload
# uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# if uploaded_file:
#     try:
#         # Save uploaded file
#         input_file = "uploaded_document.pdf"
#         with open(input_file, "wb") as f:
#             f.write(uploaded_file.read())

#         st.info("Processing the uploaded document...")

#         # Initialize parser
#         parser = initialize_parser()

#         # Process the document
#         parsed_documents = process_document(parser, input_file)
        
#         if not parsed_documents:
#             st.error("No content was extracted from the document.")
#             st.stop()
            
#         docs = split_document(parsed_documents)

#         # Generate flashcards
#         st.info("Generating flashcards...")
#         client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#         flashcards = []

#         for doc in docs:
#             chunk = doc.page_content
#             prompt = f"""
#             You are an AI assistant tasked with creating diverse and effective flashcards to help students learn and memorize information efficiently. 
#             The provided content may include text, tables, lists, definitions, or descriptions of diagrams and images. 

#             For the following content, generate flashcards in a variety of formats, ensuring a mix of:
#             1. **Question and Answer**: Pose insightful questions and provide detailed answers to explain key concepts.
#             2. **True/False**: Create statements for students to evaluate as true or false.
#             3. **Fill-in-the-Blank**: Leave out key terms or phrases for students to fill in.
#             4. **Multiple Choice**: Provide a question with 3-4 plausible options and indicate the correct answer.
#             5. **Table Interpretation**: If the content includes a table, create flashcards that ask students to interpret or analyze the data.
#             6. **Diagram Explanation**: Simplify and describe any diagram content into digestible learning points.

#             Content: {chunk}

#             Format for each flashcard:
#             ---
#             **Type**: [Question and Answer / True/False / Fill-in-the-Blank / Multiple Choice / Table Interpretation / Diagram Explanation]

#             **Content**: 
#             [The actual flashcard content with the question, statement, or description]

#             **Answer**: 
#             [The answer or explanation corresponding to the flashcard]
#             ---
#             """
#             try:
#                 response = client.chat.completions.create(
#                     model="gpt-4",
#                     messages=[
#                         {"role": "system", "content": "You are a helpful assistant."},
#                         {"role": "user", "content": prompt}
#                     ],
#                     max_tokens=500,
#                     temperature=0.2,
#                 )
#                 flashcards.append(response.choices[0].message.content.strip())
#             except Exception as e:
#                 st.error(f"Error generating flashcard: {str(e)}")
#                 continue

#         if flashcards:
#             # Display flashcards in a creative way
#             st.subheader("Generated Flashcards")
#             for i, flashcard in enumerate(flashcards, start=1):
#                 with st.expander(f"Flashcard {i}"):
#                     st.markdown(flashcard)

#             # Option to download flashcards
#             st.download_button(
#                 label="Download Flashcards",
#                 data="\n\n".join(flashcards),
#                 file_name="flashcards.txt",
#                 mime="text/plain"
#             )
#         else:
#             st.error("No flashcards were generated. Please try again.")

#     except Exception as e:
#         st.error(f"An error occurred: {str(e)}")
#         st.error("Please make sure your PDF file is valid and try again.")


import streamlit as st
from llama_parse import LlamaParse
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import json
import re
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

def generate_with_openrouter(prompt, model):
    """Generate response using OpenRouter API with OpenAI client."""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
    
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": os.getenv('APP_DOMAIN', 'http://localhost:8501'),
                "X-Title": "Flashcard Generator"
            },
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        raise Exception(f"API call failed: {str(e)}")
    
# def parse_flashcard(text):
#     """Parse the flashcard text into a structured dictionary."""
#     type_pattern = r'\*\*Type\*\*:\s*([^\n]+)'
#     content_pattern = r'\*\*Content\*\*:\s*([^\n]+(?:\n(?!\*\*)[^\n]+)*)'
#     answer_pattern = r'\*\*Answer\*\*:\s*([^\n]+(?:\n(?!\*\*)[^\n]+)*)'

#     flashcard_type = re.search(type_pattern, text)
#     content = re.search(content_pattern, text)
#     answer = re.search(answer_pattern, text)

#     if flashcard_type and content and answer:
#         return {
#             "type": flashcard_type.group(1).strip(),
#             "content": content.group(1).strip(),
#             "answer": answer.group(1).strip()
#         }
#     return None

def initialize_parser():
    return LlamaParse(
        api_key=os.getenv("LLAMA_API_KEY"),
        result_type="markdown",
        language="en",
        verbose=True,
        is_formatting_instruction=False,
        parsing_instruction= """  The document is educational material, such as a textbook, presentation, or class notes.
        Your task is to process the document and provide concise summaries of its contents. The summaries should be structured to assist with generating flashcards. Focus on the following:

        1. **Summarization**:
           - Summarize each section of the document in 2-3 sentences, preserving the key points and main ideas.
           - Condense the content while retaining important terms, definitions, and concepts.

        2. **Structure**:
           - Use headings and subheadings to organize the summaries, reflecting the structure of the document.
           - If the document has bullet points or numbered lists, incorporate their essence into the summaries.

        3. **Tables**:
           - If tables are present, summarize their content by highlighting key data trends or insights.

        4. **Mathematical Equations**:
           - Summarize equations briefly, focusing on their significance or what they represent.

        5. **Key Highlights**:
           - Pay special attention to bolded, italicized, or underlined text, and ensure that critical terms or phrases are included in the summary.

        The goal is to create compact summaries that represent the essence of each section, enabling the generation of diverse flashcards without needing to process the entire document content verbatim.
        """
    )

def process_document(parser, input_file):
    parsed_content = parser.load_data(input_file)
    if not isinstance(parsed_content, list):
        parsed_content = [parsed_content]
    
    documents = []
    for content in parsed_content:
        if isinstance(content, str):
            text = content
        else:
            text = str(content)
        
        doc = Document(page_content=text, metadata={})
        documents.append(doc)
    
    return documents

def split_document(documents, chunk_size=4000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

# Streamlit app
st.title("Flashcard Generator")

# Model selection
model_options = {
    "Gemini": "google/gemini-2.0-flash-exp:free",
    "Deepseek-33B": "deepseek/deepseek-r1",
    "Qwen-7B": "qwen/qwen-2-7b-instruct",
    "Mistral-7B": "mistralai/mistral-7b-instruct"
}
selected_model = st.selectbox("Select LLM Model", list(model_options.keys()))

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
# if uploaded_file:
#     try:
#         # Save uploaded file
#         input_file = "uploaded_document.pdf"
#         with open(input_file, "wb") as f:
#             f.write(uploaded_file.read())

#         st.info("Processing the uploaded document...")

#         # Initialize parser
#         parser = initialize_parser()

#         # Process the document
#         parsed_documents = process_document(parser, input_file)
        
#         if not parsed_documents:
#             st.error("No content was extracted from the document.")
#             st.stop()
            
#         docs = split_document(parsed_documents)

    #     # Generate flashcards
    #     st.info("Generating flashcards...")
    #     flashcards = []
    #     structured_flashcards = []

    #     for doc in docs:
    #         chunk = doc.page_content
    #         prompt = f"""
    #         Create diverse and effective flashcards from the following content. Include a mix of:
    #         1. Question and Answer
    #         2. True/False
    #         3. Fill-in-the-Blank
    #         4. Multiple Choice (3-4 options)
    #         5. Table Interpretation (if applicable)
        

    #         Content: {chunk}

    #         Format each flashcard exactly as:
    #         ---
    #         **Type**: [Question and Answer / True/False / Fill-in-the-Blank / Multiple Choice / Table Interpretation]

    #         **Content**: 
    #         [Question/statement]

    #         **Answer**: 
    #         [Answer/explanation]
    #         ---
    #         """
    #         try:
    #             response = generate_with_openrouter(prompt, model_options[selected_model])
    #             if response:
    #                 flashcards.append(response)
                    
    #                 # Parse the flashcard and add to structured list
    #                 parsed_flashcard = parse_flashcard(response)
    #                 if parsed_flashcard:
    #                     structured_flashcards.append(parsed_flashcard)
    #                 else:
    #                     st.warning("Flashcard format was invalid; skipped this flashcard.")
    #             else:
    #                 st.warning("API returned no content for this chunk; skipping...")
    #         except Exception as e:
    #             st.error(f"Error generating flashcard: {str(e)}")
    #             continue

    #     if structured_flashcards:
    #         # Display flashcards
    #         st.subheader("Generated Flashcards")
    #         for i, flashcard in enumerate(flashcards, start=1):
    #             with st.expander(f"Flashcard {i}"):
    #                 st.markdown(flashcard)

    #         # Create JSON data
    #         flashcards_json = {
    #             "flashcards": structured_flashcards,
    #             "metadata": {
    #                 "total_cards": len(structured_flashcards),
    #                 "model_used": selected_model,
    #                 "generated_date": str(st.session_state.get('current_date', ''))
    #             }
    #         }

    #         # Download options
    #         st.download_button(
    #             label="Download Flashcards (JSON)",
    #             data=json.dumps(flashcards_json, indent=2),
    #             file_name="flashcards.json",
    #             mime="application/json"
    #         )

    #         # Also provide text format as an alternative
    #         st.download_button(
    #             label="Download Flashcards (TXT)",
    #             data="\n\n".join(flashcards),
    #             file_name="flashcards.txt",
    #             mime="text/plain",
    #             key="txt_download"
    #         )
    #     else:
    #         st.error("No flashcards were generated. Please try again.")
    # except Exception as e:
    #     st.error(f"An error occurred: {str(e)}")

if uploaded_file:
    try:
        # Save uploaded file
        input_file = "uploaded_document.pdf"
        with open(input_file, "wb") as f:
            f.write(uploaded_file.read())

        st.info("Processing the uploaded document...")

        # Initialize parser
        parser = initialize_parser()

        # Process the document
        parsed_documents = process_document(parser, input_file)
        
        if not parsed_documents:
            st.error("No content was extracted from the document.")
            st.stop()
            
        docs = split_document(parsed_documents)
        #Generate flashcards
        st.info("Generating flashcards...")
        flashcards = []
        structured_flashcards = []

        for doc in docs:
            chunk = doc.page_content
            prompt = f"""
            Create diverse and effective flashcards from the following content. Include a mix of:
            1. Question and Answer
            2. True/False
            3. Fill-in-the-Blank
            4. Multiple Choice (3-4 options)
            

            Content: {chunk}

            Format each flashcard exactly as:
            ---
            **Type**: [Question and Answer / True/False / Fill-in-the-Blank / Multiple Choice]

            **Content**: 
            [Question/statement]

            **Answer**: 
            [Answer/explanation]
            ---
            """
            try:
                response = generate_with_openrouter(prompt, model_options[selected_model])
                if response:
                    flashcards.append(response)
                    
                    # Parse the flashcard and add to structured list
                    # parsed_flashcard = parse_flashcard(response)
                    # if parsed_flashcard:
                    #     structured_flashcards.append(parsed_flashcard)
                    # else:
                    #     st.warning("Flashcard format was invalid; skipped this flashcard.")
                else:
                    st.warning("API returned no content for this chunk; skipping...")
            except Exception as e:
                st.error(f"Error generating flashcard: {str(e)}")
                continue

        if structured_flashcards:
            # Display flashcards
            st.subheader("Generated Flashcards")
            for i, flashcard in enumerate(flashcards, start=1):
                with st.expander(f"Flashcard {i}"):
                    st.markdown(flashcard)

            # Create JSON data
            flashcards_json = {
                "flashcards": structured_flashcards,
                "metadata": {
                    "total_cards": len(structured_flashcards),
                    "model_used": selected_model,
                    "generated_date": str(st.session_state.get('current_date', ''))
                }
            }

            # Download options
            st.download_button(
                label="Download Flashcards (JSON)",
                data=json.dumps(flashcards_json, indent=2),
                file_name="flashcards.json",
                mime="application/json"
            )

            # Also provide text format as an alternative
            st.download_button(
                label="Download Flashcards (TXT)",
                data="\n\n".join(flashcards),
                file_name="flashcards.txt",
                mime="text/plain",
                key="txt_download"
            )
        else:
            st.error("No flashcards were generated. Please try again.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please make sure your PDF file is valid and try again.")


    #     #Generate flashcards
    #     st.info("Generating flashcards...")
    #     flashcards = []
    #     structured_flashcards = []

    #     for doc in docs:
    #         chunk = doc.page_content
    #         prompt = f"""
    #         Create diverse and effective flashcards from the following content. Include a mix of:
    #         1. Question and Answer
    #         2. True/False
    #         3. Fill-in-the-Blank
    #         4. Multiple Choice (3-4 options)
    #         5. Table Interpretation (if applicable)

    #         Content: {chunk}

    #         Format each flashcard exactly as:
    #         ---
    #         **Type**: [Question and Answer / True/False / Fill-in-the-Blank / Multiple Choice / Table Interpretation]

    #         **Content**: 
    #         [Question/statement]

    #         **Answer**: 
    #         [Answer/explanation]
    #         ---
    #         """
    #         try:
    #             response = generate_with_openrouter(prompt, model_options[selected_model])
    #             flashcards.append(response)
                
    #             # Parse the flashcard and add to structured list
    #             parsed_flashcard = parse_flashcard(response)
    #             if parsed_flashcard:
    #                 structured_flashcards.append(parsed_flashcard)
                
    #         except Exception as e:
    #             st.error(f"Error generating flashcard: {str(e)}")
    #             continue

    #     if structured_flashcards:
    #         # Display flashcards
    #         st.subheader("Generated Flashcards")
    #         for i, flashcard in enumerate(flashcards, start=1):
    #             with st.expander(f"Flashcard {i}"):
    #                 st.markdown(flashcard)

    #         # Create JSON data
    #         flashcards_json = {
    #             "flashcards": structured_flashcards,
    #             "metadata": {
    #                 "total_cards": len(structured_flashcards),
    #                 "model_used": selected_model,
    #                 "generated_date": str(st.session_state.get('current_date', ''))
    #             }
    #         }

    #         # Download options
    #         st.download_button(
    #             label="Download Flashcards (JSON)",
    #             data=json.dumps(flashcards_json, indent=2),
    #             file_name="flashcards.json",
    #             mime="application/json"
    #         )

    #         # Also provide text format as an alternative
    #         st.download_button(
    #             label="Download Flashcards (TXT)",
    #             data="\n\n".join(flashcards),
    #             file_name="flashcards.txt",
    #             mime="text/plain",
    #             key="txt_download"
    #         )
    #     else:
    #         st.error("No flashcards were generated. Please try again.")

    # except Exception as e:
    #     st.error(f"An error occurred: {str(e)}")
    #     st.error("Please make sure your PDF file is valid and try again.")