import streamlit as st
from resume_parser import parse_resume
import random
from langchain_utils import extract_keywords, generate_resume_content, update_resume
from langchain_analyze import summary_prompt, strengths_prompt
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
import openai
from scrape import scrape_resume, fetch_url_content
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

st.title("Resume Analyzer and Optimizer")
action = st.radio(
    "Select Action:",
    ("Analyze Resume", "Create a New Resume", "Update Existing Resume"),
)


# Function to toggle input visibility
def toggle_inputs():
    st.session_state.show_inputs = not st.session_state.show_inputs


def pdf_to_chunks(pdf):
    # read pdf and it returns memory addresss
    pdf_reader = PdfReader(pdf)

    # extract text forom each page separately
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split the long text into small chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=200, length_function=len
    )

    chunks = text_splitter.split_text(text=text)
    return chunks


def openai_function(openai_api_key, chunks, analyze):

    # Using OpenAI service for embedding
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Facebook AI Similiarity Search (FAISS) to help us convert text data to numerical vector
    vectorstores = FAISS.from_texts(chunks, embedding=embeddings)

    # compares the query and chunks, enabling the selection of the top 'K' most similiar chunks based on their similiarity score
    docs = vectorstores.similarity_search(query=analyze, k=3)

    # creates an OpenAI object, using the ChatGPT 4o model
    llm = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)

    # question-answering (QA) pipeline, making use of the load_qa_chain function
    chain = load_qa_chain(llm=llm, chain_type="stuff")

    response = chain.run(input_documents=docs, question=analyze)
    return response


# Initialize session state for visibility
if "show_inputs" not in st.session_state:
    st.session_state.show_inputs = True
if "show_strengths_button" not in st.session_state:
    st.session_state.show_strengths_button = False

# Display file uploader for specific actions
if action in ["Analyze Resume", "Update Existing Resume"]:
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
else:
    uploaded_file = None


# Display input boxes based on the selected action
if action != "Analyze Resume":
    job_description = st.text_area("Job Description")
    role = st.text_input("Interested Role")

# Conditionally buttons for "Analyze Resume"
if action == "Analyze Resume" and uploaded_file is not None:
    if st.button("Summarize Resume"):
        toggle_inputs()  # Hide inputs when processing starts

        pdf_chunks = pdf_to_chunks(uploaded_file)
        st.session_state.pdf_chunks = pdf_chunks  # Store chunks in session state

        analyze_prompt = summary_prompt(query_with_chunks=pdf_chunks)
        response = openai_function(
            openai_api_key=openai.api_key,
            chunks=pdf_chunks,
            analyze=analyze_prompt
        )
        st.write(response)

    if st.button("Outline Resume Strengths"):
        # Log to outline resume strengths
        pdf_chunks = st.session_state.pdf_chunks
        prompt_summary = summary_prompt(query_with_chunks=pdf_chunks)
        summary = openai_function(
            openai_api_key=openai.api_key,
            chunks=pdf_chunks,
            analyze=prompt_summary
        )
        prompt_strengths = strengths_prompt(query_with_chunks=summary)
        strengths = openai_function(
            openai_api_key=openai.api_key,
            chunks=pdf_chunks,
            analyze=prompt_strengths
        )
        st.write(strengths)

if action in ["Create a New Resume", "Update Exisiting Resume"]:
    if st.button("Process"):
        # if action == "Analyze Resume" and uploaded_file is not None:
        #     toggle_inputs()  # Hide inputs when processing starts

        #     pdf_chunks = pdf_to_chunks(uploaded_file)
        #     analyze_prompt = summary_prompt(query_with_chunks=pdf_chunks)
        #     response = openai_function(openai_api_key=openai.api_key, chunks=pdf_chunks, analyze=analyze_prompt)
        #     st.write(response)

        #     # Show the "Outline Resume Strengths" button
        #     st.session_state.show_strengths_button = True

        if (
            action == "Update Existing Resume"
            and uploaded_file is not None
            and job_description
            and role
        ):
            resume_text = parse_resume(uploaded_file)
            keywords = extract_keywords(job_description)
            urls_keywords = scrape_resume(role)

            fetched_data = [fetch_url_content(url) for url in urls_keywords]

            updated_resume = update_resume(
                original_resume=resume_text,
                role=role,
                keywords=keywords,
                fetched_resumes=fetched_data,
            )

            st.download_button(
                "Download Updated Resume", updated_resume, file_name="updated_resume.txt"
            )

            st.write("Updated Resume:")
            st.write(updated_resume)

            st.write("Similar resume urls found:")
            st.write(urls_keywords)

        elif action == "Create a New Resume" and job_description and role:
            keywords = extract_keywords(job_description)
            urls_keywords = scrape_resume(role)

            fetched_data = [fetch_url_content(url) for url in urls_keywords]

            new_resume_content = generate_resume_content(
                keywords=keywords, role=role, fetched_resumes=fetched_data
            )

            st.download_button(
                "Download New Generated Resume",
                new_resume_content,
                file_name="new_resume.txt",
            )

            st.write("Generated Resume:")
            st.write(new_resume_content)

            st.write("Similar resume urls found:")
            st.write(urls_keywords)

        else:
            st.error("Please fill all fields to proceed.")
