import pandas as pd
import json
import re
import streamlit as st
from io import BytesIO
from docx import Document
from resume_parser import parse_resume
from langchain_utils import extract_keywords, generate_resume_content, update_resume
from langchain_prompts import (
    summary_prompt,
    strengths_prompt,
    weaknesses_prompt,
    job_title_prompt,
    cover_letter_prompt,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
import openai
from scrape import scrape_resume, fetch_url_content
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
from jobspy import scrape_jobs

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

st.title("Resume Analyzer and Optimizer")
action = st.radio("Select Action:", ("Analyze Resume", "Create a New Resume", "Update Existing Resume"))

def create_docx(text: str, title: str = None) -> BytesIO:
    doc = Document()
    if title:
        doc.add_heading(title, level=1)
    for para in text.split("\\n\\n"):
        doc.add_paragraph(para.strip())
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

def create_pdf(text: str) -> BytesIO:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split("\\n"):
        pdf.multi_cell(0, 10, line)
    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

def pdf_to_chunks(pdf):
    pdf_reader = PdfReader(pdf)
    text = "".join(page.extract_text() for page in pdf_reader.pages)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
    return text_splitter.split_text(text)

def openai_function(openai_api_key, chunks, analyze):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstores = FAISS.from_texts(chunks, embedding=embeddings)
    docs = vectorstores.similarity_search(query=analyze, k=3)
    llm = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    return chain.run(input_documents=docs, question=analyze)

if action in ["Analyze Resume", "Update Existing Resume"]:
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

if action != "Analyze Resume":
    job_description = st.text_area("Job Description")
    role = st.text_input("Interested Role")

if action == "Analyze Resume" and uploaded_file:
    pdf_chunks = pdf_to_chunks(uploaded_file)
    st.session_state.pdf_chunks = pdf_chunks
    col1, col2, col3, col4 = st.columns(4)
    if col1.button("Summarize Resume"):
        response = openai_function(openai.api_key, pdf_chunks, summary_prompt(pdf_chunks))
        st.write(response)
    if col2.button("Outline Resume Strengths"):
        summary = openai_function(openai.api_key, pdf_chunks, summary_prompt(pdf_chunks))
        strengths = openai_function(openai.api_key, pdf_chunks, strengths_prompt(summary))
        st.write(strengths)
    if col3.button("Outline Resume Weaknesses"):
        summary = openai_function(openai.api_key, pdf_chunks, summary_prompt(pdf_chunks))
        weaknesses = openai_function(openai.api_key, pdf_chunks, weaknesses_prompt(summary))
        st.write(weaknesses)
    if col4.button("Provide Job Examples"):
        summary = openai_function(openai.api_key, pdf_chunks, summary_prompt(pdf_chunks))
        jobs = openai_function(openai.api_key, pdf_chunks, job_title_prompt(summary))
        st.session_state.job_examples = re.findall(r"\\*\\*(.*?)\\*\\*", jobs)
        st.write(jobs)

if st.session_state.get("job_examples") and st.button("Search for Jobs"):
    st.write("Web Scraping for relevant jobs:", st.session_state.job_examples)
    jobs_df = pd.DataFrame()
    for job in st.session_state.job_examples:
        results = scrape_jobs(["indeed", "linkedin", "glassdoor", "google"], job, "New York, NY", 20, "USA")
        results["job_type"] = job
        jobs_df = pd.concat([jobs_df, results], ignore_index=True)
    st.write(jobs_df)

if action in ["Create a New Resume", "Update Existing Resume"]:
    if st.button("Process"):
        if action == "Update Existing Resume" and uploaded_file and job_description and role:
            text = parse_resume(uploaded_file)
            keywords = extract_keywords(job_description)
            urls = scrape_resume(role)
            data = [fetch_url_content(url) for url in urls]

            resume_option_1 = update_resume(text, role, keywords, data)
            resume_option_2 = update_resume(text, role, keywords["skills"] + ["variation"], data)
            
            # Store both resumes in session to persist after UI refresh
            st.session_state.resume_option_1 = resume_option_1
            st.session_state.resume_option_2 = resume_option_2

        elif action == "Create a New Resume" and job_description and role:
            keywords = extract_keywords(job_description)
            urls = scrape_resume(role)
            data = [fetch_url_content(url) for url in urls]
            resume_option_1 = generate_resume_content(role, keywords, data)
            resume_option_2 = generate_resume_content(role, keywords["skills"] + ["variation"], data)
            
            # Store both resumes in session to persist after UI refresh
            st.session_state.resume_option_1 = resume_option_1
            st.session_state.resume_option_2 = resume_option_2

        else:
            st.error("❌ Please fill all fields to proceed.")

            # Restore resume visual
            resume_option_1 = st.session_state.get("resume_option_1", "")
            resume_option_2 = st.session_state.get("resume_option_2", "")

        if resume_option_1 and resume_option_2:
            # Restore from session in case user just selected an option
            resume_option_1 = st.session_state.get("resume_option_1", "")
            resume_option_2 = st.session_state.get("resume_option_2", "")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Resume Option 1")
                st.text_area("Resume Option 1 Content", resume_option_1, height=400, key="visual_opt1")

                st.download_button(
                    "📄 Download as TXT",
                    resume_option_1,
                    file_name="resume_option_1.txt",
                    key="dl_txt_opt1"
                )
                st.download_button(
                    "📄 Download as DOCX",
                    create_docx(resume_option_1, "Resume Option 1"),
                    file_name="resume_option_1.docx",
                    key="dl_docx_opt1"
                )

            with col2:
                st.markdown("### Resume Option 2")
                st.text_area("Resume Option 2 Content", resume_option_2, height=400, key="visual_opt2")

                st.download_button(
                    "📄 Download as TXT",
                    resume_option_2,
                    file_name="resume_option_2.txt",
                    key="dl_txt_opt2"
                )
                st.download_button(
                    "📄 Download as DOCX",
                    create_docx(resume_option_2, "Resume Option 2"),
                    file_name="resume_option_2.docx",
                    key="dl_docx_opt2"
                )

if "selected_resume" in st.session_state and job_description and role:
    st.subheader("✍️ Create a Matching Cover Letter")
    if st.button("Generate Cover Letter", key="generate_cover_letter"):
        prompt = cover_letter_prompt(st.session_state.selected_resume, job_description, role)
        cover = openai_function(openai.api_key, [st.session_state.selected_resume], prompt)
        st.subheader("📄 Suggested Cover Letter")
        st.text_area("Generated Cover Letter", cover, height=300)
        st.download_button("Download as TXT", cover, file_name="cover_letter.txt", key="cov_txt")
        st.download_button("Download as DOCX", create_docx(cover, "Cover Letter"), file_name="cover_letter.docx", key="cov_docx")

    else:
            st.error("Please fill all fields to proceed.")
