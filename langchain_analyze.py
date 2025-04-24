from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import warnings
warnings.filterwarnings('ignore')
import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

def pdf_to_chunks(pdf):
    # read pdf and it returns memory addresss
    pdf_reader = PdfReader(pdf)

    # extract text forom each page separately
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split the long text into small chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=200,
        length_function=len)

    chunks = text_splitter.split_text(text=text)

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
