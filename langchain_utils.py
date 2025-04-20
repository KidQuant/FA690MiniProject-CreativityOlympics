import os

import openai
from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
llm = ChatOpenAI(model="gpt-4o")


def extract_keywords(job_description):
    parser = JsonOutputParser()

    prompt_extract = PromptTemplate.from_template(
        """
            ### SCRAPED TEXT FROM WEBSITE:
            {job_description}
            ### INSTRUCTION:
            The scraped text is the job description from the career's page of a website.
            Your job is to extract the technical skills and keywords from the job description, and return them in JSON format containing the following keys: `skills` and `keywords`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
    )
    chain = prompt_extract | llm | parser
    result = chain.invoke({"job_description": job_description})
    return result


def update_resume(original_resume, role, keywords, fetched_resumes):
    parser = StrOutputParser()
    fetched_content = "\n".join(fetched_resumes)
    prompt_extract = PromptTemplate.from_template(
        """
            ### ORIGINAL RESUME
            {original_resume}
            ### CONTENT FROM SIMILAR RESUMES:
            {resumes}
            ### KEYWORDS
            {keywords}
            ### INSTRUCTION:
            You are a professional in the field of {role}.
            Update the original resume for the role {role} by integrating the keywords and insights from these similar resumes, ensuring it remains coherent and professional.
        """
    )
    chain = prompt_extract | llm | parser
    result = chain.invoke(
        {
            "original_resume": original_resume,
            "resumes": fetched_content,
            "keywords": keywords,
            "role": role,
        }
    )
    return result
