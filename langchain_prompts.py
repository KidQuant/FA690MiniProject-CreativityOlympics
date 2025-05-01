import warnings

warnings.filterwarnings("ignore")
import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


def summary_prompt(query_with_chunks):
    query = f'''
    I need a detailed summary of the resume below. Finally, provide us with a conclusion
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    {query_with_chunks}
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    '''
    return query


def strengths_prompt(query_with_chunks):
    query = f''' I need a detailed analysis and explaination of the strengths of the resume. List out the strengths.
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    {query_with_chunks}
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    '''
    return query


def weaknesses_prompt(query_with_chunks):
    query = f''' I need a detailed analysis and explaination of the weaknesses of the resume. List out the weaknesses.
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    {query_with_chunks}
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    '''
    return query


def job_title_prompt(query_with_chunks):
    query = f'''Based on my resume, come up with some job roles that would best fit my skills and abilities.
                """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                {query_with_chunks}

                """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                '''
    return query


def analyze_skills_from_jobs(jobs_json):
    query = f"""Analyze the following job data to extract the most important skills for candidates:
                \"\"\"{jobs_json}\"\"\"
             """
    return query


# Added query for cover letter option
def cover_letter_prompt(resume_text, job_description, role):
    query = f'''You are a professional cover letter writer. Based on my resume and the job description below, generate a formal and customized cover letter for the position of {role}. The tone should be confident, professional, and highlight relevant skills and motivation.

                Resume:
                """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                {resume_text}
                """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

                Job Description:
                """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                {job_description}
                """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                '''
    return query
