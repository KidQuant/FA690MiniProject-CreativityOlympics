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


