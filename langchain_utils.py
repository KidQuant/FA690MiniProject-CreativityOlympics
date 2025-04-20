import os

import openai
from dotenv import load_dotenv
from langchain_core.exception import OutputParserException
from langchain_core.output_parsers import JsonOutParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
