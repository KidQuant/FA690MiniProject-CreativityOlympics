# Project: Resume Optimization and Analysis Resource (ROAR!)

## Overview

The ROAR! App is a comprehensive tool designed to assist users in optimizing and analyzing resumes. It leverages advanced AI models to provide insights into resume strengths and weaknesses, generate new resumes, and suggest job roles based on the user's skills and experiences. The application also includes a web scraping feature to find relevant job listings.

## Features

1. **Resume Analysis**:
   - Upload a PDF resume to receive a detailed analysis.
   - Identify strengths and weaknesses in the resume.
   - Generate a summary of the resume content.

2. **Resume Creation and Update**:
   - Create a new resume based on a job description and desired role.
   - Update an existing resume to better match job requirements.
   - Generate multiple resume versions for comparison.

3. **Job Role Suggestions**:
   - Suggest job roles that align with the user's skills and experiences.
   - Provide examples of job titles based on resume content.

4. **Job Search**:
   - Scrape job listings from popular job sites like Indeed, LinkedIn, and Google.
   - Display relevant job opportunities based on suggested job roles.

5. **Cover Letter Generation**:
   - Generate a personalized cover letter to accompany the resume.
   - Download the cover letter in TXT or DOCX format.

## Technical Details

- **Technologies Used**:
  - Python, Streamlit for the web interface.
  - OpenAI's GPT models for natural language processing.
  - FAISS for similarity search and vector storage.
  - PyPDF2 for PDF processing.
  - Various scraping tools for job data collection.

- **Key Libraries**:
  - `langchain` for text processing and AI model integration.
  - `pandas` for data manipulation.
  - `docx` and `FPDF` for document creation.

## Getting Started

1. **Installation**:
   - Clone the repository.
   - Install the required packages using `pip install -r requirements.txt`.

2. **Environment Setup**:
   - Set up your OpenAI API key in a `.env` file.

3. **Running the App**:
   - Launch the Streamlit app using `streamlit run app.py`.

## Contribution

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License.
