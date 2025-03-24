import huggingface_hub
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
import os
import pandas as pd
import PyPDF2
from datetime import datetime, timedelta

st.title("🐾 Pet Sitting Runbook Generator with HuggingFace")

# Get the HuggingFace API key from environment variable
hf_api_key = os.getenv("HF_TOKEN")

# Sidebar to display the GitHub info
with st.sidebar:
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/1_File_Q%26A.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

# Allow multiple file uploads (between 1 and 10 files) including CSV, TXT, and MD
# Function to upload and save user files
def upload_and_save_files():
    user_files = st.file_uploader("Upload pet care files (1-10 files)", type=["pdf", "docx", "txt", "md", "csv"], accept_multiple_files=True)
    if user_files:
        for file in user_files:
            filename = file.name
            with open(os.path.join("uploaded_files", filename), "wb") as f:
                f.write(file.getvalue())
        st.success("Files uploaded and saved successfully!")

def read_system_input():
    pdf_file = open("Pet Sitting Runbook Template.pdf", "rb")
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    system_info = ""
    for page in range(len(pdf_reader.pages)):
        system_info += pdf_reader.pages[page].extractText()
    pdf_file.close()
    return system_info

# Upload user files
upload_and_save_files()

# Read system input file
system_info = read_system_input()

# Display user and system inputs
st.subheader("User Inputs")
uploaded_files = os.listdir("uploaded_files")
for file in uploaded_files:
    st.write(file)

st.subheader("System Input")
st.write(system_info)

# Function to process CSV file content
def process_csv(file):
    df = pd.read_csv(file)
    
    # Print the columns of the CSV file to debug
    # st.write("CSV Columns:", df.columns.tolist())
    
    # Check if both 'Question' and 'Answer' columns exist
    if 'Question' in df.columns and 'Answer' in df.columns:
        # Handle missing values and ensure the columns are treated as strings
        combined_text = "\n\n".join(
            df['Question'].fillna("").astype(str) + " " + df['Answer'].fillna("").astype(str)
        )
        return combined_text
    else:
        # Fallback: Use the first two columns if 'Question' and 'Answer' are not found
        combined_text = "\n\n".join(
            df.iloc[:, 0].fillna("").astype(str) + " " + df.iloc[:, 1].fillna("").astype(str)
        )
        return combined_text

# Section for date range selection
st.subheader("Date Range Selection")
start_date = st.date_input("Start Date", datetime.now())
end_date = st.date_input("End Date", datetime.now() + timedelta(days=7))

# Generate AI prompt and get user confirmation
with st.expander("AI Prompt Preview"):
    user_confirmation = st.checkbox("Show AI Prompt")
    if user_confirmation:
        prompt = f"""
        Generate a comprehensive pet sitting runbook based on the following user and system inputs:
        
        User Inputs:
        {uploaded_files}
        
        System Input(from PDF):
        {system_info}
        
        Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}
        
        Instructions:
        - Create a detailed runbook tailored to the user's pets.
        - Include sections for basic information, health, feeding, grooming, daily routine, and emergency contacts.
        - Adapt the runbook based on the number and types of pets provided.
        
        Output Format:
        - Use a clear structure with headings for each pet.
        - Provide a weekly schedule, feeding instructions, and individual care routines.
        
        Example User Input:
        - Pet 1:
          - Name: Fluffy
          - Type: Cat
          - ...
        
        Example System Input:
        [System input content]
        
        Example Output:
        [Provide an example runbook section here]
        """
        st.code(prompt)

# Generate comprehensive output using Hugging Face API
if st.button("Generate Runbook"):
    if user_confirmation:
        # Use Hugging Face API for model inference
        hf = huggingface_hub.InferenceEndpoint(
            repo_id="mistralai/Mistral-Nemo-Instruct-2407",
            task="text-generation",  # Specifying the task type as text generation
            max_new_tokens=1500,
            temperature=0.5,
            token=hf_api_key,  # Use the API key from the environment variable
        )
        response = hf(prompt)
        output = response[0]["generated_text"]
        st.success("Runbook generated successfully!")
        st.write(output)
    else:
        st.warning("Please confirm the AI prompt before generating the runbook.")
