import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
import os
import pandas as pd

st.title("📝 File Q&A with HuggingFace")

# Get the HuggingFace API key from environment variable
hf_api_key = os.getenv("HF_TOKEN")

# Sidebar to display the GitHub info
with st.sidebar:
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/1_File_Q%26A.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

# Allow multiple file uploads (between 1 and 10 files) including CSV, TXT, and MD
uploaded_files = st.file_uploader("Upload articles", type=("txt", "md", "csv"), accept_multiple_files=True)

# Check if the number of files uploaded is between 12 and 10
if uploaded_files:
    if len(uploaded_files) < 1 or len(uploaded_files) > 10:
        st.warning("Please upload between 1 and 10 files.")
    else:
        st.write(f"Uploaded {len(uploaded_files)} files.")
else:
    st.warning("Please upload between 1 and 10 files.")

# Question input
question = st.text_input(
    "Ask something about the articles",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_files,
)

# Check if the API key is missing
if uploaded_files and question and not hf_api_key:
    st.info("Please add your HuggingFace API key to the environment variables to continue.")

import pandas as pd

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

# If both files, question, and API key are provided
if uploaded_files and question and hf_api_key:
    combined_article = ""
    
    # Combine the content of all uploaded files
    for file in uploaded_files:
        file_type = file.name.split('.')[-1]
        
        if file_type == 'csv':
            # Process CSV files
            combined_article += process_csv(file)
        else:
            # Process text and markdown files
            combined_article += file.read().decode()
        
        combined_article += "\n\n"  # Separate articles with a newline for clarity

    # Set up the HuggingFace model inference using HuggingFaceEndpoint
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"  # You can replace this with your desired HuggingFace model (e.g., GPT-3, GPT-Neo, etc.)
    hf = HuggingFaceEndpoint(
        repo_id=model_id,
        task="text-generation",  # Specifying the task type as text generation
        max_new_tokens=100,
        temperature=0.7,
        token=hf_api_key,  # Use the API key from the environment variable
    )

    # Prepare the prompt for the question
    prompt = f"Here are some articles:\n\n{combined_article}\n\nQuestion: {question}\nAnswer:"

    # Get the response from the model
    response = hf(prompt)

    # Display the response
    st.write("### Answer")
    st.write(response)  # Directly display the string response from the model