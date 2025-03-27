from mistralai import Mistral
import streamlit as st
#from langchain_huggingface import HuggingFaceEndpoint
import os
import pandas as pd
import PyPDF2
from datetime import datetime, timedelta
from fpdf import FPDF
import io

st.title("🐾 Pet Sitting Runbook Generator with Mistral")

# Get the Mistral API key from environment variable
api_key = os.getenv("MISTRAL_TOKEN")

# Sidebar to display the GitHub info
with st.sidebar:
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/1_File_Q%26A.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

# Section Start User Input
# Allow multiple file uploads (between 1 and 10 files) including CSV, TXT, and MD
# Function to upload and save user files

st.subheader("Start Sharing Your Pet(s) Information")

def upload_and_save_files():
    user_files = st.file_uploader("Upload between 1-10 pet care files.", type=["pdf", "docx", "txt", "md", "csv"], accept_multiple_files=True)
    if user_files:
        for file in user_files:
            filename = file.name
            with open(os.path.join("uploaded_files", filename), "wb") as f:
                f.write(file.getvalue())
        st.success("Files uploaded and saved successfully!")

# Create the "uploaded_files" directory if it doesn't exist
if not os.path.exists("uploaded_files"):
    os.makedirs("uploaded_files")

# Wait for user to upload files
upload_and_save_files()  

def read_system_input():
    pdf_file = open("Pet Sitting Runbook Template.pdf", "rb")
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    system_info = ""
    for page in range(len(pdf_reader.pages)):
        system_info += pdf_reader.pages[page].extract_text()
    pdf_file.close()
    return system_info

# Read system input file
system_info = read_system_input()

# Display user and system inputs
if os.listdir("uploaded_files"):
    st.subheader("User Inputs")
    uploaded_files = os.listdir("uploaded_files")
    for file in uploaded_files:
        st.write(file)

# st.subheader("System Input")
# st.write(system_info)

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
st.subheader("Choose Date(s) or Timeframe")
st.write ("Choose a Timeframe you would like a runbook generated for.")

# Define the options
options = ["Pick Dates", "Weekdays Only", "Weekend Only", "Default"]

# Create a radio selection
choice = st.radio("Choose an option:", options)

if choice == "Pick Dates":
    start_date = st.date_input("Select Start Date:", datetime.now())
    end_date = st.date_input("Select End Date:", datetime.now() + timedelta(days=7))
    st.write(f"You selected specific dates from {start_date} to {end_date}.")
elif choice == "Weekdays Only":
    st.write("You selected weekdays only.")
elif choice == "Weekend Only":
    st.write("You selected weekend only.")
elif choice == "Default":
    st.write("You selected a general schedule.")
else:
    st.write("Invalid choice.")

# Generate AI prompt and get user confirmation
with st.expander("AI Prompt Preview"):
    user_confirmation = st.checkbox("Show AI Prompt")
    if user_confirmation:
        if choice == "Pick Dates":
            prompt = f"""
            Generate a comprehensive pet sitting runbook for the selected date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.
            
            User Inputs:
            {uploaded_files}
            
            System Input(from PDF):
            {system_info}
            
            Instructions:
            - Create a detailed runbook tailored to the user's pets for the specified dates.
            - Include sections for basic information, health, feeding, grooming, daily routine, and emergency contacts.
            - Adapt the runbook based on the number and types of pets provided.
            
            Output Format:
            - Use a clear structure with headings for each pet.
            - Provide a schedule, feeding instructions, and individual care routines for the selected dates.
            
            Example User Input:
            - Pet 1:
              - Name: Fluffy
              - Type: Cat
              - ...
            
            Example System Input:
            [System input content]
            
            Example Output:
            [Provide an example runbook section for the selected dates here]
            """
        elif choice == "Weekdays Only":
            prompt = f"""
            Generate a comprehensive pet sitting runbook for weekdays only.
            
            User Inputs:
            {uploaded_files}
            
            System Input(from PDF):
            {system_info}
            
            Instructions:
            - Create a detailed runbook tailored to the user's pets for weekdays.
            - Include sections for basic information, health, feeding, grooming, and emergency contacts.
            - Adapt the runbook based on the number and types of pets.
            
            Output Format:
            - Use a clear structure with headings for each pet.
            - Provide a weekly schedule, feeding instructions, and individual care routines for weekdays.
            
            Example User Input:
            - Pet 1:
              - Name: Fluffy
              - Type: Cat
              - ...
            
            Example System Input:
            [System input content]
            
            Example Output:
            [Provide an example runbook section for weekdays here]
            """
        elif choice == "Weekend Only":
            prompt = f"""
            Generate a comprehensive pet sitting runbook for the weekend only.
            
            User Inputs:
            {uploaded_files}
            
            System Input(from PDF):
            {system_info}
            
            Instructions:
            - Create a detailed runbook tailored to the user's pets for the weekend.
            - Include sections for basic information, health, feeding, grooming, and emergency contacts.
            - Adapt the runbook based on the number and types of pets.
            
            Output Format:
            - Use a clear structure with headings for each pet.
            - Provide a schedule for the weekend, focusing on pet care tasks.
            
            Example User Input:
            - Pet 1:
              - Name: Fluffy
              - Type: Cat
              - ...
            
            Example System Input:
            [System input content]
            
            Example Output:
            [Provide an example runbook section for the weekend here]
            """
        else:
            prompt = f"""
            Generate a comprehensive pet sitting runbook based on the following user and system inputs:
            
            User Inputs:
            {uploaded_files}
            
            System Input(from PDF):
            {system_info}
            
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
st.subheader("Runbook Creation")
st.write ("Click the button to generate your persoanlized Runbook")

import streamlit as st

if st.button("Generate Runbook"):
    if user_confirmation:
        # Use Mistral for model inference
        client = Mistral(api_key=api_key)
        
        # Define the prompt as a "chat" message format
        completion = client.chat.complete(
            model="open-mistral-nemo",  # Specify the model ID
            messages=[  # Pass a message format similar to a conversation
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,  # Set the max tokens
            temperature=0.5,  # Control the randomness of the output
        )
        
        # Get the generated text from the response
        output = completion.choices[0].message  # Access the generated message
        
        # Convert `output` to string if it's not already a string
        if isinstance(output, str):
            output_text = output
        else:
            # If output is an object, extract its string representation
            output_text = str(output)  # You can also try accessing specific attributes if needed
        
        st.success("Runbook generated successfully!")
        st.write(output_text)

        # Create a PDF from the output text
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)
        
        # Add the output text to the PDF
        pdf.multi_cell(0, 10, output_text)

        # Save PDF to a BytesIO object to provide as a download
        pdf_output = io.BytesIO()
        pdf.output(pdf_output)
        pdf_output.seek(0)  # Move to the beginning of the PDF
        
        # Provide a download button for the PDF
        st.download_button(
            label="Download Runbook as PDF",
            data=pdf_output,
            file_name="runbook.pdf",
            mime="application/pdf"
        )
    else:
        st.warning("Please confirm the AI prompt before generating the runbook.")