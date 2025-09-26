import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import pandas as pd

# Backend server URL (ensure this matches your docker-compose service name)
SERVER_URL = "http://backend:8000"

# --- Page Configuration ---
st.set_page_config(
    page_title="AI EDA Assistant",
    page_icon="📊",
    layout="wide"
)

# --- App Title ---
st.title("📊 AI Exploratory Data Analysis Assistant")
st.markdown("Upload your CSV file and use the sidebar to explore it, or use natural language to generate plots.")

# --- Helper function to check if a file has been uploaded ---
def check_file_uploaded():
    if "file_uploaded" not in st.session_state or not st.session_state.file_uploaded:
        st.warning("Please upload a CSV file first.")
        return False
    return True

# --- Sidebar for File Upload and Basic Info ---
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
    
    if uploaded_file:
        if st.button("Load Data"):
            with st.spinner("Uploading and processing file..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                try:
                    response = requests.post(f"{SERVER_URL}/upload", files=files)
                    if response.status_code == 200:
                        st.session_state.file_uploaded = True
                        st.success("File uploaded successfully!")
                    else:
                        st.session_state.file_uploaded = False
                        st.error(f"Error: {response.json().get('detail', 'Failed to upload file.')}")
                except requests.exceptions.ConnectionError:
                    st.error("Connection Error: Could not connect to the backend. Is it running?")
    
    st.markdown("---")
    
    if "file_uploaded" in st.session_state and st.session_state.file_uploaded:
        st.header("2. Data Actions")
        
        # Button for Data Info
        if st.button("Show Info"):
            response = requests.get(f"{SERVER_URL}/info")
            if response.status_code == 200:
                st.session_state.action = "info"
                st.session_state.data = response.json().get("info")
            else:
                st.error("Failed to get info.")

        # Button for Describe
        if st.button("Show Descriptive Statistics"):
            response = requests.get(f"{SERVER_URL}/describe")
            if response.status_code == 200:
                st.session_state.action = "describe"
                # Convert dict back to DataFrame for better display
                st.session_state.data = pd.DataFrame(response.json())
            else:
                st.error("Failed to get description.")
        
        # Button for Missing Values
        if st.button("Show Missing Values"):
            response = requests.get(f"{SERVER_URL}/missing")
            if response.status_code == 200:
                st.session_state.action = "missing"
                st.session_state.data = pd.DataFrame.from_dict(
                    response.json(), orient='index', columns=['Missing Count']
                )
            else:
                st.error("Failed to get missing values.")

# --- Main Content Area ---

# Display content based on the action selected in the sidebar
if "action" in st.session_state and "data" in st.session_state:
    action = st.session_state.action
    data = st.session_state.data

    if action == "info":
        st.header("Dataset Information")
        st.text(data)
    elif action == "describe":
        st.header("Descriptive Statistics")
        st.dataframe(data)
    elif action == "missing":
        st.header("Missing Values Count")
        st.dataframe(data)

# --- Separator ---
st.markdown("---")


# --- Preview Section ---
st.header("Preview Dataset")
if check_file_uploaded():
    num_rows = st.number_input("Enter number of rows to preview", min_value=1, value=5, key="preview_rows")
    if st.button("Show Preview"):
        response = requests.get(f"{SERVER_URL}/preview", params={"rows": num_rows})
        if response.status_code == 200:
            st.dataframe(response.json())
        else:
            st.error("Could not retrieve preview.")

# --- Separator ---
st.markdown("---")

# --- Natural Language Prompt for Plots ---
st.header("Generate Plots with Natural Language")
if check_file_uploaded():
    prompt = st.text_input("e.g., 'show histogram of Age' or 'correlation heatmap'")
    if st.button("Generate Plot"):
        if prompt:
            with st.spinner("Generating plot..."):
                response = requests.get(f"{SERVER_URL}/eda_prompt", params={"prompt": prompt})
                
                if response.headers.get("content-type") == "image/png":
                    img = Image.open(BytesIO(response.content))
                    st.image(img, caption=f"Plot for: {prompt}", use_column_width=True)
                else:
                    st.error(f"Error: {response.json().get('error', 'Could not generate plot.')}")
        else:
            st.warning("Please enter a prompt.")

# --- Separator ---
st.markdown("---")

# --- Download All Plots ---
st.header("Download Plots")
if check_file_uploaded():
    if st.button("Prepare Download Link for All Plots"):
        response = requests.get(f"{SERVER_URL}/download_all_plots")
        if response.status_code == 200 and response.headers.get("content-type") == "application/zip":
            # Use st.download_button to provide the file to the user
            st.download_button(
                label="Click to Download ZIP",
                data=response.content,
                file_name="eda_plots.zip",
                mime="application/zip"
            )
        else:
            st.error(response.json().get("error", "Failed to get plots or no plots generated yet."))