import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import pandas as pd

# --- CONFIGURATION ---
SERVER_URL = "http://backend:8000"
st.set_page_config(
    page_title="Automated EDA Assistant",
    page_icon="🔬",
    layout="wide"
)

# --- HELPER FUNCTIONS ---
def check_file_uploaded():
    """Checks if a file has been uploaded via the session state."""
    if "file_uploaded" not in st.session_state or not st.session_state.file_uploaded:
        st.warning("Please upload and load a CSV file first using the sidebar.")
        return False
    return True

def display_model_results(results):
    """Takes the JSON response from the model and displays it nicely."""
    st.success("Model trained successfully!")
    
    # Use columns for a cleaner layout
    col1, col2 = st.columns(2)
    
    r_squared = results.get("r_squared", 0)
    col1.metric(label="R-squared (Model Fit)", value=f"{r_squared:.3f}")
    
    st.info(f"**Interpretation:** {results.get('interpretation', 'N/A')}")
    
    with st.expander("View Full Model Details (JSON)"):
        st.json(results)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🔬 Automated EDA Assistant")
    st.markdown("Upload your data and explore its insights instantly.")
    
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file:
        if st.button("Load Data", key="load_data"):
            with st.spinner("Uploading and caching file..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                try:
                    response = requests.post(f"{SERVER_URL}/upload", files=files)
                    if response.status_code == 200:
                        st.session_state.file_uploaded = True
                        st.success("File loaded and cached!")
                    else:
                        st.session_state.file_uploaded = False
                        st.error(f"Error: {response.json().get('detail', 'Failed to upload.')}")
                except requests.exceptions.ConnectionError:
                    st.error("Connection Error: Backend is not available.")
    
    if "file_uploaded" in st.session_state and st.session_state.file_uploaded:
        st.markdown("---")
        st.header("2. Basic Data Info")
        
        if st.button("Show Info"):
            st.session_state.action = "info"
        if st.button("Show Descriptive Statistics"):
            st.session_state.action = "describe"
        if st.button("Show Missing Values"):
            st.session_state.action = "missing"

# --- MAIN PAGE ---
st.header("EDA & Modeling Workspace")

if not check_file_uploaded():
    st.stop()

# --- TABS FOR ORGANIZATION ---
tab1, tab2, tab3 = st.tabs(["📄 Data Overview", "🎨 Plotting Playground", "🤖 Model Training"])

with tab1:
    st.subheader("Data Overview & Summaries")
    
    # Display results from sidebar actions
    if "action" in st.session_state:
        action = st.session_state.action
        with st.spinner("Fetching data..."):
            if action == "info":
                response = requests.get(f"{SERVER_URL}/info")
                st.text(response.json().get("info"))
            elif action == "describe":
                response = requests.get(f"{SERVER_URL}/describe")
                st.dataframe(pd.DataFrame(response.json()))
            elif action == "missing":
                response = requests.get(f"{SERVER_URL}/missing")
                st.dataframe(pd.DataFrame.from_dict(response.json(), orient='index', columns=['Missing Count']))
    
    st.subheader("Dataset Preview")
    num_rows = st.number_input("Rows to display:", min_value=1, value=5, key="preview_rows")
    if st.button("Refresh Preview"):
        response = requests.get(f"{SERVER_URL}/preview", params={"rows": num_rows})
        st.dataframe(response.json())

with tab2:
    st.subheader("Generate Plots with Natural Language")
    prompt = st.text_input("e.g., 'bar plot of price by room_num' or 'correlation heatmap'", key="nlp_prompt")
    if st.button("Generate Plot"):
        if prompt:
            with st.spinner("Generating plot..."):
                response = requests.get(f"{SERVER_URL}/eda_prompt", params={"prompt": prompt})
                if response.headers.get("content-type") == "image/png":
                    st.image(Image.open(BytesIO(response.content)), caption=f"Plot for: {prompt}")
                else:
                    st.error(f"Error: {response.json().get('error', 'Could not generate plot.')}")
    
    st.markdown("---")
    st.subheader("Download All Generated Plots")
    if st.button("Prepare Download Link"):
        with st.spinner("Zipping plots..."):
            response = requests.get(f"{SERVER_URL}/download_all_plots")
            if response.status_code == 200 and response.headers.get("content-type") == "application/zip":
                st.download_button(
                    label="Click to Download ZIP",
                    data=response.content,
                    file_name="eda_plots.zip",
                    mime="application/zip"
                )
            else:
                st.error(response.json().get("error", "No plots generated yet."))

with tab3:
    st.subheader("Train a Regression Model")
    try:
        response = requests.get(f"{SERVER_URL}/columns")
        if response.status_code == 200:
            columns = response.json().get("columns", [])
            
            # Use columns for layout
            col1, col2 = st.columns(2)

            with col1:
                st.info("Simple Linear Regression (One Variable)")
                simple_y = st.selectbox("Select Dependent Variable (Y):", columns, key="simple_y")
                simple_x = st.selectbox("Select Independent Variable (X):", [c for c in columns if c != simple_y], key="simple_x")
                if st.button("Run Simple Model"):
                    params = {"x_col": simple_x, "y_col": simple_y}
                    response_simple = requests.get(f"{SERVER_URL}/linear_regression", params=params)
                    if response_simple.status_code == 200:
                        display_model_results(response_simple.json())
                    else:
                        st.error(f"Error: {response_simple.json().get('error', 'Failed.')}")

            with col2:
                st.info("Multiple Linear Regression (Multiple Variables)")
                multi_y = st.selectbox("Select Dependent Variable (Y):", columns, key="multi_y")
                multi_x = st.multiselect("Select Independent Variables (X):", [c for c in columns if c != multi_y], key="multi_x")
                if st.button("Run Multiple Model"):
                    if len(multi_x) < 1:
                        st.warning("Please select at least one independent variable.")
                    else:
                        params = [("x_cols", var) for var in multi_x] + [("y_col", multi_y)]
                        response_multi = requests.get(f"{SERVER_URL}/multiple_regression", params=params)
                        if response_multi.status_code == 200:
                            display_model_results(response_multi.json())
                        else:
                            st.error(f"Error: {response_multi.json().get('error', 'Failed.')}")
    except Exception as e:
        st.error(f"An error occurred while loading model controls: {e}")