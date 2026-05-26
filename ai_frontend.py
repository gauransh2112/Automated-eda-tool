import streamlit as st
import requests
import os
from PIL import Image
from io import BytesIO
import pandas as pd

# --- CONFIGURATION ---
SERVER_URLS = [
    url.strip().rstrip("/")
    for url in os.getenv("SERVER_URL", "http://backend:8000,http://localhost:8000").split(",")
    if url.strip()
]
st.set_page_config(
    page_title="Automated EDA Assistant",
    page_icon="🔬",
    layout="wide"
)

# --- HELPER FUNCTIONS ---

def safe_get_error(response, default="Something went wrong"):
    try:
        return response.json().get("detail") or response.json().get("error") or default
    except:
        return response.text or default


def check_file_uploaded():
    if "file_uploaded" not in st.session_state or not st.session_state.file_uploaded:
        st.warning("Please upload and load a CSV file first using the sidebar.")
        return False
    return True


def display_model_results(results):
    if "error" in results:
        st.error(results["error"])
        return

    st.success("Model trained successfully!")
    
    col1, col2 = st.columns(2)
    
    r_squared = results.get("r_squared") or 0
    col1.metric(label="R-squared (Model Fit)", value=f"{r_squared:.3f}")
    
    st.info(f"**Interpretation:** {results.get('interpretation', 'N/A')}")
    
    with st.expander("View Full Model Details (JSON)"):
        st.json(results)


def backend_request(method, endpoint, **kwargs):
    last_error = None
    timeout = kwargs.pop("timeout", 30)

    for server_url in SERVER_URLS:
        try:
            response = requests.request(
                method,
                f"{server_url}{endpoint}",
                timeout=timeout,
                **kwargs,
            )
            st.session_state.server_url = server_url
            return response
        except requests.exceptions.RequestException as exc:
            last_error = exc

    raise requests.exceptions.ConnectionError(last_error)

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
                    response = backend_request("POST", "/upload", files=files)
                    
                    if response.status_code == 200:
                        st.session_state.file_uploaded = True
                        st.success("File loaded and cached!")
                    else:
                        st.session_state.file_uploaded = False
                        st.error(f"Error: {safe_get_error(response)}")

                except requests.exceptions.ConnectionError:
                    st.error("Connection Error: Backend is not available. Start the FastAPI server on port 8000, or set SERVER_URL.")

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

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📄 Data Overview", "🎨 Plotting Playground", "🤖 Model Training"])

# --- TAB 1 ---
with tab1:
    st.subheader("Data Overview & Summaries")
    
    if "action" in st.session_state:
        action = st.session_state.action
        with st.spinner("Fetching data..."):
            response = backend_request("GET", f"/{action}")
            
            if response.status_code != 200:
                st.error(safe_get_error(response))
            else:
                data = response.json()
                
                if action == "info":
                    st.text(data.get("info"))
                elif action == "describe":
                    st.dataframe(pd.DataFrame(data))
                elif action == "missing":
                    st.dataframe(pd.DataFrame.from_dict(data, orient='index', columns=['Missing Count']))

    st.subheader("Dataset Preview")
    num_rows = st.number_input("Rows to display:", min_value=1, value=5, key="preview_rows")

    if st.button("Refresh Preview"):
        response = backend_request("GET", "/preview", params={"rows": num_rows})
        
        if response.status_code == 200:
            st.dataframe(response.json())
        else:
            st.error(safe_get_error(response))

# --- TAB 2 ---
with tab2:
    st.subheader("Generate Plots with Natural Language")
    prompt = st.text_input("e.g., 'bar plot of price by room_num' or 'correlation heatmap'", key="nlp_prompt")

    if st.button("Generate Plot"):
        if prompt:
            with st.spinner("Generating plot..."):
                response = backend_request("GET", "/eda_prompt", params={"prompt": prompt}, timeout=120)

                if response.headers.get("content-type") == "image/png":
                    st.image(Image.open(BytesIO(response.content)), caption=f"Plot for: {prompt}")
                else:
                    st.error(f"Error: {safe_get_error(response)}")

    st.markdown("---")
    st.subheader("Download All Generated Plots")

    if st.button("Prepare Download Link"):
        with st.spinner("Zipping plots..."):
            response = backend_request("GET", "/download_all_plots", timeout=120)

            if response.status_code == 200 and response.headers.get("content-type") == "application/zip":
                st.download_button(
                    label="Click to Download ZIP",
                    data=response.content,
                    file_name="eda_plots.zip",
                    mime="application/zip"
                )
            else:
                st.error(safe_get_error(response))

# --- TAB 3 ---
with tab3:
    st.subheader("Train a Regression Model")

    try:
        response = backend_request("GET", "/columns")

        if response.status_code != 200:
            st.error(safe_get_error(response))
        else:
            data = response.json()
            columns = data.get("numeric_columns") or data.get("columns", [])

            if len(columns) < 2:
                st.warning("Regression needs at least two numeric columns.")
                st.stop()
            
            col1, col2 = st.columns(2)

            # --- SIMPLE MODEL ---
            with col1:
                st.info("Simple Linear Regression (One Variable)")
                simple_y = st.selectbox("Select Dependent Variable (Y):", columns, key="simple_y")
                simple_x = st.selectbox("Select Independent Variable (X):", [c for c in columns if c != simple_y], key="simple_x")

                if st.button("Run Simple Model"):
                    params = {"x_col": simple_x, "y_col": simple_y}
                    response_simple = backend_request("GET", "/linear_regression", params=params)

                    if response_simple.status_code == 200:
                        display_model_results(response_simple.json())
                    else:
                        st.error(safe_get_error(response_simple))

            # --- MULTIPLE MODEL ---
            with col2:
                st.info("Multiple Linear Regression (Multiple Variables)")
                multi_y = st.selectbox("Select Dependent Variable (Y):", columns, key="multi_y")
                multi_x = st.multiselect("Select Independent Variables (X):", [c for c in columns if c != multi_y], key="multi_x")

                if st.button("Run Multiple Model"):
                    if len(multi_x) < 1:
                        st.warning("Please select at least one independent variable.")
                    else:
                        params = [("x_cols", var) for var in multi_x] + [("y_col", multi_y)]
                        response_multi = backend_request("GET", "/multiple_regression", params=params)

                        if response_multi.status_code == 200:
                            display_model_results(response_multi.json())
                        else:
                            st.error(safe_get_error(response_multi))

    except Exception as e:
        st.error(f"An error occurred while loading model controls: {e}")
