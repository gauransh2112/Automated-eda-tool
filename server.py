import io
import os
import uuid
import pandas as pd
import zipfile
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import redis
import pickle
from typing import List
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse, FileResponse
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --- App Setup ---
app = FastAPI(title="EDA Tool API")

# --- Global Dataframe as a Fallback ---
df_global = None

# --- Redis Connection ---
try:
    r = redis.Redis(host='redis', port=6379, db=0, decode_responses=False)
    r.ping()
    print("✅ Successfully connected to Redis.")
    redis_available = True
except redis.exceptions.ConnectionError as e:
    print(f"⚠️ Could not connect to Redis: {e}. Using in-memory fallback.")
    redis_available = False

# --- Directory Setup ---
PLOT_DIR, ZIP_DIR = "plots", "zip_downloads"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(ZIP_DIR, exist_ok=True)

# --- Helper Functions ---
def save_plot(fig):
    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join(PLOT_DIR, filename)
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    return filepath

def get_dataframe(session_id: str):
    global df_global
    if redis_available:
        pickled_df = r.get(session_id)
        return pickle.loads(pickled_df) if pickled_df else None
    return df_global

def set_dataframe(session_id: str, df: pd.DataFrame):
    global df_global
    if redis_available:
        r.set(session_id, pickle.dumps(df))
    else:
        df_global = df

# --------------------------
# API Endpoints
# --------------------------
# --- NEW UPLOAD FUNCTION ---
# In server.py, replace the /upload function

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    session_id = "user123"
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    # --- Smarter Data Cleaning Step ---
    for col in df.columns:
        # If a column is text but looks like it contains only numbers, try to convert it
        if df[col].dtype == 'object':
            try:
                # Attempt to convert to numeric without forcing errors on real text
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                # This column is genuinely categorical text, so we leave it alone
                pass 
    
    set_dataframe(session_id, df)
    return {"filename": file.filename, "message": "File uploaded and cleaned successfully"}

@app.get("/columns")
async def get_columns():
    df = get_dataframe("user123")
    if df is None: return {"error": "No file uploaded yet"}
    return {"columns": df.columns.tolist()}

@app.get("/preview")
async def preview_data(rows: int = 5):
    df = get_dataframe("user123")
    if df is None: return {"error": "No file uploaded yet"}
    preview_df = df.head(rows).replace([np.inf, -np.inf], None).fillna("NA")
    return JSONResponse(content=preview_df.to_dict(orient="records"))

# (The rest of your endpoints: /info, /describe, /missing, /eda_prompt, etc.
# should all start with this line to get the dataframe)
# df = get_dataframe("user123")
# if df is None: return {"error": "No file uploaded yet"}
# ... then proceed with the rest of the function logic.

@app.get("/describe")
async def describe_data():
    df = get_dataframe("user123")
    if df is None:
        return JSONResponse(status_code=404, content={"error": "No file uploaded yet"})
    
    # Use the same logic as before
    description = df.describe(include="all").fillna("").to_dict()
    return JSONResponse(content=description)

@app.get("/missing")
async def missing_values():
    df = get_dataframe("user123")
    if df is None:
        return JSONResponse(status_code=404, content={"error": "No file uploaded yet"})

    # Use the same logic as before
    missing = df.isnull().sum().to_dict()
    return JSONResponse(content=missing)

# --- Example for /info ---
@app.get("/info")
async def get_info():
    df = get_dataframe("user123")
    if df is None: return {"error": "No file uploaded yet"}
    buffer = io.StringIO()
    df.info(buf=buffer)
    return {"info": buffer.getvalue()}
    
# (Make sure to apply the same `get_dataframe` pattern to ALL your other endpoints)

# In server.py, replace your existing eda_prompt function with this

@app.get("/eda_prompt")
async def eda_prompt(prompt: str):
    """
    Handles natural language prompts to generate various EDA plots.
    """
    df = get_dataframe("user123")
    if df is None:
        return {"error": "No file uploaded yet"}

    prompt = prompt.lower()
    
    # Pie Chart for categorical proportions
    if "pie" in prompt or "proportion" in prompt:
        for col in df.select_dtypes(exclude=np.number).columns:
            if col.lower() in prompt:
                if df[col].nunique() > 10:
                    return {"error": f"Column '{col}' has too many unique values for a pie chart. Try a 'count plot of {col}' instead."}
                
                fig, ax = plt.subplots(figsize=(8, 8))
                value_counts = df[col].value_counts()
                ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                ax.set_title(f"Proportion of {col}")
                return FileResponse(save_plot(fig))
        return {"error": "Categorical column for pie chart not found."}

    # Count Plot for categorical data
    elif "count" in prompt:
        for col in df.select_dtypes(exclude=np.number).columns:
            if col.lower() in prompt:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.countplot(y=df[col], ax=ax, order=df[col].value_counts().index)
                ax.set_title(f"Count Plot for {col}")
                plt.tight_layout()
                return FileResponse(save_plot(fig))
        return {"error": "Categorical column for count plot not found."}

    # Bar Plot for numerical vs. categorical
    elif "bar plot" in prompt:
        words = prompt.replace("by", " ").split()
        
        numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c.lower() in words]
        categorical_cols = [c for c in df.select_dtypes(exclude=np.number).columns if c.lower() in words]

        if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
            num_col = numeric_cols[0]
            cat_col = categorical_cols[0]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=df[cat_col], y=df[num_col], ax=ax)
            ax.set_title(f"Average of {num_col} by {cat_col}")
            plt.xticks(rotation=45, ha='right') # Rotate labels to prevent overlap
            plt.tight_layout()
            return FileResponse(save_plot(fig))
            
        return {"error": "For a bar plot, please provide one numeric and one categorical column, like 'bar plot of Salary by Department'."}

    # Correlation heatmap
    elif "correlation" in prompt or "heatmap" in prompt:
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.shape[1] < 2:
            return {"error": "Need at least two numeric columns for a heatmap."}
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        return FileResponse(save_plot(fig))

    # Histogram
    elif "hist" in prompt or "distribution" in prompt:
        for col in df.select_dtypes(include=np.number).columns:
            if col.lower() in prompt:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(df[col].dropna(), kde=True, ax=ax)
                ax.set_title(f"Histogram of {col}")
                return FileResponse(save_plot(fig))
        return {"error": "Numeric column for histogram not found."}

    # Boxplot or Outlier detection
    elif "box" in prompt or "outlier" in prompt:
        for col in df.select_dtypes(include=np.number).columns:
            if col.lower() in prompt:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.boxplot(x=df[col], ax=ax)
                ax.set_title(f"Boxplot of {col}")
                return FileResponse(save_plot(fig))
        return {"error": "Numeric column for boxplot not found."}
        
    # Scatterplot
    elif "scatter" in prompt:
        words = prompt.replace("vs", " ").split()
        cols = [c for c in df.select_dtypes(include=np.number).columns if c.lower() in words]
        if len(cols) >= 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x=df[cols[0]], y=df[cols[1]], ax=ax)
            ax.set_title(f"Scatterplot: {cols[0]} vs {cols[1]}")
            return FileResponse(save_plot(fig))
        return {"error": "Need two numeric column names for scatterplot."}

    # Optimized Pair Plot
    elif "pair" in prompt or "grid" in prompt:
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.shape[1] < 2:
             return {"error": "Need at least two numeric columns for a pairplot."}
        
        # Optimization: If data is large, use a random sample to speed up plotting
        sample_size = 1000
        if len(numeric_df) > sample_size:
            plot_df = numeric_df.sample(n=sample_size, random_state=42)
            title_suffix = f"(based on a random sample of {sample_size} rows)"
        else:
            plot_df = numeric_df
            title_suffix = ""
            
        fig = sns.pairplot(plot_df)
        fig.fig.suptitle(f"Pair Plot {title_suffix}", y=1.02) # y=1.02 raises title
        return FileResponse(save_plot(fig))

    return {"error": "Sorry, I couldn't understand that prompt or find the specified column."}


# --- Linear Regression Endpoint ---
@app.get("/linear_regression")
async def linear_regression(x_col: str, y_col: str):
    df = get_dataframe("user123")
    if df is None:
        return {"error": "No file uploaded yet"}
    # Rest of your regression logic
    df_copy = df.copy()
    df_copy.dropna(subset=[x_col, y_col], inplace=True)
    if len(df_copy) < 2:
        return {"error": "Not enough data to perform regression after dropping missing values."}
    X = df_copy[[x_col]]
    y = df_copy[y_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    coef = model.coef_[0]
    return {
        "model": "Linear Regression",
        "independent_variable": x_col,
        "dependent_variable": y_col,
        "r_squared": score,
        "coefficient": coef,
        "interpretation": f"For a one-unit increase in '{x_col}', '{y_col}' is expected to change by {coef:.2f} units."
    }

# MULTIPLE REGRESSION

@app.get("/multiple_regression")
async def multiple_regression(
    x_cols: List[str] = Query(None), 
    y_col: str = Query(None)
):
    df = get_dataframe("user123")
    if df is None:
        return {"error": "No file uploaded yet"}
    
    all_cols = x_cols + [y_col]
    df_copy = df.copy()
    df_copy.dropna(subset=all_cols, inplace=True)
    
    if len(df_copy) < len(all_cols) + 1:
        return {"error": "Not enough data for multiple regression."}

    X = df_copy[x_cols]
    y = df_copy[y_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    
    # Create a dictionary of coefficients for each feature
    coefficients = {feature: coef for feature, coef in zip(x_cols, model.coef_)}
    
    return {
        "model": "Multiple Linear Regression",
        "independent_variables": x_cols,
        "dependent_variable": y_col,
        "r_squared": score,
        "coefficients": coefficients,
        "interpretation": f"The model's R-squared value is {score:.2f}, indicating it explains {score:.1%} of the variance in '{y_col}'. See individual coefficients for feature effects."
    }

# --- Download Endpoint ---
@app.get("/download_all_plots")
async def download_all_plots():
    files = [os.path.join(PLOT_DIR, f) for f in os.listdir(PLOT_DIR) if f.endswith(".png")]
    if not files:
        return {"error": "No plots available to download"}
    zip_filename = f"eda_plots_{uuid.uuid4().hex}.zip"
    zip_filepath = os.path.join(ZIP_DIR, zip_filename)
    with zipfile.ZipFile(zip_filepath, "w") as zipf:
        for f in files:
            zipf.write(f, os.path.basename(f))
    return FileResponse(
        zip_filepath,
        media_type="application/zip",
        filename=zip_filename
    )