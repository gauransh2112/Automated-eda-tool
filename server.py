import io
import os
import uuid
import pandas as pd
import numpy as np 
import zipfile
import seaborn as sns
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse

# --------------------------
# FastAPI App
# --------------------------
app = FastAPI(title="EDA Tool API")

# Global dataframe
df = None

# Folder for saving plots
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)
ZIP_DIR = "zip_downloads"
os.makedirs(ZIP_DIR, exist_ok=True)


def save_plot(fig):
    """Helper to save matplotlib figures with unique filenames"""
    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join(PLOT_DIR, filename)
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    return filepath


# --------------------------
# File Upload
# --------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global df
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    return {"filename": file.filename, "message": "File uploaded successfully"}


# --------------------------
# Basic Data Info
# --------------------------
@app.get("/info")
async def get_info():
    global df
    if df is None:
        return {"error": "No file uploaded yet"}
    buffer = io.StringIO()
    df.info(buf=buffer)
    return {"info": buffer.getvalue()}


@app.get("/preview")
async def preview_data(rows: int = 5):
    global df
    if df is None:
        return {"error": "No file uploaded yet"}

    # Get the first 'rows' of the dataframe
    preview_df = df.head(rows)

    # Replace infinite values with None, which becomes 'null' in JSON
    preview_df = preview_df.replace([np.inf, -np.inf], None)

    # Fill any remaining NaN values with a string placeholder
    preview_df = preview_df.fillna("NA")

    # Convert to dictionary and return the JSON response
    return JSONResponse(content=preview_df.to_dict(orient="records"))


@app.get("/describe")
async def describe_data():
    global df
    if df is None:
        return {"error": "No file uploaded yet"}
    return JSONResponse(df.describe(include="all").fillna("").to_dict())


@app.get("/missing")
async def missing_values():
    global df
    if df is None:
        return {"error": "No file uploaded yet"}
    missing = df.isnull().sum()
    return JSONResponse(missing.to_dict())


# --------------------------
# EDA Prompt Endpoint
# --------------------------
# In server.py

@app.get("/eda_prompt")
async def eda_prompt(prompt: str):
    """
    Provide a natural language prompt like:
    - 'show histogram of Age'
    - 'pie chart of Department'
    - 'bar plot of Salary by Department'
    - 'correlation heatmap'
    - 'scatter Age vs Salary'
    - 'count of Department'
    - 'pairplot'
    """
    global df
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
                ax.axis('equal')
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

    # --- New Plot: Bar Plot for numerical vs. categorical ---
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

    # Boxplot
    elif "box" in prompt:
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
        return {"error": "Need two column names for scatterplot."}

    # Outlier detection (using boxplot)
    elif "outlier" in prompt:
        for col in df.select_dtypes(include=np.number).columns:
            if col.lower() in prompt:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.boxplot(x=df[col], ax=ax)
                ax.set_title(f"Outlier Detection for {col}")
                return FileResponse(save_plot(fig))
        return {"error": "Numeric column for outlier detection not found."}

    # Optimized Pair Plot
    elif "pair" in prompt or "grid" in prompt:
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.shape[1] < 2:
             return {"error": "Need at least two numeric columns for a pairplot."}
        
        sample_size = 1000
        if len(numeric_df) > sample_size:
            plot_df = numeric_df.sample(n=sample_size)
            title_suffix = f"(based on a random sample of {sample_size} rows)"
        else:
            plot_df = numeric_df
            title_suffix = ""
            
        fig = sns.pairplot(plot_df)
        fig.fig.suptitle(f"Pair Plot {title_suffix}", y=1.02)
        return FileResponse(save_plot(fig))

    return {"error": "Sorry, I couldn't understand that prompt or find the specified column."}

# Endpoint to download all plots as ZIP
# --------------------------
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

