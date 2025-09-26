# Automated EDA Tool with Docker 📊

A web-based Exploratory Data Analysis (EDA) tool that allows users to upload a CSV file and perform various data analysis tasks using a simple, natural language interface. The application is built with a decoupled frontend and backend and is fully containerized using Docker.



## ✨ Features

- **Interactive Frontend:** Built with Streamlit for a clean and responsive user experience.
- **Powerful Backend:** A robust API built with FastAPI to handle all data processing.
- **Natural Language Prompts:** Generate plots like histograms, bar plots, pie charts, heatmaps, and scatter plots just by typing commands.
- **Comprehensive Analysis:** Get instant data previews, descriptive statistics, and missing value reports.
- **Containerized & Reproducible:** The entire application is containerized with Docker and orchestrated with Docker Compose for easy setup and deployment.

## 🛠️ Tech Stack

- **Frontend:** Streamlit, Python
- **Backend:** FastAPI, Pandas, Seaborn, Matplotlib, Python
- **DevOps:** Docker, Docker Compose

## 🚀 Getting Started

To run this project locally, you need to have Docker and Docker Compose installed.

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/gauransh2112/automated-eda-tool-docker.git](https://github.com/gauransh2112/automated-eda-tool-docker.git)
   cd automated-eda-tool-docker

2. Run the application:
   docker-compose up --build
   
3.Access the application:
  Frontend: Open your browser to http://localhost:8501
  Backend API Docs: Open your browser to http://localhost:8000/docs

💬 Example Prompts:
  ---> show histogram of Age
  ---> bar plot of Salary by Department
  ---> pie chart of Country
  ---> correlation heatmap
