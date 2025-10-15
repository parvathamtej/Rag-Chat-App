# ---- Base Image ----
FROM python:3.11-slim

# ---- Set working directory ----
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


# ---- Copy main app code ----
COPY main.py .

# ---- Expose Streamlit port ----
EXPOSE 8501

# ---- Run Streamlit ----
# We will pass the .env at runtime, not bake it into the image
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
