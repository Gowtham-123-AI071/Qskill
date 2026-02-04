FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Ensure nltk stopwords are available
RUN python -c "import nltk; nltk.download('stopwords')"
EXPOSE 8000
CMD ["uvicorn", "backend.src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
