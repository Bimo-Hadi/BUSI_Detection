FROM python:3.11-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY models ./models
ENV BEEONIC_MODEL_DIR=/srv/models
EXPOSE 8000
# Set BEEONIC_API_KEY at run time:  docker run -e BEEONIC_API_KEY=... -p 8000:8000 beeonic-api
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
