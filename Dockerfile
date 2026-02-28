FROM python:3.12-alpine
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
COPY templates/ ./templates/
COPY static/ ./static/
COPY migrate/ ./migrate/
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
