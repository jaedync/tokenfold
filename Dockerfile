FROM python:3.12-alpine
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
COPY templates/ ./templates/
COPY static/ ./static/
COPY migrate/ ./migrate/
COPY client/ ./client/
# An open dashboard holds an SSE stream that never ends by itself: without a
# graceful timeout, uvicorn waits on it forever and Docker kills the process
# before the lifespan shutdown runs (app/tests/test_graceful_shutdown.py).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000", "--timeout-graceful-shutdown", "5"]
