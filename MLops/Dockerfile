FROM python:3.12-slim

WORKDIR /app

# Copy correct requirements file
COPY req.txt .
RUN pip install --no-cache-dir -r req.txt

# Copy all code and ML artifacts
COPY . .

# FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]