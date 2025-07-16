 # Use official Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code and model files
COPY main.py .
COPY ckd_model.pt .
COPY scaler.pkl .

# Expose port for FastAPI (default 8000)
EXPOSE 8000

# Command to run FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
