# 1. Start with a lightweight Python base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy ONLY the requirements first (this optimizes Docker's caching)
COPY requirements.txt .

# 4. Install the strict dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code (app.py, predict.py, etc.)
COPY . .

# 6. Expose the port that FastAPI will run on
EXPOSE 8000

# 7. Define the exact command to start the server when the container boots
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]