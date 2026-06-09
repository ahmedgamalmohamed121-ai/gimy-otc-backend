# Use official lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Create a non-root user (Hugging Face recommends user ID 1000)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory inside the container
WORKDIR /app

# Copy requirements file first and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the backend files
COPY --chown=user . .

# Expose the default Hugging Face Space port
EXPOSE 7860

# Run the FastAPI application
CMD ["python", "main.py"]
