# Use an official Python base image
FROM python:3.9-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir torch torchvision flask pillow

# Expose port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
