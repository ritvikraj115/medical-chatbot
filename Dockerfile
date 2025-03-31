# Use an official Python runtime as a base image
FROM python:3.8-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the entire repository into the container
COPY . /app/

# Expose port 8080 (Flask will run on this port)
EXPOSE 8080

# Run store_index.py first to build the vector store, then start the Flask app
CMD ["sh", "-c", "python store_index.py && python app.py"]
