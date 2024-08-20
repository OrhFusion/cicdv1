
# Use a slim Python image for a smaller footprint
FROM python:3.11-slim-buster
# Set the working directory for the application
WORKDIR /app
# Copy requirements.txt for dependency installation
COPY requirements.txt ./
# Install dependencies using pip
RUN pip install --default-timeout=350 --no-cache-dir -r requirements.txt


# Copy application files and model
COPY . .
# Expose the port for Flask app (adjust if needed)
EXPOSE 5000  
# Set the command to run the Flask app
CMD ["flask", "run", "--host", "0.0.0.0"]
