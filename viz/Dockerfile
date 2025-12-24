# Dockerfile for Dash Visualization Dashboard
# Deploy to Cloud Run

FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt requirements-viz.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt -r requirements-viz.txt

# Copy application code
COPY config ./config
COPY viz ./viz
COPY data_module ./data_module

# Set environment variables
ENV PORT=8080
ENV DASH_DEBUG=False
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "viz/dash_app.py"]
