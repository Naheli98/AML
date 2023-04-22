# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory to /app
WORKDIR AML_Naheli\AML\AML_assignment_4

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code to the working directory
COPY app.py .
COPY score.py .
COPY model.joblib .

# Expose the port the app will run on
EXPOSE 8000

# Run the command to start the app
CMD ["python", "app.py"]

