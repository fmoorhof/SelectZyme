# Use an official Python runtime as a base image
FROM nvcr.io/nvidia/pytorch:22.08-py3

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/requirements.txt

# Install the required Python packages
RUN pip install --no-cache-dir -r /app/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

# Copy the rest of the app code into the container
COPY . /app

# Expose the port Dash will run on
EXPOSE 8050

# Run the Dash app
CMD ["python", "src/main.py"]
