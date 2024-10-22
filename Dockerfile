# more modern containers use cuda 12 and not 11.8. this will cause incompabilities with ocean server - use conda to create python3.10 environment
FROM nvcr.io/nvidia/pytorch:22.08-py3

# Set the working directory in the container
WORKDIR /app

# Copy the environment file into the container
COPY environment_docker.yml /app/environment_docker.yml

RUN conda env create -f environment_docker.yml
RUN conda activate my-env
RUN python --version

# Copy the rest of the app code into the container
COPY . /app

# Expose the port Dash will run on
EXPOSE 8050

# Run the Dash app
CMD ["python", "src/main.py"]
