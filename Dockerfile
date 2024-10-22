# more modern containers use cuda 12 and not 11.8. this will cause incompabilities with ocean server - use conda to create python3.10 environment
FROM nvcr.io/nvidia/pytorch:22.08-py3

# Set the working directory in the container
WORKDIR /app

# Copy the environment file into the container
COPY environment_docker.yml /app/environment_docker.yml

# Install the required Python packages and create the environment
RUN conda env create -f environment_docker.yml

# Set the PATH to use the newly created conda environment (conda activate not possible since conda init <SHELL_NAME> problem)
ENV PATH /opt/conda/envs/my-env/bin:$PATH
RUN python --version

# Copy the rest of the app code into the container
COPY . /app

# Expose the port Dash will run on
EXPOSE 8050

# Run the Dash app
CMD ["python", "src/main.py"]
