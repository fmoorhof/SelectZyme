# more modern containers use cuda 12 and not 11.8. this will cause incompabilities with ocean server - use conda to create python3.10 environment
FROM nvcr.io/nvidia/pytorch:22.08-py3
# FROM nvcr.io/nvidia/rapidsai/rapidsai:23.06-cuda11.8-runtime-ubuntu22.04-py3.10

# Set the working directory in the container
WORKDIR /app

COPY environment.yml /app/environment.yml
RUN conda env create -f environment.yml

# Set the PATH to use the newly created conda environment (conda activate not possible since conda init <SHELL_NAME> problem)
ENV PATH /opt/conda/envs/my-env/bin:$PATH
RUN python --version

# Copy the rest of the app code into the container
COPY . /app

# Expose the port Dash will run on
EXPOSE 8050

# Run the Dash app
CMD ["python", "src/main.py"]
