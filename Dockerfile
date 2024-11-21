# rapids version: 23.06 (includes cuml,cudf,...), cuda11.8, python3.10 (builds in 5mins without cache) [missing pytorch]
FROM nvcr.io/nvidia/rapidsai/rapidsai:23.06-cuda11.8-base-ubuntu22.04-py3.10

# Set the working directory in the container
WORKDIR /app

COPY . /app
COPY environment_docker.yml /app/environment_docker.yml

# rapids env needs to be used unfortunately, else some packages are missing (cudf etc.)
RUN conda env update --name rapids --file environment_docker.yml

# Expose the port Dash will run on
EXPOSE 8050

# Run the Dash app
CMD ["python", "src/main.py"]
