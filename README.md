# EnzyNavi
Explore and navigate protein sequence space interactively.

## Install
For optimal GPU support, the conda installation is reccomended.
Please clone the repository:
```
git clone https://github.com/fmoorhof/ec.git
```

### Conda
For optimal GPU support, the conda installation is reccomended.
```
conda env create -f environment.yml
```

### Pip
```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu11==24.2.* cuml-cu11==24.2.*
```
Note: Please install RAPIDSAI CuMl and CuDf manually since otherwise the entire extra-index is installed and that causes the docker containers or CI runner to exit on: `OSError: [Errno 28] No space left on device`  

Outlook: pass extra-index-url in pyproject.toml. Doesnt work for me yet!
```
# local installation of the package:
pip install .

# editable install:
pip install -e .

# editable install with optional dependencies:
pip install -e `.[full]`

# on failure, provide extra package indices manually:
pip install -r requirements.txt --extra-index-url=https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu118
```

### Docker
```
docker build -t fmoorhof/enzynavi:rapids23.06-cuda11.8-base-ubuntu22.04-py3.10 .
docker run --gpus all -it --entrypoint /bin/bash fmoorhof/enzynavi

# optional: re-start your container later
docker start CONTAINERID (find CONTAINERID with `docker ps` or `docker ps -a`)
docker exec -it CONTAINERID
docker exec -it CONTAINERID /bin/bash
```

## Usage
```
conda activate ec
# use config.yml files (reccomended for reproducibly saving input)
python app.py --config=results/test_config.yml
# or parameter passing
python src/main.py -p 'argparse_test' -q="ec:1.13.11.85" -q "ec:1.13.11.84" --length '200 TO 601' -loc "/raid/data/fmoorhof/PhD/SideShit/LCP/custom_seqs_no_signals.csv" -o 'argparse_test'
```
Please pay attention if using multiple query words to repeat the `-p` the argument. Please also specify protein sequence length filters in the mentioned format 'from TO to'

Additional information about the parsing options can be displayed by:
```
python src/main.py -h
```


#### Development tools
This project uses the following tools to improve code quality:
- [black](https://black.readthedocs.io/) for code formatting
- [isort](https://pycqa.github.io/isort/) for sorting imports
- [flake8](https://flake8.pycqa.org/) for style guide enforcement
- [pytest](https://docs.pytest.org/) for testing
- [pytest-cov](https://github.com/pytest-dev/pytest-cov) for measuring code coverage


# License
OpenGPL 3.0 License