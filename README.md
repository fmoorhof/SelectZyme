# ec
EC number prediction tool

## Directory structure
├── LICENSE
├── README.md
├── setup.py
├── src
      ├── __init__.py
      ├── logic.py
      └── main.py

## Install
For optimal GPU support, the conda installation is reccomended.
### Conda
```
conda env create -f environment.yml
```

### Pip
```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu11==24.2.* cuml-cu11==24.2.*
```
Note: Please install RAPIDSAI CuMl and CuDf manually since otherwise the entire extra-index is installed and that causes the CI runner to exit on: `OSError: [Errno 28] No space left on device`  

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

#### Development tools
This project uses the following tools to improve code quality:
- [black](https://black.readthedocs.io/) for code formatting
- [isort](https://pycqa.github.io/isort/) for sorting imports
- [flake8](https://flake8.pycqa.org/) for style guide enforcement
- [pytest](https://docs.pytest.org/) for testing
- [pytest-cov](https://github.com/pytest-dev/pytest-cov) for measuring code coverage


# License
OpenGPL 3.0 License