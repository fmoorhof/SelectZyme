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
# local installation of the package:
pip install .

# editable install:
pip install -e .

# editable install with optional dependencies:
pip install -e `.[full]`
```

Warning GPU support with CuDF and CuML are unfortunately only available (stable) via conda :/
However, you could still try to install it seperately by:
```
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu11==23.12.* cuml-cu11==23.12.* cugraph-cu11==23.12.*
```




#### Development tools
This project uses the following tools to improve code quality:
- [black](https://black.readthedocs.io/) for code formatting
- [isort](https://pycqa.github.io/isort/) for sorting imports
- [flake8](https://flake8.pycqa.org/) for style guide enforcement
- [pytest](https://docs.pytest.org/) for testing
- [pytest-cov](https://github.com/pytest-dev/pytest-cov) for measuring code coverage
