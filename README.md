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
### Conda
```
conda create -n ENVNAME "python>=3.7" --file requirements.txt
conda update --name ENVNAME --file requirements.txt
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





#### Development tools
This project uses the following tools to improve code quality:
- [black](https://black.readthedocs.io/) for code formatting
- [isort](https://pycqa.github.io/isort/) for sorting imports
- [flake8](https://flake8.pycqa.org/) for style guide enforcement
- [pytest](https://docs.pytest.org/) for testing
- [pytest-cov](https://github.com/pytest-dev/pytest-cov) for measuring code coverage
