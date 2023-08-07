from setuptools import setup, find_packages

setup(
    name="ec",
    version="0.0.0",
    author="Felix Moorhoff",
    author_email="fmoorhof@ipb-halle.de",
    url="https://www.ipb-halle.de/en/research/bioorganic-chemistry/research-groups/computational-chemistry/projects/",
    description="Predict EC numbers for protein sequences",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["click"],
    entry_points={"console_scripts": ["ec = src.main:main"]},
)