from __future__ import annotations

from setuptools import find_packages, setup

setup(
    name="selectzyme",
    version="0.0.1",
    author="Felix Moorhoff",
    author_email="fmoorhof@ipb-halle.de",
    url="https://www.ipb-halle.de/en/research/bioorganic-chemistry/research-groups/computational-chemistry/projects/",
    description="Selectzyme: Interactive Protein Space Visualization for Enzyme Discovery, Selection, and Mining",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": ["selectzyme = src.selectzyme.main:main"]
    },  # very important to make local imports work!
)
