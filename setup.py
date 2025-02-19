from setuptools import setup, find_packages

setup(
    name="ec",
    version="v0.0.1",
    author="Felix Moorhoff",
    author_email="fmoorhof@ipb-halle.de",
    url="https://www.ipb-halle.de/en/research/bioorganic-chemistry/research-groups/computational-chemistry/projects/",
    description="Selectzyme: Interactive Protein Space Visualization for Enzyme Discovery, Selection, and Mining",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={"console_scripts": ["ec = src.main:main"]},  # very important to make local imports work!
)