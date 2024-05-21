from setuptools import setup, find_packages

setup(
    name="shmex",
    version="0.1.0",
    url="https://github.com/matsengrp/netam-experiments-1.git",
    author="Matsen Group",
    author_email="ematsen@gmail.com",
    description="Accessory code to train and evaluate SHM models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "biopython",
        "pandas",
        "pyyaml",
        "seaborn",
        "tensorboardX",
        "torch",
        "tqdm",
    ],
    python_requires="==3.9.*",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
    ],
)
