#!/usr/bin/env python3

"""
Setup script for Class Transcriber project.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="class-transcriber",
    version="2.3.0",
    author="Karlis Benefelds",
    description="A tool for transcribing lecture audio with forum integration and report generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/class-transcriber",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "class-transcriber=src.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)