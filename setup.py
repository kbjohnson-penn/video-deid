"""
Setup script for video-deid package

This setup.py file is provided for backward compatibility with
older tools that do not yet support pyproject.toml directly.
"""
from setuptools import setup, find_namespace_packages

# Let setuptools read metadata from pyproject.toml
setup(
    # All configuration is in pyproject.toml
    # PEP 621 compliant setuptools will read it automatically
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
)
