"""
Setup script for video-deid package

This setup.py file is provided for backward compatibility with
older tools that do not yet support pyproject.toml directly.
"""
from setuptools import setup

# Let setuptools read metadata from pyproject.toml
setup(
    # All configuration is in pyproject.toml
    # PEP 621 compliant setuptools will read it automatically
    packages=["video_deid"],
    package_dir={"": "src"},
)
