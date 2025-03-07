"""
Setup script for video-deid package
"""
from setuptools import setup, find_packages

setup(
    name="video-deid",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "opencv-python",
        "tqdm",
        "ultralytics",
    ],
    entry_points={
        "console_scripts": [
            "video-deid=video_deid.cli:main",
        ],
    },
    python_requires=">=3.7",
    author="Author",
    author_email="sriharshamvs@gmail.com",
    description="A package for de-identifying faces in videos",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kbjohnson-penn/video-deid.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)