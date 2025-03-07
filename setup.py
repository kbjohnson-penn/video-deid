"""
Setup script for video-deid package
"""
from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read the long description from README.md
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="video-deid",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "flake8>=6.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
            "mkdocs>=1.4.0",
            "mkdocs-material>=9.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "video-deid=video_deid.cli:main",
        ],
    },
    python_requires=">=3.7",
    author="Author",
    author_email="sriharshamvs@gmail.com",
    description="A tool for de-identifying faces in videos while preserving body movements",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kbjohnson-penn/video-deid",
    project_urls={
        "Bug Tracker": "https://github.com/kbjohnson-penn/video-deid/issues",
        "Documentation": "https://github.com/kbjohnson-penn/video-deid",
        "Source Code": "https://github.com/kbjohnson-penn/video-deid",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Video",
        "Topic :: Security",
    ],
    keywords="video, face, de-identification, privacy, anonymization, blur, yolo",
)