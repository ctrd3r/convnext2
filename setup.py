#!/usr/bin/env python3
"""
Setup script for ConvNeXt Earthquake Precursor Detection.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip() 
        for line in requirements_path.read_text().splitlines() 
        if line.strip() and not line.startswith("#")
    ]
else:
    requirements = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "Pillow>=9.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "PyYAML>=6.0",
    ]

setup(
    name="convnext-earthquake-precursor",
    version="1.0.0",
    author="Earthquake Prediction Research Team",
    author_email="your.email@example.com",
    description="ConvNeXt-based earthquake precursor detection from geomagnetic spectrograms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/convnext-earthquake-precursor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
        ],
        "visualization": [
            "grad-cam>=1.4.0",
            "captum>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "convnext-train=scripts.train:main",
            "convnext-evaluate=scripts.evaluate:main",
            "convnext-predict=scripts.predict:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
