"""Setup script for Media Recommender System."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core dependencies only (optional packages commented out in requirements.txt)
core_requirements = [
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "scipy>=1.11.0",
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
    "pydantic>=2.0.0",
]

setup(
    name="media-recommender",
    version="1.0.0",
    author="Coco",
    author_email="coco@northeastern.edu",
    description="A production-ready personalized media content recommendation system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/media-recommender",
    packages=find_packages(where="."),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
    ],
    python_requires=">=3.9",
    install_requires=core_requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "isort>=5.12.0",
        ],
        "nlp": [
            "transformers>=4.30.0",
            "sentence-transformers>=2.2.0",
        ],
        "graph": [
            "torch-sparse",
            "torch-scatter",
            "torch-geometric",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
