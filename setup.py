from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ZeroModel",
    version="1.0.2",
    author="Ernan Hughes",
    author_email="ernanhughes@gmail.com",
    description="Zero-Model Intelligence: Spatially-optimized visual policy maps",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ernanhughes/zeromodel",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "PyWavelets>=1.1.1",
        "pyyaml",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "matplotlib>=3.4.0",
            "imageio>=2.9.0",
            "scikit-learn>=0.24.0",
            "pandas>=1.2.0",
            "ipykernel>=5.3.0",
            
        ],
    },
    entry_points={
        "console_scripts": [
            "zeromodel-demo=zeromodel.demo:demo_zeromodel",
        ],
    },
)