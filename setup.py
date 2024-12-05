from setuptools import setup, find_packages

setup(
    name="dataset_tools",
    version="0.1.0",
    author="huangqj23",
    author_email="huangquanjin24@gmail.com",
    description="A toolkit for converting and visualizing various dataset formats",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/huangqj23/dataset_tools",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.18.0",
        "opencv-python>=4.5.0",
        "tqdm>=4.45.0",
    ],
) 