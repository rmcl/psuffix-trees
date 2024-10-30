from setuptools import setup, find_packages

setup(
    name="pypst",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],  # List any dependencies here
    python_requires=">=3.7",
    description="A Python library for creating Probabilistic Suffix Trees (PST)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Russell McLoughlin",
    author_email="russ.mcl@gmail.com",
    url="https://github.com/rmcl/pypst",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
