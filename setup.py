from setuptools import setup, find_packages

# Function to read the requirements.txt file
def read_requirements():
    with open("requirements.txt", "r") as f:
        return [
            line.strip() for line in f 
            if line.strip() and not line.startswith("--")
        ]

setup(
    name="nn-dataset",
    version="0.1.0",
    description="A neural network dataset management tool",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="ABrain-One",
    author_email="dmytro.ignatov@uni-wuerzburg.de",
    url="https://github.com/ABrain-One/nn-dataset",
    packages=find_packages(include=['ab', 'ab.nn', 'ab.nn.*']),
    package_dir={'ab': 'ab'},
    install_requires=read_requirements(),  # Dynamically load dependencies from requirements.txt
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
     package_data={
        "ab.nn.stat": ["**/*.json"],  # Include all JSON files under `ab/nn/stat/`
    },
    include_package_data=True,
)