from setuptools import setup, find_packages

setup(
    name="torch-stub",
    version="0.1.0",
    description="Minimal torch/torchvision/accelerate stubs for import compatibility",
    packages=find_packages(),
    python_requires=">=3.10",
)
