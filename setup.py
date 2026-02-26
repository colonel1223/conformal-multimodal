from setuptools import setup, find_packages

setup(
    name="conformal-multimodal",
    version="0.1.0",
    description="Conformal prediction for multimodal uncertainty quantification",
    author="Spencer Kitaro Cottrell",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["numpy>=1.21"],
)
