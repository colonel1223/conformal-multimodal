from setuptools import setup, find_packages
setup(
    name="conformal-multimodal",
    version="0.1.0",
    author="Spencer Cottrell",
    packages=find_packages(),
    install_requires=["torch>=2.0.0", "numpy>=1.24.0"],
)
