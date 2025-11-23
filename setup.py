from setuptools import setup, find_packages

setup(
    name="portfolio-gambling-theory",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "cvxpy>=1.3.0",
    ],
)
