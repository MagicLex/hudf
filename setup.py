setup(
    name="hudf",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "scikit-learn>=0.24.0",  
    ],
    author="Hopsworks",
    description="Hopsworks User Defined Functions - Common utilities for feature engineering",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/magiclex/hudf",
    python_requires=">=3.7",
)
