from setuptools import find_packages, setup

setup(
    name="straw",
    version="0.1.0",
    description="Straw: A preprocessing and filtering tool for the "
    "Pile and other large-scale natural language corpora",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
    ],
    setup_requires=["setuptools>=18.0"],
    install_requires=["numpy", "tokenizers", "tqdm"],
    packages=find_packages(exclude=["fiction", "fiction.*",]),
    package_data={"straw": ["*.json", "*.pkl"]},
    entry_points={
        "console_scripts": [
            "straw-normalize = straw_cli.normalize:cli_main",
            "straw-filter-lang = straw_cli.filter_lang:cli_main",
            "straw-process-pile = straw_cli.main:cli_main",
        ],
    },
    include_package_data=True,
)
