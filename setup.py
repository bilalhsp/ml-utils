from setuptools import setup, find_packages


setup(
    name="ml_utils",
    version="0.1",
    packages=find_packages(),
    author="Bilal Ahmed",
    author_email="bilalhsp@gmail.com",
    url="https://github.com/bilalhsp/ml-utils.git",
    description="Customizable Functions and objects for ML stuff.",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy', 'tqdm','torch>=2.0.1',
    ],
    python_required='>=3.7',
)


