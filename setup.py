"""Temporary setup file."""

from setuptools import setup, find_packages

setup(
    name="DUE",
    version="v0.1.0",
    description="A deep learning library for unknown equations",
    author="Junfeng Chen",
    author_email="junfeng.chen22@gmail.com",
    keywords=[
        "Deep learning",
        "Neural networks",
        "Differential equations",
        "Education software"
    ],
    packages=find_packages(),
    install_requires=[
        'torch==1.13',
        'pyyaml',
        'numpy==1.26.4',
        'scipy',
        'matplotlib'
    ],
)
