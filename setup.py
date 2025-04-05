from setuptools import setup, find_packages

setup(
    name='stms',
    version='0.2.0',  
    author='Bayu Suseno',
    author_email='bayu.suseno@outlook.com',
    description='Spatiotemporal Filling and Multistep Smoothing for satellite time series reconstruction',
    long_description=open("README.md", encoding="utf-8").read(),  
    long_description_content_type='text/markdown',
    url='https://github.com/byususen/stms',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pygam',
        'tqdm',
        'matplotlib'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
