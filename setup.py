from setuptools import setup, find_packages

setup(
    name='stms-filler',
    version='0.1.0',
    author='Bayu Suseno',
    author_email='bayu.suseno@outlook.com',
    description='Spatiotemporal Filling and Multistep Smoothing for satellite time series reconstruction',
    long_description=open("README.md").read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/stms-filler',
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
