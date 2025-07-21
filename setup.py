from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='custom-cv',
    version='0.1.1',
    author='Akinsanya Joel',
    author_email='akinsanyajoel82@gmail.com',
    description='A Python library for specialized cross-validation strategies (stratified, time-series, spatial).',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/kiojoel/custom-cv-lib',

    package_dir={"": "src"},
    packages=find_packages(where="src"),

    install_requires=[
        'scikit-learn',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scipy',
    ],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
)