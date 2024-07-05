import pkg_resources
from setuptools import setup, find_packages
from pathlib import Path

setup(
    name='smart',
    version='0.1',
    author='sutcuremzi',
    description="Stock Market Analysis and Research Tool",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/sutcuremzi/smart',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            Path(__file__).with_name("requirements.txt").open()
        )
    ],
    include_package_data=True,
    python_requires='>=3.7',
)
