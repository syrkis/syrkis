from setuptools import setup, find_packages

# Read the contents of your requirements.txt file
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f 
                    if line.strip() and not line.startswith('#') 
                    and not line.startswith('-e')]

setup(
    name='syrkis',
    version='0.1.0',
    packages=find_packages(),
    install_requires=requirements,
)
