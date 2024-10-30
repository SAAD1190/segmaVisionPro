from setuptools import setup, find_packages
import os

# Function to read the requirements file and install packages
def parse_requirements(filename):
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

setup(
    name='segmaVisionPro',
    version='1.0',
    description='SegmaVisionPro is a project focused on advanced object detection, segmentation, and prompt generation',
    packages=find_packages(),  # This will automatically find all Python packages in your project
    install_requires=parse_requirements('requirements.txt'),
)
