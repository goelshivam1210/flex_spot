from setuptools import setup, find_packages

setup(
    name="flex_spot_robot",
    version="0.1.0",
    description="Execution interface to deploy trained policies on Spot using its SDK",
    author="Shivam Goel",
    author_email="shivam.goel@tufts.edu",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'PyYAML',
        'requests',
        'bosdyn-client',
        'bosdyn-core',
        'bosdyn-api'
    ],
    python_requires='>=3.8',
)
