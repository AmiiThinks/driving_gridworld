from setuptools import setup, find_packages

setup(
    name='driving_gridworld',
    version='0.0.1',
    license='',
    packages=find_packages(),
    install_requires=[
        'future >= 0.15.2',
        'setuptools >= 20.2.2',
        'fire',
        'pycolab',
        'tensorflow >= 2', 
    ],
    tests_require=['pytest', 'pytest-cov'],
    setup_requires=['pytest-runner'],
)
