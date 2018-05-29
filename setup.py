from setuptools import setup, find_packages

setup(
    name='driving_gridworld',
    version='0.0.1',
    license='',
    packages=find_packages(),
    install_requires=[
        'future == 0.15.2',
        'setuptools >= 20.2.2',
        # 'pyyaml == 3.12',
        # tensorflow or tensorflow-gpu v1.2
        'fire',
        'pycolab'
    ],
    tests_require=['pytest', 'pytest-cov'],
    setup_requires=['pytest-runner'],
)
