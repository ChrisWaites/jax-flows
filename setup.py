from setuptools import setup, find_packages
from pip._internal.req import parse_requirements

with open('README.md', 'r') as f:
    long_description = f.read()

install_reqs = parse_requirements('requirements.txt', session=False)
requirements = [str(ir.req) for ir in install_reqs]

extras = {}
extras['docs'] = ['recommonmark', 'sphinx', 'sphinx-markdown-tables', 'sphinx-rtd-theme']
extras['testing'] = ['pytest', 'pytest-xdist']
extras['quality'] = ['black', 'isort', 'flake8']
extras['dev'] = extras['testing'] + extras['quality']

setup(
    name='jax-flows',
    version='0.0.0',
    author='Chris Waites',
    author_email='cwaites10@gmail.com',
    description='Normalizing Flows for JAX',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    url='http://github.com/ChrisWaites/jax-flows',
    packages=find_packages(),
    install_requires=[
        'jax',
        'jaxlib',
        'numpy',
        'scipy',
    ],
    extras_require=extras,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6.0',
)
