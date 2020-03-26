from setuptools import setup, find_packages
from pip.req import parse_requirements

install_reqs = parse_requirements('requirements.txt')
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
    license='MIT',
    url='http://github.com/ChrisWaites/jax-flows',
    packages=find_packages(),
    extras_require=extras,
    install_requires=requirements,
    python_requires='>=3.6.0',
)
