from setuptools import setup, find_packages

setup(
    name='jax-flows',
    version='0.0.0',
    author='Chris Waites',
    author_email='cwaites10@gmail.com',
    description='Normalizing Flows for JAX',
    license='MIT',
    url='http://github.com/ChrisWaites/jax-flows',
    packages=find_packages(),
    python_requires='>=3.6.0',
)
