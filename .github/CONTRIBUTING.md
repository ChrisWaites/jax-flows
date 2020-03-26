## How do I contribute to `jax-flows`?

Great to hear you're interested in contributing to the project! First, you'll need to fork this repository and make a branch for your edits.

### Setting up your environment

To make changes and run things, you'll likely need to install the necessary dependencies.

```
pip install .[dev]
```

### Tests and documentation

In general, code committed to this repository needs to be tested and documented. That means including two things, namely:

- Explicit unit tests in the `tests` directory
- Docstrings for public facing functions
    - These should include at least a rudimentary `doctest`.
    - Docstrings follow the [Google style guide](https://google.github.io/styleguide/pyguide.html).

To run the tests, in the root of the repository:

```
make test
```

### Style

We also adhere to strict style guidelines.

To run a linter which will automatically format a majority of your code:

```
make style
```

This will do many but not all things. Ultimately, your changes will need to pass the following quality check:

```
make quality
```

### Submitting a PR

Once you've made your changes, push and submit a pull request, filling out the necessary information. An admin will review your branch shortly and accept it if they see fit!

