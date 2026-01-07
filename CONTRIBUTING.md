# Contributing to pythonradex

Thank you for your interest in contributing to `pythonradex`. All kinds of contributions are welcome, including bug reports, code improvements and documentation updates.

---

## Reporting issues

If you encounter a bug or other problems, please [open an issue](https://github.com/gica3618/pythonradex/issues) with:

- A clear description of the problem
- Steps to reproduce it
- Expected vs. actual behavior
- Version information (`python`, `pythonradex`, OS)

---

## Requesting new features

If you have ideas for new features, feel free to [open an issue](https://github.com/gica3618/pythonradex/issues).

---

## Contributing Code

### Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:

        git clone https://github.com/<your-username>/pythonradex.git

3. **Create a development environment** (use a python version supported by `pythonradex`). You can use either `conda` or `venv`. For example:

        cd pythonradex
        python3 -m venv virt_env_name
        source virt_env_name/bin/activate

4. **Install in editable mode with development dependencies:**

        pip install -e .[dev]

   This installs `pythonradex`, `pytest` and additional packages to modify and build the documentation and run the example notebooks.


### Running Tests

**If you contribute code, please also add corresponding test code whenever possible**. All test code uses `pytest` and is located in the `tests/` folder.

To run the whole test suite, you can for example do:

    python3 -m pytest

To see detailed output:

    python3 -m pytest -v

If you only want to run the tests of a specific test file, for example `test_LAMDA_file.py`:

    python3 -m pytest tests/test_LAMDA_file.py

If you only want to run a specific test:

    python3 -m pytest tests/test_LAMDA_file.py::test_levels

**Please make sure all tests pass before submitting a pull request.**

---

## Contributing to the documentation and example notebooks

Documentation is located in the `docs/` folder and hosted on [ReadTheDocs](https://pythonradex.readthedocs.io/en/latest/).

### Documentation Style Guide

- Use **reStructuredText (`.rst`)** syntax for Sphinx documentation.
- Use **double backticks** for code, commands, and Python package names, e.g.:
  ```
  You can install ``pythonradex`` using ``pip``.
  ```

### Build the documentation locally

To build the documentation locally:

    cd docs
    make html

or on Windows:

    cd docs
    .\make.bat html

The generated HTML files will appear in `docs/_build/html/`. Open `docs/_build/html/index.html` in your browser to preview the site.

### Notebooks

Example notebooks are located in `docs/examples/`. You can run them with

    jupyter notebook docs/examples/

Before submitting a pull request, please clear all output and re-run the notebook fully to ensure it executes cleanly from top to bottom.

---

## Submitting a Pull Request

1. Create a new branch:

        git checkout -b fix-something

2. Make your changes.
3. Run tests to ensure everything passes.
4. Commit with a clear message
5. Push your branch:

        git push origin fix-something

6. Open a **Pull Request** on GitHub against the `main` branch.

---

## License

By contributing to `pythonradex`, you agree that your contributions will be licensed under the [MIT License](./LICENSE).

---

Thank you for contributing to `pythonradex`!


*This file was initially drafted with assistance from ChatGPT (OpenAI GPT-5).*