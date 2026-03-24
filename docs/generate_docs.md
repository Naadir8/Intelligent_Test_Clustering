# Project Documentation Generation (Lab Work 5)

## Objective

This document describes the process of automatically generating project documentation using **Sphinx**. The documentation is generated directly from in-code docstrings written in **Google style**.

---

## Tools Used

* **Sphinx**
* Extensions:

  * `sphinx.ext.autodoc`
  * `sphinx.ext.napoleon`
* Theme:

  * **Read the Docs**

---

## Documentation Structure

* `docs/source/` — source `.rst` files (auto-generated)
* `docs/build/html/` — final HTML documentation (excluded from version control)
* `docs/conf.py` — main Sphinx configuration file

---

## Documentation Generation Instructions

### 1. Initial Setup (Run Once)

Navigate to the project root and execute:

```bash
cd docs
make clean
```

---

### 2. Update Module Documentation

Run this step after any changes to the codebase or docstrings:

```bash
sphinx-apidoc -o source ../src --separate --module-first --force
```

This command automatically generates or updates module documentation files (e.g., `loader.rst`, `embedder.rst`, `clusterer.rst`, etc.).

---

### 3. Generate HTML Documentation

Run the following command:

```bash
make html
```

Alternatively, execute everything in one command from the project root:

```bash
cd docs && make clean && sphinx-apidoc -o source ../src --separate --module-first --force && make html
```

---

### 4. View Generated Documentation

Open the following file in a web browser:

```
docs/build/html/index.html
```

Example (absolute path):

```
D:\Intelligent_Test_Clustering\docs\build\html\index.html
```

---

## Updating Documentation After Code Changes

Each time you:

* add a new public class or method,
* modify a docstring,
* add a new module,

run:

```bash
cd docs && sphinx-apidoc -o source ../src --separate --module-first --force && make html
```

---

## Documentation Standards

All contributors must follow these rules:

* Use **Google-style docstrings** (PEP 257 + Google Python Style Guide)
* Include the following sections where applicable:

  * `Args`
  * `Returns`
  * `Raises`
  * `Example`
* Document all public classes, methods, and functions
* Use **type hints** consistently

---

## Documentation Locations

* Source files: `docs/source/`
* Generated HTML: `docs/build/html/`
* Main entry point: `docs/build/html/index.html`

---
