# -- Project information -----------------------------------------------------
project = 'Intelligent Test Case Clustering'
copyright = '2026, Чухліб [твоє прізвище]'
author = 'Чухліб [твоє прізвище]'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',      # automatically extracts docstrings
    'sphinx.ext.napoleon',     # supports Google-style docstrings
    'sphinx.ext.viewcode',     # links to the source code
]

# Way to src/, so that Sphinx can properly process the code
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

# Configuration for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

# Theme
html_theme = 'sphinx_rtd_theme'

# Language
language = 'en'
