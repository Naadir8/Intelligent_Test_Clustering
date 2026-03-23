# -- Project information -----------------------------------------------------
project = 'Intelligent Test Case Clustering'
copyright = '2026, Чухліб Євгеній ІН-23'
author = 'Чухліб Євгеній ІН-23'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

# Correct path to src/
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

# Theme
html_theme = 'sphinx_rtd_theme'

# Language
language = 'en'