import os
import sys

# ====================== Project Information ======================
project = 'Intelligent Test Case Clustering'
copyright = '2026, Чухліб Євгеній ІН-23'
author = 'Чухліб Євгеній ІН-23'
release = '0.1.0'

# ====================== General Configuration ======================
extensions = [
    'sphinx.ext.autodoc',      # Automatically extracts docstrings
    'sphinx.ext.napoleon',     # Supports Google-style docstrings
    'sphinx.ext.viewcode',     # Links to the source code
]

# Correct path to src/
sys.path.insert(0, os.path.abspath('../../src'))

# Napoleon settings (Google style)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

# ====================== Multilingual Support ======================
# Primary language
language = 'en'

locale_dirs = ['../locale/']           # Folder where translations will be stored
gettext_compact = False

# List of all supported documentation languages
html_context = {
    'languages': [
        ('en', 'English'),
        ('uk', 'Українська'),
        ('de', 'Deutsch'),
    ],
}

# ====================== HTML Output ======================
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Order of class members
autodoc_member_order = 'bysource'