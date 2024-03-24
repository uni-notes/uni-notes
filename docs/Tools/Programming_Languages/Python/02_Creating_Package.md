# Creating Packages

## Folder Structure

```
base_folder
- package_name
	- init.py
	- package_name.py
- setup.py
- requirements.txt
```

## `package_name.py`

```python
class Package_Name:
  pass
```

## `init.py`

```python
from package_name import Package_Name
```

## `setup.py`

```python
"""
Author: Ahmed Thahir

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <http://unlicense.org/>
"""

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

VERSION = "0.0.1"
DESCRIPTION = "St. Louis Federal Reserve FRED API"

import os
here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "requirements.txt"), "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

with open(os.path.join(here, "README.md"), "r", encoding="utf-8") as f:
    long_description = "\n" + f.read()

setup(
    name = "fredx",
    version = VERSION,
    description = DESCRIPTION,

    long_description = long_description,
    long_description_content_type = "text/markdown",

    keywords = ["fred", "api", "federal reserve", "st. louis fed", "async"],
    author = "Ahmed Thahir",
    author_email = "ahmedthahir2002@gmail.com",
    url = "https://github.com/AhmedThahir/fredx",
    license = "MIT",
    packages = find_packages(),
    install_requires = requirements,
    classifiers = [
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Internet",
        "Topic :: Internet :: WWW/HTTP",
        "Natural Language :: English",
    ],
)
```

## Building

```bash
pip install setuptools
python setup.py sdist bdist_wheel
```

## Uploading

```python
pip install twine
twine upload dist/*
```

