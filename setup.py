from setuptools import setup
from setuptools import find_packages
import os


def get_version():
    g = {}
    exec(open(os.path.join("twaml", "version.py")).read(), g)
    return g["__version__"]


with open("requirements.txt") as f:
    requirements = f.read().splitlines()

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), "rb") as f:
    long_description = f.read().decode("utf-8")

setup(
    name="twaml",
    version=get_version(),
    scripts=[],
    packages=find_packages(exclude=["tests"]),
    entry_points={
        "console_scripts": ["twaml-root2pytables = twaml._apps:root2pytables"]
    },
    description="tW Analysis Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Doug Davis",
    author_email="ddavis@ddavis.io",
    license="MIT",
    url="https://github.com/drdavis/twaml",
    test_suite="tests",
    python_requires=">3.6.5",
    install_requires=requirements,
    tests_require=["pytest>=4.0"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
