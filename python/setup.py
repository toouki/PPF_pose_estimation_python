#!/usr/bin/env python3
"""
Setup script for PPF Python bindings
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from pybind11.setup_helpers import Pybind11Extension, build_ext as pybind11_build_ext
import pybind11


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(pybind11_build_ext):
    def build_extension(self, ext):
        if isinstance(ext, CMakeExtension):
            # Get paths
            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
            cmake_args = [
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
                f"-DPYTHON_EXECUTABLE={sys.executable}",
                f"-DCMAKE_BUILD_TYPE={'Debug' if self.debug else 'Release'}",
            ]

            cfg = 'Debug' if self.debug else 'Release'
            build_args = ['--config', cfg]

            # Add parallel build
            if hasattr(self, 'parallel'):
                build_args.extend(['--parallel', str(self.parallel)])

            # Create build directory
            build_temp = Path(self.build_temp) / ext.name
            build_temp.mkdir(parents=True, exist_ok=True)

            # Run CMake
            subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=build_temp)
            subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_temp)
        else:
            super().build_extension(ext)


# Get the long description from README
def get_long_description():
    readme_path = Path(__file__).parent.parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding='utf-8')
    return "PPF Surface Match - Fast and robust point pair feature matching library"


setup(
    name="ppf-surface-match",
    version="1.0.0",
    author="PPF Surface Match Team",
    description="Fast and robust point pair feature matching library",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/toouki/ppf-surface-match-python",
    
    packages=find_packages(),
    py_modules=["ppf_wrapper"],
    
    ext_modules=[
        CMakeExtension("ppf", sourcedir="."),
    ],
    
    cmdclass={"build_ext": CMakeBuild},
    
    install_requires=[
        "numpy>=1.19.0",
        "pybind11>=2.6.0",
    ],
    
    python_requires=">=3.6",
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    keywords="3d computer-vision point-cloud surface-matching ppf",
    
    project_urls={
        "Bug Reports": "https://github.com/toouki/ppf-surface-match-python/issues",
        "Source": "https://github.com/toouki/ppf-surface-match-python",
    },
)