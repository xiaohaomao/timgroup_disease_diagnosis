

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

from distutils.core import setup
from Cython.Build import cythonize

# python3 setup.py build_ext --inplace
setup(
	name='cycommon_lib',
	ext_modules=cythonize("cycommon.pyx", annotate=True),
	include_dirs=[np.get_include()]
)



