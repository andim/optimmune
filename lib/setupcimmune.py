from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy.distutils.misc_util

include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs()

setup(
    name='cimmune',
    version='1.0',
    ext_modules=[Extension('cimmune', ['cimmune.pyx'],
                           include_dirs=include_dirs)],
    cmdclass={'build_ext': build_ext}
    )
