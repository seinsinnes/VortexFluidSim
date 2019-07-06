from distutils.core import setup, Extension
import numpy.distutils.misc_util
import numpy
print(numpy.get_include())
setup(
    ext_modules=[Extension("fluid", ["fluid.c", "fluid_wrapper.c"],include_dirs=[numpy.get_include()])],
    include_dirs=numpy.get_include())
