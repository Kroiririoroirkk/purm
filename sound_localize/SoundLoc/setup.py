from distutils.core import Extension, setup
import numpy

module = Extension("soundloc", sources=["soundloc.c"], include_dirs=[numpy.get_include()])
setup(name="soundloc", version="1.0", ext_modules=[module])
