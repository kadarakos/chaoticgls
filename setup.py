from distutils.core import setup, Extension
from Cython.Build import cythonize


ext = Extension(name="glsneuron", sources=["glsneuron.pyx"])
setup(ext_modules=cythonize(ext))



