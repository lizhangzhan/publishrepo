from setuptools import setup, Extension
from Cython.Distutils import build_ext

import numpy as np

read_md = lambda f: open(f, 'r').read()

setup(
    description='Experiment code for online mlp',

    long_description=read_md('README.md'),

    packages = ['com.dp',
                'com.dp.mlp',
                'com.dp.bbp'],

    install_requires=[
        'cython',
        'numpy',
    ],

    cmdclass={'build_ext': build_ext},

    ext_modules=[Extension('com.dp.mlp.ftrl_mlp',
                           ['com/dp/mlp/ftrl_mlp.pyx'],
                           libraries=[],
                           include_dirs=[np.get_include(), '.'],
                           extra_compile_args=['-O3']),
                 Extension('com.dp.mlp.adadelta_mlp',
                           ['com/dp/mlp/adadelta_mlp.pyx'],
                           libraries=[],
                           include_dirs=[np.get_include(), '.'],
                           extra_compile_args=['-O3']),
                 Extension('com.dp.mlp.ftrl_mlp2',
                           ['com/dp/mlp/ftrl_mlp2.pyx'],
                           libraries=[],
                           include_dirs=[np.get_include(), '.'],
                           extra_compile_args=['-O3']),
                 Extension('com.dp.mlp.adadelta_mlp2',
                           ['com/dp/mlp/adadelta_mlp2.pyx'],
                           libraries=[],
                           include_dirs=[np.get_include(), '.'],
                           extra_compile_args=['-O3']),
                 Extension('com.dp.bbp.adadelta_bmlp2', # new
                           ['com/dp/bbp/adadelta_bmlp2.pyx'],
                           libraries=[],
                           include_dirs=[np.get_include(), '.'],
                           extra_compile_args=['-O3']),
                 Extension('com.dp.bbp.adadelta_bmlp', # new
                           ['com/dp/bbp/adadelta_bmlp.pyx'],
                           libraries=[],
                           include_dirs=[np.get_include(), '.'],
                           extra_compile_args=['-O3']),
                 Extension('com.dp.util',
                           ['com/dp/util.pxd', 'com/dp/util.pyx'],
                           libraries=[],
                           include_dirs=[np.get_include(), '.'],
                           extra_compile_args=['-O3']),
                 ],
)
