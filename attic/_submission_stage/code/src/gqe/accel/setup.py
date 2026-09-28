from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "src.gqe.accel._fast_parity",
        sources=["src/gqe/accel/_fast_parity.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3", "-march=native"],
    ),
]

setup(
    name="fast_parity",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
