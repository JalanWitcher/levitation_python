import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Automatically apply the best hardware optimizations based on the OS
compiler_args = ["/O2", "/fp:fast"] if sys.platform == "win32" else ["-O3", "-ffast-math", "-march=native"]

ext_modules = [
    Pybind11Extension(
        "ode_cpp",  # Name of the module to be created         
        ["levitation_python/odeSolvers/ode_cpp/dop853_bind.cpp"], 
        cxx_std=17,
        extra_compile_args=compiler_args,
    ),
]

setup(
    packages=["levitation_python"], # Tells pip to grab your whole Python folder
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)