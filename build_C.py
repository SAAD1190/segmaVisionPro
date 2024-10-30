# build_c.py
import os
import glob
import torch
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
from setuptools import setup

# Function to determine if CUDA is available and compile the extension
def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "groundingdino", "models", "GroundingDINO", "csrc")

    main_source = os.path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu"))

    sources = [main_source] + sources

    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    if torch.cuda.is_available():
        print("Compiling with CUDA")
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    else:
        print("Compiling without CUDA")
        extra_compile_args["nvcc"] = []

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "groundingdino._C",
            sources=sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args
        )
    ]

    return ext_modules

# Run the extension build
if __name__ == "__main__":
    setup(
        name="groundingdino_ext",
        version="1.0",
        ext_modules=get_extensions(),
        cmdclass={"build_ext": BuildExtension}
    )