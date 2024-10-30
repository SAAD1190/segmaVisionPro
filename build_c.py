from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import glob
import torch

def get_extensions():
    # Define the directory where the source files are located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(current_dir, "GroundingSam", "groundingdino", "models", "GroundingDINO", "csrc")

    # Source files
    main_source = os.path.join(extensions_dir, "vision.cpp")
    source_files = glob.glob(os.path.join(extensions_dir, "**", "*.cpp")) + glob.glob(os.path.join(extensions_dir, "**", "*.cu"))

    # Determine if CUDA is available
    use_cuda = torch.cuda.is_available()

    # Define the extension
    ext_modules = [
        CUDAExtension(
            name="groundingdino._C",
            sources=[main_source] + source_files,
            include_dirs=[extensions_dir],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-DCUDA_HAS_FP16=1',
                    '-D__CUDA_NO_HALF_OPERATORS__',
                    '-D__CUDA_NO_HALF_CONVERSIONS__',
                    '-D__CUDA_NO_HALF2_OPERATORS__',
                    '--expt-relaxed-constexpr'
                ] if use_cuda else []
            },
            define_macros=[('WITH_CUDA', None)] if use_cuda else []
        )
    ]

    return ext_modules

setup(
    name="groundingdino",
    version="0.1.0",
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension}
)