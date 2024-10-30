from setuptools import setup, Extension
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

# Define the correct directory path to the "groundingdino" sources
current_dir = os.path.dirname(os.path.abspath(__file__))
extensions_dir = os.path.join(current_dir, "GroundingSam", "groundingdino", "models", "GroundingDINO", "csrc")

# Source files for building
main_source = os.path.join(extensions_dir, "vision.cpp")
source_files = [
    main_source,
    os.path.join(extensions_dir, "MsDeformAttn", "ms_deform_attn_cpu.cpp"),
    os.path.join(extensions_dir, "MsDeformAttn", "ms_deform_attn_cuda.cu")
]

# Check if CUDA is available
use_cuda = os.getenv("CUDA_HOME") is not None

# Configure the extension module
extensions = [
    CUDAExtension(
        name="groundingdino._C",
        sources=source_files,
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

# Setup for building
setup(
    name="groundingdino",
    version="0.1.0",
    ext_modules=extensions,
    cmdclass={"build_ext": BuildExtension}
)