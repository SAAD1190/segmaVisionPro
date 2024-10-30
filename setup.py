from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
import os
import glob

# Function to read the requirements file and install packages
def parse_requirements(filename):
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

# Function to determine if CUDA is available
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

setup(
    name='segmaVisionPro',
    version='1.0',
    description='SegmaVisionPro is a project focused on advanced object detection, segmentation, and prompt generation',
    packages=find_packages(),  # This will automatically find all Python packages in your project
    install_requires=parse_requirements('requirements.txt'),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
