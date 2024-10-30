import glob
import os
import subprocess
import sys
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension


# Function to ensure PyTorch is installed
def install_torch():
    try:
        import torch
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])


# Call the function to ensure torch is installed
install_torch()

import torch


# groundingdino version info
version = "1.0.0"
package_name = "groundingdino"
cwd = os.path.dirname(os.path.abspath(__file__))

# Attempt to get the git SHA for the version
sha = "Unknown"
try:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode("ascii").strip()
except Exception:
    pass

# Write version file
def write_version_file():
    version_path = os.path.join(cwd, "groundingdino", "version.py")
    with open(version_path, "w") as f:
        f.write(f"__version__ = '{version}'\n")
        f.write(f"git_version = {repr(sha)}\n")


# Parse requirements file
def parse_requirements(fname="requirements.txt", with_version=True):
    """Parse the package dependencies listed in a requirements file."""
    import re
    import sys
    from os.path import exists

    require_fpath = fname

    def parse_line(line):
        if line.startswith("-r "):
            target = line.split(" ")[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {"line": line}
            if line.startswith("-e "):
                info["package"] = line.split("#egg=")[1]
            elif "@git+" in line:
                info["package"] = line
            else:
                pat = "(" + "|".join([">=", "==", ">"]) + ")"
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]
                info["package"] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ";" in rest:
                        version, platform_deps = map(str.strip, rest.split(";"))
                        info["platform_deps"] = platform_deps
                    else:
                        version = rest
                    info["version"] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info["package"]]
                if with_version and "version" in info:
                    parts.extend(info["version"])
                if not sys.version.startswith("3.4"):
                    platform_deps = info.get("platform_deps")
                    if platform_deps is not None:
                        parts.append(";" + platform_deps)
                item = "".join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


# Function to get C++/CUDA extensions
def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "groundingdino", "models", "GroundingDINO", "csrc")

    main_source = os.path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu")) + glob.glob(
        os.path.join(extensions_dir, "*.cu")
    )

    sources = [main_source] + sources

    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    # Compile with CUDA if available
    if CUDA_HOME is not None and (torch.cuda.is_available() or "TORCH_CUDA_ARCH_LIST" in os.environ):
        print("Compiling with CUDA support")
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
        print("Compiling without CUDA support")
        define_macros += [("WITH_HIP", None)]
        extra_compile_args["nvcc"] = []

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "groundingdino._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


if __name__ == "__main__":
    print(f"Building wheel {package_name}-{version}")

    with open("LICENSE", "r", encoding="utf-8") as f:
        license = f.read()

    write_version_file()

    setup(
        name="groundingdino",
        version="1.0.0",
        author="International Digital Economy Academy, Shilong Liu",
        url="https://github.com/IDEA-Research/GroundingDINO",
        description="Open-set object detector",
        license=license,
        install_requires=parse_requirements("requirements.txt"),
        packages=find_packages(
            exclude=(
                "configs",
                "tests",
            )
        ),
        ext_modules=get_extensions(),
        cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    )
