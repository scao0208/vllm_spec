"""
Build paged bitmap attention CUDA extension.

Usage:
    cd csrc
    python setup_paged_bitmap_attention.py build_ext --inplace
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# CUTLASS headers
CUTLASS_INCLUDE = os.path.join(
    os.path.dirname(__file__), "..", "thirdparty", "cutlass", "include"
)

# Fallback: flashinfer's bundled CUTLASS
if not os.path.exists(CUTLASS_INCLUDE):
    import site
    for sp in site.getsitepackages():
        candidate = os.path.join(sp, "flashinfer", "data", "cutlass", "include")
        if os.path.exists(candidate):
            CUTLASS_INCLUDE = candidate
            break

print(f"CUTLASS include: {CUTLASS_INCLUDE}")
assert os.path.exists(os.path.join(CUTLASS_INCLUDE, "cutlass", "cutlass.h")), \
    f"CUTLASS headers not found at {CUTLASS_INCLUDE}"

SRC_DIR = os.path.dirname(__file__)

setup(
    name="paged_bitmap_attention_ext",
    ext_modules=[
        CUDAExtension(
            name="paged_bitmap_attention_ext",
            sources=["paged_bitmap_attention/launch.cu"],
            include_dirs=[CUTLASS_INCLUDE, SRC_DIR],
            extra_compile_args={
                "nvcc": [
                    "-arch=sm_80",
                    "-std=c++17",
                    "-O3",
                    "--expt-relaxed-constexpr",
                    "-lineinfo",
                    "--use_fast_math",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
