"""
Build bitmap attention CUDA extension.

Usage:
    cd csrc
    python setup_bitmap_attention.py build_ext --inplace
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# CUTLASS headers (same resolution as setup_sparse_gemm.py)
CUTLASS_INCLUDE = os.path.join(
    os.path.dirname(__file__), "..", "thirdparty", "cutlass", "include"
)

# Fallback: try flashinfer's bundled CUTLASS
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

setup(
    name="bitmap_attention",
    ext_modules=[
        CUDAExtension(
            name="bitmap_attention",
            sources=["bitmap_attention.cu"],
            include_dirs=[CUTLASS_INCLUDE],
            extra_compile_args={
                "nvcc": [
                    "-arch=sm_80",    # SM80+ (Ampere, Hopper, Blackwell)
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
