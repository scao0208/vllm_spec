"""
Build CUTLASS sparse GEMM extension.

Usage:
    cd csrc
    python setup_sparse_gemm.py build_ext --inplace
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# CUTLASS headers from flashinfer's bundled copy
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
    name="sparse_gemm_cutlass",
    ext_modules=[
        CUDAExtension(
            name="sparse_gemm_cutlass",
            sources=["sparse_gemm_cutlass.cu"],
            include_dirs=[CUTLASS_INCLUDE],
            extra_compile_args={
                "nvcc": [
                    "-arch=sm_80",    # SM80 sparse MMA (forward compatible)
                    "-std=c++17",
                    "-O3",
                    "--expt-relaxed-constexpr",
                    "-lineinfo",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
