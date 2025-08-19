from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os, sys, torch

ROOT = os.path.dirname(__file__)

# MuJoCo paths
MUJOCO_ROOT    = os.path.join(ROOT, "mujoco")
MUJOCO_INCLUDE = os.path.join(MUJOCO_ROOT, "_install", "include")
MUJOCO_LIB     = os.path.join(MUJOCO_ROOT, "_install", "lib")
MUJOCO_SRC     = os.path.join(MUJOCO_ROOT, "src")  # internal headers

# Your engine extension (C code + headers)
ENGINE_EXT_INC = os.path.join(ROOT, "src")  

# Torch shared libs (for libc10, libtorch, etc)
TORCH_LIB_DIR  = os.path.join(os.path.dirname(torch.__file__), "lib")

# sanity checks
hdr = os.path.join(MUJOCO_INCLUDE, "mujoco", "mujoco.h")
if not os.path.exists(hdr):
    sys.exit(f"[error] MuJoCo header not found at {hdr}\n"
             f"Did you run 'cmake --install build' inside the mujoco submodule?")

lib = os.path.join(MUJOCO_LIB, "libmujoco.so")
if not os.path.exists(lib):
    sys.exit(f"[error] MuJoCo library not found at {lib}\n"
             f"Did you run 'cmake --install build' inside the mujoco submodule?")

print(f"[torchmj] Using MUJOCO_INCLUDE={MUJOCO_INCLUDE}")
print(f"[torchmj] Using MUJOCO_LIB={MUJOCO_LIB}")
print(f"[torchmj] Using TORCH_LIB_DIR={TORCH_LIB_DIR}")

# --- If your C code is strict-C, use a wrapper TU that #includes the .c files ---
# Create src/engine_extension_build.cpp with:
#   extern "C" {
#     #include "engine_extension/foo.c"
#     #include "engine_extension/bar.c"
#   }
# and DO NOT also list foo.c/bar.c in sources to avoid duplicate symbols.
SOURCES = [
    "src/mjstep_module.cpp",
    "src/utils.cpp",
    "src/engine_extension_build.cpp",  # <- wrapper that pulls in your C implementations
]

setup(
    name='torchmj',
    ext_modules=[
        CppExtension(
            name='torchmj',
            sources=SOURCES,
            include_dirs=[
                MUJOCO_INCLUDE,   # public headers <mujoco/mujoco.h>
                MUJOCO_SRC,       # internal headers (e.g. src/engine/**)
                ENGINE_EXT_INC,   # your C headers
            ],
            libraries=['mujoco'],
            library_dirs=[MUJOCO_LIB, TORCH_LIB_DIR],
            # RPATHs so Python can dlopen without manual LD_LIBRARY_PATH
            extra_link_args=[
                f"-Wl,-rpath,{MUJOCO_LIB}",
                f"-Wl,-rpath,{TORCH_LIB_DIR}",
                "-Wl,-rpath,$ORIGIN",   # helpful if you later co-locate libs
                "-g", "-fopenmp",
            ],
            extra_compile_args=[
                "-O3", "-g", "-std=c++17",
                "-fno-omit-frame-pointer",
                "-D_GLIBCXX_ASSERTIONS",
                "-fopenmp",
            ],
            language='c++',
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
