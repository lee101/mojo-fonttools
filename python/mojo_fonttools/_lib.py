from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB_PATH = os.environ.get(
    "MOJO_FONTTOOLS_LIB", os.path.join(ROOT, "dist", "libmojo-fonttools.so")
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mft_quadratic_point": ([F] * 7 + [I], None),
    "mft_cubic_point": ([F] * 9 + [I], None),
    "mft_quadratic_bounds": ([F] * 6 + [I], None),
    "mft_cubic_bounds": ([F] * 8 + [I], None),
    "mft_quadratic_bounds_batch": ([I] * 3, None),
    "mft_cubic_bounds_batch": ([I] * 3, None),
    "mft_quadratic_arc_length": ([F] * 6, F),
    "mft_cubic_arc_length": ([F] * 8, F),
    "mft_decode_simple_glyph": ([I] * 7, I),
    "mft_iup_contour": ([I] * 6, None),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    src = os.path.join(ROOT, "src", "fonttools.mojo")
    if not force and os.path.exists(LIB_PATH) and os.path.getmtime(LIB_PATH) >= os.path.getmtime(src):
        return LIB_PATH
    mojo = shutil.which("mojo")
    if mojo is None:
        raise BuildError("mojo executable not found; run `pixi run build`")
    os.makedirs(os.path.dirname(LIB_PATH), exist_ok=True)
    proc = subprocess.run(
        [mojo, "build", "--emit", "shared-lib", src, "-o", LIB_PATH],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB_PATH):
        raise BuildError((proc.stderr or proc.stdout).strip())
    return LIB_PATH


_LIB = None
_PY_LIB = None


def lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(build())
        for name, (args, result) in _SIGNATURES.items():
            fn = getattr(_LIB, name)
            fn.argtypes = args
            fn.restype = result
    return _LIB


def py_lib() -> ctypes.PyDLL:
    global _PY_LIB
    if _PY_LIB is None:
        _PY_LIB = ctypes.PyDLL(build())
        _PY_LIB.mft_iup_result.argtypes = [I, I]
        _PY_LIB.mft_iup_result.restype = ctypes.py_object
    return _PY_LIB


def addr(
    value: np.ndarray,
    dtype: np.dtype | type,
    *,
    writable: bool = False,
) -> int:
    """Return an address only for an array that satisfies the native ABI."""
    if not isinstance(value, np.ndarray):
        raise TypeError("native buffers must be NumPy arrays")
    expected = np.dtype(dtype)
    if value.dtype != expected:
        raise TypeError(f"native buffer must have dtype {expected}, got {value.dtype}")
    if not value.flags.c_contiguous:
        raise ValueError("native buffers must be C-contiguous")
    if not value.flags.aligned:
        raise ValueError("native buffers must be aligned")
    if writable and not value.flags.writeable:
        raise ValueError("native output buffers must be writable")
    if value.size == 0:
        raise ValueError("native buffers must not be empty")
    address = int(value.ctypes.data)
    if address == 0:
        raise ValueError("native buffers must have a non-null address")
    return address
