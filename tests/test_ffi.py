import numpy as np
import pytest

from mojo_fonttools._lib import addr


def test_native_address_contract():
    values = np.zeros(4, dtype=np.float64)
    assert addr(values, np.float64, writable=True) == values.ctypes.data
    with pytest.raises(TypeError, match="dtype"):
        addr(values.astype(np.float32), np.float64)
    with pytest.raises(ValueError, match="C-contiguous"):
        addr(np.zeros((2, 2), dtype=np.float64).T, np.float64)
    with pytest.raises(ValueError, match="empty"):
        addr(np.empty(0, dtype=np.float64), np.float64)
    values.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        addr(values, np.float64, writable=True)
