import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _as_uint8(arr):
    """
    Convierte un arreglo al rango [0, 255] y lo pasa a uint8.
    """
    return np.clip(arr, 0, 255).astype(np.uint8)


def gaussian_filter_numpy(image):
    """
    Aplica filtro gaussiano 3x3 usando NumPy.
    """
    image = image.astype(np.float32)
    padded = np.pad(image, 1, mode="constant")
    windows = sliding_window_view(padded, (3, 3))

    kernel = np.array(
        [
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1],
        ],
        dtype=np.float32,
    ) / 16.0

    out = (windows * kernel).sum(axis=(-1, -2))
    return _as_uint8(np.rint(out))


def sobel_filter_numpy(image):
    """
    Aplica filtro Sobel usando NumPy.
    """
    image = image.astype(np.float32)
    padded = np.pad(image, 1, mode="constant")
    windows = sliding_window_view(padded, (3, 3))

    kx = np.array(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ],
        dtype=np.float32,
    )

    ky = np.array(
        [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ],
        dtype=np.float32,
    )

    gx = (windows * kx).sum(axis=(-1, -2))
    gy = (windows * ky).sum(axis=(-1, -2))

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return _as_uint8(magnitude)


def median_filter_numpy(image):
    """
    Aplica filtro de mediana 3x3 usando NumPy.
    """
    padded = np.pad(image, 1, mode="constant")
    windows = sliding_window_view(padded, (3, 3))

    out = np.median(windows, axis=(-1, -2))
    return _as_uint8(out)