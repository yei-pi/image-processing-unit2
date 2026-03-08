# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
cimport numpy as np
from libc.math cimport sqrt

ctypedef np.uint8_t UINT8_t
ctypedef np.int32_t INT32_t


def gaussian_filter_cython(np.ndarray[UINT8_t, ndim=2] image):
    """
    Filtro gaussiano 3x3 implementado en Cython.
    """
    cdef int h = image.shape[0]
    cdef int w = image.shape[1]

    cdef np.ndarray[INT32_t, ndim=2] padded = np.zeros((h + 2, w + 2), dtype=np.int32)
    cdef np.ndarray[UINT8_t, ndim=2] out = np.zeros((h, w), dtype=np.uint8)

    cdef int i, j, total

    padded[1:h + 1, 1:w + 1] = image

    for i in range(h):
        for j in range(w):
            total = (
                padded[i, j] + 2 * padded[i, j + 1] + padded[i, j + 2] +
                2 * padded[i + 1, j] + 4 * padded[i + 1, j + 1] + 2 * padded[i + 1, j + 2] +
                padded[i + 2, j] + 2 * padded[i + 2, j + 1] + padded[i + 2, j + 2]
            )
            out[i, j] = <UINT8_t>((total + 8) // 16)

    return out


def sobel_filter_cython(np.ndarray[UINT8_t, ndim=2] image):
    """
    Filtro Sobel implementado en Cython.
    """
    cdef int h = image.shape[0]
    cdef int w = image.shape[1]

    cdef np.ndarray[INT32_t, ndim=2] padded = np.zeros((h + 2, w + 2), dtype=np.int32)
    cdef np.ndarray[UINT8_t, ndim=2] out = np.zeros((h, w), dtype=np.uint8)

    cdef int i, j, gx, gy
    cdef double mag

    padded[1:h + 1, 1:w + 1] = image

    for i in range(h):
        for j in range(w):
            gx = (
                -padded[i, j] + padded[i, j + 2]
                - 2 * padded[i + 1, j] + 2 * padded[i + 1, j + 2]
                - padded[i + 2, j] + padded[i + 2, j + 2]
            )

            gy = (
                -padded[i, j] - 2 * padded[i, j + 1] - padded[i, j + 2]
                + padded[i + 2, j] + 2 * padded[i + 2, j + 1] + padded[i + 2, j + 2]
            )

            mag = sqrt(gx * gx + gy * gy)

            if mag > 255:
                mag = 255

            out[i, j] = <UINT8_t>mag

    return out


def median_filter_cython(np.ndarray[UINT8_t, ndim=2] image):
    """
    Filtro de mediana 3x3 implementado en Cython.
    """
    cdef int h = image.shape[0]
    cdef int w = image.shape[1]

    cdef np.ndarray[INT32_t, ndim=2] padded = np.zeros((h + 2, w + 2), dtype=np.int32)
    cdef np.ndarray[UINT8_t, ndim=2] out = np.zeros((h, w), dtype=np.uint8)

    cdef int i, j
    cdef int vals[9]
    cdef int a, b, tmp

    padded[1:h + 1, 1:w + 1] = image

    for i in range(h):
        for j in range(w):
            vals[0] = padded[i, j]
            vals[1] = padded[i, j + 1]
            vals[2] = padded[i, j + 2]
            vals[3] = padded[i + 1, j]
            vals[4] = padded[i + 1, j + 1]
            vals[5] = padded[i + 1, j + 2]
            vals[6] = padded[i + 2, j]
            vals[7] = padded[i + 2, j + 1]
            vals[8] = padded[i + 2, j + 2]

            for a in range(8):
                for b in range(a + 1, 9):
                    if vals[b] < vals[a]:
                        tmp = vals[a]
                        vals[a] = vals[b]
                        vals[b] = tmp

            out[i, j] = <UINT8_t>vals[4]

    return out