import math


def _pad_image_zero(image, pad=1):
    """
    Agrega padding de ceros alrededor de la imagen.

    Parámetros:
        image (list[list[int]]): imagen como lista de listas
        pad (int): grosor del padding

    Retorna:
        list[list[int]]: imagen con padding
    """
    h = len(image)
    w = len(image[0])

    padded = [[0 for _ in range(w + 2 * pad)] for _ in range(h + 2 * pad)]

    for i in range(h):
        for j in range(w):
            padded[i + pad][j + pad] = int(image[i][j])

    return padded


def gaussian_filter_python(image):
    """
    Aplica filtro gaussiano 3x3 usando Python puro.

    Kernel:
        1 2 1
        2 4 2
        1 2 1

    Todo dividido entre 16.
    """
    kernel = [
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1],
    ]
    denom = 16

    h = len(image)
    w = len(image[0])

    padded = _pad_image_zero(image, 1)
    out = [[0 for _ in range(w)] for _ in range(h)]

    for i in range(h):
        for j in range(w):
            total = 0

            for ki in range(3):
                for kj in range(3):
                    total += padded[i + ki][j + kj] * kernel[ki][kj]

            value = int(round(total / denom))
            value = max(0, min(255, value))
            out[i][j] = value

    return out


def sobel_filter_python(image):
    """
    Aplica filtro Sobel usando Python puro.

    Calcula gradiente en X y Y, luego combina:
        magnitud = sqrt(gx^2 + gy^2)
    """
    gx_kernel = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ]

    gy_kernel = [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ]

    h = len(image)
    w = len(image[0])

    padded = _pad_image_zero(image, 1)
    out = [[0 for _ in range(w)] for _ in range(h)]

    for i in range(h):
        for j in range(w):
            gx = 0
            gy = 0

            for ki in range(3):
                for kj in range(3):
                    pixel = padded[i + ki][j + kj]
                    gx += pixel * gx_kernel[ki][kj]
                    gy += pixel * gy_kernel[ki][kj]

            mag = math.sqrt(gx * gx + gy * gy)
            mag = int(max(0, min(255, mag)))
            out[i][j] = mag

    return out


def median_filter_python(image):
    """
    Aplica filtro de mediana 3x3 usando Python puro.
    """
    h = len(image)
    w = len(image[0])

    padded = _pad_image_zero(image, 1)
    out = [[0 for _ in range(w)] for _ in range(h)]

    for i in range(h):
        for j in range(w):
            window = []

            for ki in range(3):
                for kj in range(3):
                    window.append(padded[i + ki][j + kj])

            window.sort()
            out[i][j] = int(window[4])

    return out