import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


def _sigma(x):
    return 1 / (1 + np.exp(-x))


def rainbowplot(
    x, y, z, fmin, fmax,
    norm=colors.Normalize(),
    win=None, ssx=1, ssy=1
):
    # Allocate the image.
    nx = len(x)
    ny = len(y)
    im = np.zeros((int(ny / ssy), int(nx / ssx), 3), dtype=float)

    # Pre-compute the norm of the output.
    n = norm(abs(z[::ssy, ::ssx]))

    # Window size chosen as 1/32 of the original x range by default.
    if not win:
        win = (max(x) - min(x)) / 32

    # Compute the frequency scale of the signal.
    dx = x[1] - x[0]
    f = np.fft.fftfreq(nx, dx)
    f = 2 * np.pi * np.fft.fftshift(f)

    # Prepare the hue scale and the saturation scales, convert that to the color
    # scale.
    hue_scale = (f - fmin) / (fmax - fmin)
    hue_scale = np.clip(hue_scale, 0.0, 1.0)

    slope = (fmax - fmin) / 32
    sat_scale = _sigma((f - fmin) / slope) - _sigma((f - fmax) / slope)

    hsv_scale = np.ones((nx, 3))
    hsv_scale[:, 0] = hue_scale
    hsv_scale[:, 1] = sat_scale

    rgb_scale = colors.hsv_to_rgb(hsv_scale)

    # For every point in the sub-sampled image we will compute a windowed
    # Fourier transform and get a local spectrum of the signal. By assigning a
    # hue to a frequency and using spectral power as the color weight we compute
    # an RGB color of an individual pixel.
    for xi, x0 in enumerate(x[::ssx]):
        # Put the signal in a window.
        window = np.exp(-(x - x0)**2 / win**2)
        zw = window * z[::ssy]

        # Compute the spectral power to be used as hue weight.
        pw = np.fft.fftshift(np.fft.ifft(zw, axis=1))
        pw = abs(pw)**2

        rgb = pw @ rgb_scale / np.sum(pw, axis=1)[:, np.newaxis]
        im[:, xi, :] = rgb

    # Convert back to HSV to adjust the value of the image.
    nx, ny, _ = im.shape
    rgb = im.reshape(-1, 3)
    hsv = colors.rgb_to_hsv(rgb)
    hsv[:, 2] = np.ravel(n)
    rgb = colors.hsv_to_rgb(hsv)
    im = rgb.reshape((nx, ny, 3))

    plt.imshow(im)
    plt.show()
