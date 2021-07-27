import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
from numpy.fft import fftfreq, fftshift, ifft


HUE_RED = 0.00
HUE_VIOLET = 0.75


def _sigma(x):
    """
    Smooth-step function.
    """
    return 1 / (1 + np.exp(-x))


def _estimate_frequency_range(f, z):
    """
    Estimate minimum and maximum frequency of a given field.
    """
    u = fftshift(ifft(z, axis=1), axes=(1, ))
    u = abs(u)**2
    u = u.sum(axis=0)

    threshold = 0.10 * u.max()

    up = u[:-1]  # u₀, u₁, u₂, ..., uₙ₋₁
    un = u[1:]   # u₁, u₂, u₃, ..., uₙ

    # We assume that the spectrum is bounded and we're looking for the
    # left-most position where the spectrum becomes greater than the threshold
    # and the right-most position where it again becomes smaller. Those two we
    # use as the estimates for the spectral width.

    f_ = f[1:]  # we throw-away zeroth frequency to match the dimension.
    fmin = f_[(up <= threshold) & (un > threshold)].min()
    fmax = f_[(up >= threshold) & (un < threshold)].max()

    return fmin, fmax


def rainbowplot(
    x, y, z,
    fmin=None, fmax=None,
    norm=colors.Normalize(),
    win=None, ssx=1, ssy=1
):
    # Allocate the output image.
    nx_in = len(x)
    ny_in = len(y)
    nx_out = int(nx_in / ssx)
    ny_out = int(ny_in / ssy)
    im = np.zeros((nx_out, ny_out, 3), dtype=float)

    # Window size chosen as 1/32 of the original x range by default.
    if not win:
        win = (max(x) - min(x)) / 32

    # Compute the frequency scale of the signal.
    dx = x[1] - x[0]
    f = fftfreq(nx_in, dx)
    f = 2 * np.pi * fftshift(f)

    # If either minimum or maximum frequency was not passed we estimate it from
    # the spectral density.
    if fmin is None or fmax is None:
        fmin_est, fmax_est = _estimate_frequency_range(f, z)
        fmin = fmin_est if fmin is None else fmin
        fmax = fmax_est if fmax is None else fmax

    # Prepare the hue scale. We color the minimal frequency as red (hue = 0.0)
    # and the maximal frequency as violet (around hue = 0.75).
    hue_scale = HUE_VIOLET * (f - fmin) / (fmax - fmin) + HUE_RED
    hue_scale = np.clip(hue_scale, HUE_RED, HUE_VIOLET)

    # Prepare the saturation scale. We are going to use 0 everywhere outside
    # of the defined frequency range and 1 within with a smooth transition
    # between the regions. This will give us vivid colors within the frequency
    # range and pure white outside. # From playing with the code we know it's
    # a good idea to pad this region a bit (around 20% of the frequency
    # range), which allows fmin and fmax to be painted with a pronounced red
    # and violet and not to dilute them due to transition to white.
    pad_f = (fmax - fmin) / 5
    sat_scale = _sigma(f - fmin + pad_f) - _sigma(f - fmax - pad_f)

    # Convert this to RGB.
    hsv_scale = np.ones((nx_in, 3))
    hsv_scale[:, 0] = hue_scale
    hsv_scale[:, 1] = sat_scale

    rgb_scale = colors.hsv_to_rgb(hsv_scale)

    # For every point in the sub-sampled image we compute a windowed Fourier
    # transform and get a local spectrum of the signal. By assigning a hue to
    # a frequency and using spectral power as the color weight and then
    # averaging we compute an RGB color of an individual pixel.
    for xi, x0 in enumerate(x[::ssx]):
        # Put the signal in a window.
        window = np.exp(-(x - x0)**2 / win**2)
        zw = window * z[::ssy]

        # Compute the spectral power to be used as hue weight.
        pw = fftshift(ifft(zw, axis=1))
        pw = abs(pw)**2

        rgb = pw @ rgb_scale / np.sum(pw, axis=1)[:, np.newaxis]
        im[:, xi, :] = rgb

    # At this point the image has the correct colors, but not the correct
    # intensity. Let's fix this by one more time transforming to HSV and using
    # the normalized absolute value of the field as the V(alue) in HSV space.
    n = norm(abs(z[::ssy, ::ssx]))

    rgb = im.reshape(-1, 3)                # unravel into a n x 3 matrix
    hsv = colors.rgb_to_hsv(rgb)           # (required by this step)
    hsv[:, 2] = np.ravel(n)                # put the norm as value
    rgb = colors.hsv_to_rgb(hsv)           # back to rgb
    im = rgb.reshape((nx_out, ny_out, 3))  # back to the original shape

    return plt.imshow(im, extent=[x.min(), x.max(), y.min(), y.max()])
