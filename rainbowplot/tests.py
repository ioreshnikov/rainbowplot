from unittest import TestCase

import numpy as np
import matplotlib.pyplot as plt

from .plot import rainbowplot


class Fixtures:
    """
    This class is a namespace for all the fixtures used in tests.
    """

    nt = 2**10
    nx = 2**10

    t = np.linspace(0, 10, nt)
    x = np.linspace(-10, +10, nx)

    @staticmethod
    def monochromatic_gaussian():
        """
        A simple monochromatic Gaussian pulse with fixed carrier frequency ω=10.
        """
        x = Fixtures.x
        u = np.exp(-x**2) * np.exp(- 1j * 10 * x)
        return np.tile(u, (Fixtures.nt, 1))

    @staticmethod
    def two_monochromatic_gaussians():
        """
        A sum of two monochromatic pulses.
        """
        x = Fixtures.x
        u = (
            np.exp(-(x - 5)**2) * np.exp(-1j * +10 * x) +
            np.exp(-(x + 5)**2) * np.exp(-1j * -10 * x))
        return np.tile(u, (Fixtures.nt, 1))

    @staticmethod
    def chirped_pulse():
        """
        A chirped Gaussian pulse.
        """
        x = Fixtures.x
        u = np.exp(-x**2/5**2) * np.exp(-1j * x**2)
        return np.tile(u, (Fixtures.nt, 1))


class SmokeTestCase(TestCase):
    def test_smoke(self):
        t = Fixtures.t
        x = Fixtures.x
        u = Fixtures.chirped_pulse()

        rainbowplot(x, t, u, win=0.5, ssx=4, ssy=4)
        plt.show()
