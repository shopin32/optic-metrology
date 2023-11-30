import cv2
from matplotlib import cm, pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
import numpy as np
from optic_metrology.interference.simulation import YungSimulation
from ipywidgets import IntSlider


def test():
    simulator = YungSimulation(wave_length=650, slit_width=0.005, screen_width=100, screen_width_pxl_qty=600, slit_count=10)
    result = simulator.simulate()
    plt.imshow(result, cmap=cm.gray, norm=LogNorm())
    plt.show()