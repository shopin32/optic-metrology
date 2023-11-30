import math
from typing import Optional

import numpy as np


class YungSimulation:

    def __init__(
        self,
        slit_width: float = 0.005,  # mm
        slit_height: Optional[float] = None,
        slit_separation: float = 0.125,  # mm
        screen_width: float = 40.0,  # mm
        screen_height: Optional[float] = None,
        wave_length: float = 650,  # nm
        screen_distance: float = 1200,  # mm
        slit_width_pxl_qty: int = 10, 
        slit_height_pxl_qty: Optional[int] = None,
        screen_width_pxl_qty: int = 201,
        screen_height_pxl_qty: Optional[int] = None,
        slit_count: int = 2
    ):
        self.slit_width = slit_width
        self.slit_height = slit_height if slit_height else 20 * slit_width
        self.slit_separation = slit_separation
        self.screen_width = screen_width
        self.screen_height = screen_height if screen_height else screen_width / 7
        self.wave_length = wave_length
        self.screen_distance = screen_distance
        self.slit_width_pxl_qty = slit_width_pxl_qty
        self.slit_height_pxl_qty = slit_height_pxl_qty if slit_height_pxl_qty else 20 * slit_width_pxl_qty
        self.screen_width_pxl_qty = screen_width_pxl_qty
        self.screen_height_pxl_qty = screen_height_pxl_qty if screen_height_pxl_qty else int(screen_width_pxl_qty / 7)
        self.wavenumber = 2 * np.pi / wave_length * 1E9  # nanometers to meters
        self.slit_count = slit_count
        self.sources = []
        counts = [0, 0]
        signs = [1, -1]
        source_x_coords = []
        for i in range(slit_count):
            direction = 0 if i % 2 == 0 else 1
            center_coord = signs[direction] * self.slit_separation / 2 + signs[direction] * counts[direction] * self.slit_separation
            x = np.linspace(center_coord - self.slit_width / 2, center_coord + self.slit_width / 2, self.slit_width_pxl_qty)
            source_x_coords.append(x)
            counts[direction] += 1
        source_x = np.concatenate(source_x_coords)
        source_y = np.linspace(-self.slit_height / 2, self.slit_height / 2, self.slit_height_pxl_qty)
        source_x, source_y = np.meshgrid(source_x, source_y)
        self.source_x = source_x.flatten()
        self.source_y = source_y.flatten()
        screen_x = np.linspace(-self.screen_width / 2, self.screen_width / 2, self.screen_width_pxl_qty)
        screen_y = np.linspace(-self.screen_height / 2, self.screen_height / 2, self.screen_height_pxl_qty)
        self.screen_x, self.screen_y = np.meshgrid(screen_x, screen_y)
    
    def distance_to_source(self, x: float, y: float) -> np.ndarray:
        return np.sqrt((self.source_x - x)**2 + (self.source_y - y)**2 + self.screen_distance ** 2) * 1e-3


    def get_wave_phase(self, x: float, y: float) -> np.ndarray:
        return self.wavenumber * self.distance_to_source(x, y)
    
    def intensity(self, x: float, y: float) -> np.ndarray:
        Ex = np.sum(np.cos(self.get_wave_phase(x, y)))
        Ey = np.sum(np.sin(self.get_wave_phase(x, y)))
        return Ex ** 2 + Ey ** 2
    
    def simulate(self):
        intensities = np.zeros((self.screen_height_pxl_qty, self.screen_width_pxl_qty))
        for i in range(self.screen_height_pxl_qty):
            for j in range(self.screen_width_pxl_qty):
                intensities[i, j] = self.intensity(self.screen_x[i, j], self.screen_y[i, j])
        return intensities
    



