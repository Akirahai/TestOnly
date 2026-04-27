from typing import Callable

class DependencyConfig:
    def __init__(
        self,
        # Parameter Preprocessing

        # Spherical Harmonics

        # 3D -> 2D projection
        build_rotation: Callable

        # Culling

        # Sorting

        # Image Rendering
    ):
        self.build_rotation = build_rotation