import numpy as np

from .constants import LANDMARK_COUNT
from .schemas import Landmark


def normalize_landmarks(landmarks: list[Landmark]) -> np.ndarray:
    """Center and scale 21 xyz landmarks, returning a 63-value feature vector."""
    coordinates = np.asarray(
        [[point.x, point.y, point.z] for point in landmarks],
        dtype=np.float64,
    )
    if coordinates.shape != (LANDMARK_COUNT, 3):
        raise ValueError(f"expected a ({LANDMARK_COUNT}, 3) landmark array")

    centered = coordinates - coordinates.mean(axis=0)
    scale = np.linalg.norm(centered, axis=1).max()
    if scale > 0:
        centered /= scale
    return centered.reshape(LANDMARK_COUNT * 3)
